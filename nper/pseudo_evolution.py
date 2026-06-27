import torch
import torch.nn.functional as F


class QualityAwarePseudoEvolution:
    def __init__(self, cfg):
        self.cfg = cfg

    def _local_attention_pseudo(self, attn_maps, size, device):
        if not attn_maps:
            return None, None
        masks = []
        for attn in attn_maps:
            if attn.ndim != 4 or attn.shape[-1] <= 1:
                continue
            cls_attn = attn[:, :, 0, 1:].float().to(device)
            grid = int(cls_attn.shape[-1] ** 0.5)
            if grid * grid != cls_attn.shape[-1]:
                continue
            cls_attn = cls_attn.reshape(cls_attn.shape[0], cls_attn.shape[1], grid, grid)
            flat = cls_attn.flatten(2)
            prob = flat / flat.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            entropy = -(prob * prob.clamp_min(1e-8).log()).sum(dim=-1) / torch.log(
                torch.tensor(float(flat.shape[-1]), device=device)
            )
            selected = entropy < 0.5
            if selected.sum().item() == 0:
                continue
            threshold = cls_attn.mean(dim=(-2, -1), keepdim=True) + 0.5 * cls_attn.std(
                dim=(-2, -1), keepdim=True, unbiased=False
            )
            head_masks = cls_attn > threshold
            selected = selected[:, :, None, None]
            mask = (head_masks & selected).float().sum(dim=1, keepdim=True) > 0
            masks.append(F.interpolate(mask.float(), size=(size, size), mode="nearest"))
        if not masks:
            return None, None
        local = (torch.stack(masks, dim=0).float().mean(dim=0) > 0.0).float()
        local_mask = local > 0.5
        return local, local_mask

    def evolve(self, batch, student_out, teacher_out, epoch):
        device = student_out["logits"].device
        p_init = batch["p_init"].to(device).float()
        pixel_weight = batch["pixel_weight"].to(device).float()
        anchor_fg = batch["anchor_fg"].to(device).bool()
        anchor_bg = batch["anchor_bg"].to(device).bool()
        quality = batch["quality_score"].to(device).float().view(-1, 1, 1, 1)
        hard = batch["hard_score"].to(device).float().view(-1, 1, 1, 1)

        size = int(self.cfg.LOSS_SIZE)
        if bool(getattr(self.cfg, "PSEUDO_USE_LOCAL_ATTN", True)):
            local_pseudo, local_mask = self._local_attention_pseudo(
                student_out.get("attn_maps", []), size, device
            )
        else:
            local_pseudo, local_mask = None, None

        teacher_weight = torch.zeros_like(quality)
        local_weight = torch.zeros_like(quality)
        if int(epoch) < int(getattr(self.cfg, "EVOLUTION_START_EPOCH", 4)):
            p_evo = p_init
        else:
            init_weight = torch.clamp(quality, min=float(getattr(self.cfg, "INIT_MIN_WEIGHT", 0.2)))
            parts = [p_init * init_weight]
            weights = [init_weight]
            if bool(getattr(self.cfg, "PSEUDO_USE_TEACHER", True)) and teacher_out is not None:
                teacher_prob = teacher_out["prob"].detach()
                teacher_conf = (teacher_prob - 0.5).abs().mean(dim=(1, 2, 3), keepdim=True) * 2.0
                teacher_weight = torch.minimum(
                    teacher_conf,
                    torch.full_like(teacher_conf, float(getattr(self.cfg, "TEACHER_MAX_WEIGHT", 0.7))),
                )
                parts.append(teacher_prob * teacher_weight)
                weights.append(teacher_weight)
            if (
                bool(getattr(self.cfg, "PSEUDO_USE_LOCAL_ATTN", True))
                and local_pseudo is not None
                and local_mask is not None
            ):
                local_weight = torch.full_like(quality, float(getattr(self.cfg, "LOCAL_ATTN_WEIGHT", 0.15)))
                parts.append(local_pseudo * local_weight)
                weights.append(local_weight)
            denom = torch.stack(weights, dim=0).sum(dim=0).clamp_min(1e-6)
            p_evo = torch.stack(parts, dim=0).sum(dim=0) / denom

        p_evo = p_evo.clone()
        p_evo[anchor_fg] = torch.maximum(
            p_evo[anchor_fg],
            torch.full_like(p_evo[anchor_fg], float(getattr(self.cfg, "ANCHOR_FG_TH", 0.7))),
        )
        p_evo[anchor_bg] = torch.minimum(
            p_evo[anchor_bg],
            torch.full_like(p_evo[anchor_bg], float(getattr(self.cfg, "ANCHOR_BG_TH", 0.2))),
        )
        dense_weight = pixel_weight * (1.0 - hard * (1.0 - float(getattr(self.cfg, "HARD_SAMPLE_DOWN_WEIGHT", 0.3))))
        dense_weight = dense_weight + (anchor_fg | anchor_bg).float() * float(getattr(self.cfg, "ANCHOR_WEIGHT", 0.1))
        anchor_ratio = (anchor_fg | anchor_bg).float().mean()
        local_ratio = torch.tensor(0.0, device=device)
        if local_mask is not None:
            local_ratio = local_mask.float().mean()

        return {
            "p_evo": p_evo.clamp(0.0, 1.0),
            "anchor_fg": anchor_fg,
            "anchor_bg": anchor_bg,
            "pixel_weight": dense_weight.clamp_min(0.05),
            "quality_weight": quality,
            "local_pseudo": local_pseudo,
            "local_mask": local_mask,
            "teacher_weight_mean": teacher_weight.mean(),
            "local_pseudo_ratio": local_ratio,
            "anchor_ratio": anchor_ratio,
        }
