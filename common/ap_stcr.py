import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def strict_teacher_binary(teacher_prob, threshold=0.5):
    teacher_prob = _require_unit_map("teacher_prob", teacher_prob)
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise RuntimeError(
            f"AP-STCR teacher threshold must be in [0,1], got {threshold}."
        )
    return (teacher_prob.detach().float() > threshold).float().detach()


def _canonical_manifest_hash(keys):
    payload = json.dumps(
        [[str(dataset), str(stem)] for dataset, stem in keys],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_map(name, tensor, shape=None):
    if not torch.is_tensor(tensor):
        raise TypeError(f"AP-STCR {name} must be a tensor.")
    if tensor.ndim != 4 or int(tensor.shape[1]) != 1:
        raise RuntimeError(
            f"AP-STCR {name} must be [B,1,H,W], got {list(tensor.shape)}."
        )
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise RuntimeError(
            f"AP-STCR {name} shape mismatch: "
            f"{list(tensor.shape)} != {list(shape)}."
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"AP-STCR {name} contains NaN/Inf.")
    return tensor


def _require_unit_map(name, tensor, shape=None):
    tensor = _require_map(name, tensor, shape=shape)
    if tensor.numel():
        minimum = float(tensor.min().item())
        maximum = float(tensor.max().item())
        if minimum < -1e-6 or maximum > 1.0 + 1e-6:
            raise RuntimeError(
                f"AP-STCR {name} must be in [0,1], got "
                f"{minimum:.8f}/{maximum:.8f}."
            )
    return tensor


def _stable_sorted_indices(values, candidate_mask, largest):
    flat_values = values.reshape(-1)
    candidate_indices = torch.nonzero(
        candidate_mask.reshape(-1), as_tuple=False
    ).flatten()
    if int(candidate_indices.numel()) == 0:
        return candidate_indices
    candidate_values = flat_values.index_select(0, candidate_indices)
    order = torch.argsort(
        candidate_values,
        descending=bool(largest),
        stable=True,
    )
    return candidate_indices.index_select(0, order)


def _bounded_anchor_count(candidate_count, ratio, minimum, maximum):
    candidate_count = int(candidate_count)
    if candidate_count <= 0:
        return 0
    requested = int(math.ceil(float(candidate_count) * float(ratio)))
    requested = max(int(minimum), requested)
    requested = min(int(maximum), requested, candidate_count)
    return requested


class APSTCRSemanticCache:
    def __init__(self, keys, height=37, width=37):
        self.keys = [(str(dataset), str(stem)) for dataset, stem in keys]
        self.manifest_hash = _canonical_manifest_hash(self.keys)
        self.height = int(height)
        self.width = int(width)
        num_samples = len(self.keys)
        self.semantic_margin = torch.zeros(
            (num_samples, 1, self.height, self.width),
            dtype=torch.float16,
            device="cpu",
        )
        self.fg_anchor_mask = torch.zeros(
            (num_samples, 1, self.height, self.width),
            dtype=torch.bool,
            device="cpu",
        )
        self.bg_anchor_mask = torch.zeros_like(self.fg_anchor_mask)
        self.valid = torch.zeros(num_samples, dtype=torch.bool, device="cpu")
        # 0=dabe_seed:bg_anchor_37, 1=pseudo_rank_fallback.
        self.bg_source = torch.full(
            (num_samples,), -1, dtype=torch.int8, device="cpu"
        )

    def _indices(self, sample_indices, datasets, stems):
        indices = torch.as_tensor(
            sample_indices, dtype=torch.long, device="cpu"
        ).flatten()
        if len(datasets) != int(indices.numel()) or len(stems) != int(
            indices.numel()
        ):
            raise RuntimeError("AP-STCR semantic-cache identity length mismatch.")
        if indices.numel() and (
            int(indices.min()) < 0 or int(indices.max()) >= len(self.keys)
        ):
            raise RuntimeError("AP-STCR sample_index is outside semantic cache.")
        for local_index, sample_index in enumerate(indices.tolist()):
            expected = self.keys[int(sample_index)]
            actual = (str(datasets[local_index]), str(stems[local_index]))
            if expected != actual:
                raise RuntimeError(
                    "AP-STCR sample identity mismatch: "
                    f"index={sample_index}, {actual} != {expected}."
                )
        return indices

    def fetch(self, sample_indices, datasets, stems, device):
        indices = self._indices(sample_indices, datasets, stems)
        source_codes = self.bg_source.index_select(0, indices).tolist()
        return {
            "indices": indices,
            "valid": self.valid.index_select(0, indices),
            "semantic_margin": self.semantic_margin.index_select(
                0, indices
            ).to(device=device, dtype=torch.float32),
            "fg_anchor_mask": self.fg_anchor_mask.index_select(
                0, indices
            ).to(device=device),
            "bg_anchor_mask": self.bg_anchor_mask.index_select(
                0, indices
            ).to(device=device),
            "bg_anchor_source": [
                (
                    "dabe_seed:bg_anchor_37"
                    if int(code) == 0
                    else "pseudo_rank_fallback"
                )
                for code in source_codes
            ],
        }

    def store(
        self,
        sample_indices,
        datasets,
        stems,
        semantic_margin,
        fg_anchor_mask,
        bg_anchor_mask,
        bg_anchor_source,
    ):
        indices = self._indices(sample_indices, datasets, stems)
        batch_size = int(indices.numel())
        expected = (batch_size, 1, self.height, self.width)
        semantic_margin = _require_map(
            "semantic_margin", semantic_margin, shape=expected
        ).detach()
        if semantic_margin.numel() and (
            float(semantic_margin.min()) < -1.0 - 1e-6
            or float(semantic_margin.max()) > 1.0 + 1e-6
        ):
            raise RuntimeError("AP-STCR semantic margin must be in [-1,1].")
        fg_anchor_mask = _require_map(
            "fg_anchor_mask", fg_anchor_mask, shape=expected
        ).detach().bool()
        bg_anchor_mask = _require_map(
            "bg_anchor_mask", bg_anchor_mask, shape=expected
        ).detach().bool()
        if bool((fg_anchor_mask & bg_anchor_mask).any().item()):
            raise RuntimeError("AP-STCR foreground/background anchors overlap.")
        if len(bg_anchor_source) != batch_size:
            raise RuntimeError("AP-STCR background source length mismatch.")
        source_codes = []
        for source in bg_anchor_source:
            if source == "dabe_seed:bg_anchor_37":
                source_codes.append(0)
            elif source == "pseudo_rank_fallback":
                source_codes.append(1)
            else:
                raise RuntimeError(
                    f"Unsupported AP-STCR background source: {source!r}."
                )
        cpu_margin = semantic_margin.to(
            device="cpu", dtype=torch.float16
        )
        cpu_fg = fg_anchor_mask.to(device="cpu")
        cpu_bg = bg_anchor_mask.to(device="cpu")
        for local_index, sample_index in enumerate(indices.tolist()):
            sample_index = int(sample_index)
            if bool(self.valid[sample_index]):
                if not torch.equal(
                    self.semantic_margin[sample_index],
                    cpu_margin[local_index],
                ):
                    raise RuntimeError(
                        "AP-STCR attempted to change cached semantic margin."
                    )
                continue
            self.semantic_margin[sample_index].copy_(cpu_margin[local_index])
            self.fg_anchor_mask[sample_index].copy_(cpu_fg[local_index])
            self.bg_anchor_mask[sample_index].copy_(cpu_bg[local_index])
            self.bg_source[sample_index] = int(source_codes[local_index])
            self.valid[sample_index] = True

    def state_dict(self):
        return {
            "schema_version": "ap_stcr_semantic_cache_v1",
            "manifest_hash": self.manifest_hash,
            "height": self.height,
            "width": self.width,
            "semantic_margin": self.semantic_margin.clone(),
            "fg_anchor_mask": self.fg_anchor_mask.clone(),
            "bg_anchor_mask": self.bg_anchor_mask.clone(),
            "valid": self.valid.clone(),
            "bg_source": self.bg_source.clone(),
        }

    def load_state_dict(self, state):
        if state.get("schema_version") != "ap_stcr_semantic_cache_v1":
            raise RuntimeError("AP-STCR semantic cache schema mismatch.")
        if state.get("manifest_hash") != self.manifest_hash:
            raise RuntimeError("AP-STCR semantic cache manifest mismatch.")
        expected_shapes = {
            "semantic_margin": self.semantic_margin.shape,
            "fg_anchor_mask": self.fg_anchor_mask.shape,
            "bg_anchor_mask": self.bg_anchor_mask.shape,
            "valid": self.valid.shape,
            "bg_source": self.bg_source.shape,
        }
        for name, shape in expected_shapes.items():
            value = state.get(name)
            if not torch.is_tensor(value) or tuple(value.shape) != tuple(shape):
                raise RuntimeError(
                    f"AP-STCR semantic cache field mismatch: {name}."
                )
            getattr(self, name).copy_(
                value.to(dtype=getattr(self, name).dtype, device="cpu")
            )


class APSTCRTemporalHistoryBank:
    def __init__(self, keys, window=3, height=37, width=37):
        self.keys = [(str(dataset), str(stem)) for dataset, stem in keys]
        self.manifest_hash = _canonical_manifest_hash(self.keys)
        self.window = int(window)
        self.height = int(height)
        self.width = int(width)
        if self.window < 1:
            raise RuntimeError("AP-STCR temporal window must be positive.")
        num_samples = len(self.keys)
        self.history = torch.zeros(
            (num_samples, self.window, 1, self.height, self.width),
            dtype=torch.float16,
            device="cpu",
        )
        self.epoch_tags = torch.full(
            (num_samples, self.window),
            -1,
            dtype=torch.int16,
            device="cpu",
        )
        self.write_position = torch.zeros(
            num_samples, dtype=torch.int16, device="cpu"
        )
        self.count = torch.zeros(
            num_samples, dtype=torch.int16, device="cpu"
        )
        self.last_write_epoch = torch.full(
            (num_samples,), -1, dtype=torch.int16, device="cpu"
        )

    def _indices(self, sample_indices, datasets, stems):
        indices = torch.as_tensor(
            sample_indices, dtype=torch.long, device="cpu"
        ).flatten()
        if len(datasets) != int(indices.numel()) or len(stems) != int(
            indices.numel()
        ):
            raise RuntimeError("AP-STCR history identity length mismatch.")
        if indices.numel() and (
            int(indices.min()) < 0 or int(indices.max()) >= len(self.keys)
        ):
            raise RuntimeError("AP-STCR sample_index is outside history bank.")
        for local_index, sample_index in enumerate(indices.tolist()):
            expected = self.keys[int(sample_index)]
            actual = (str(datasets[local_index]), str(stems[local_index]))
            if expected != actual:
                raise RuntimeError(
                    "AP-STCR history sample identity mismatch: "
                    f"index={sample_index}, {actual} != {expected}."
                )
        return indices

    def fetch(self, sample_indices, datasets, stems, device):
        indices = self._indices(sample_indices, datasets, stems)
        values = self.history.index_select(0, indices).to(
            device=device, dtype=torch.float32
        )
        tags = self.epoch_tags.index_select(0, indices).to(device=device)
        valid_slots = tags >= 0
        weights = valid_slots[:, :, None, None, None].float()
        denominator = weights.sum(dim=1).clamp_min(1.0)
        mean = (values * weights).sum(dim=1) / denominator
        counts = valid_slots.sum(dim=1)
        valid_images = counts > 0
        return {
            "history_mean": mean.detach(),
            "history_count": counts.detach(),
            "history_valid": valid_images.detach(),
            "epoch_tags": tags.detach(),
        }

    def update(
        self,
        sample_indices,
        datasets,
        stems,
        teacher_soft_37,
        epoch,
    ):
        indices = self._indices(sample_indices, datasets, stems)
        batch_size = int(indices.numel())
        expected = (batch_size, 1, self.height, self.width)
        teacher_soft_37 = _require_unit_map(
            "teacher_soft_37", teacher_soft_37, shape=expected
        ).detach().to(device="cpu", dtype=torch.float16)
        epoch = int(epoch)
        if epoch < 1 or epoch > 32767:
            raise RuntimeError(f"Invalid AP-STCR history epoch: {epoch}.")
        for local_index, sample_index in enumerate(indices.tolist()):
            sample_index = int(sample_index)
            if int(self.last_write_epoch[sample_index]) == epoch:
                raise RuntimeError(
                    "AP-STCR history received the same sample twice in one "
                    f"epoch: index={sample_index}, epoch={epoch}."
                )
            position = int(self.write_position[sample_index])
            self.history[sample_index, position].copy_(
                teacher_soft_37[local_index]
            )
            self.epoch_tags[sample_index, position] = epoch
            self.write_position[sample_index] = (position + 1) % self.window
            self.count[sample_index] = min(
                self.window, int(self.count[sample_index]) + 1
            )
            self.last_write_epoch[sample_index] = epoch

    def clear(self):
        self.history.zero_()
        self.epoch_tags.fill_(-1)
        self.write_position.zero_()
        self.count.zero_()
        self.last_write_epoch.fill_(-1)

    def state_dict(self):
        return {
            "schema_version": "ap_stcr_history_v1",
            "manifest_hash": self.manifest_hash,
            "window": self.window,
            "height": self.height,
            "width": self.width,
            "history": self.history.clone(),
            "epoch_tags": self.epoch_tags.clone(),
            "write_position": self.write_position.clone(),
            "count": self.count.clone(),
            "last_write_epoch": self.last_write_epoch.clone(),
        }

    def load_state_dict(self, state):
        if state.get("schema_version") != "ap_stcr_history_v1":
            raise RuntimeError("AP-STCR history schema mismatch.")
        if state.get("manifest_hash") != self.manifest_hash:
            raise RuntimeError("AP-STCR history manifest mismatch.")
        for name in (
            "history",
            "epoch_tags",
            "write_position",
            "count",
            "last_write_epoch",
        ):
            value = state.get(name)
            target = getattr(self, name)
            if not torch.is_tensor(value) or tuple(value.shape) != tuple(
                target.shape
            ):
                raise RuntimeError(f"AP-STCR history field mismatch: {name}.")
            target.copy_(value.to(device="cpu", dtype=target.dtype))


class AnchorPropagatedSemanticTemporalCorrection:
    def __init__(self, config, sample_keys):
        self.config = dict(config)
        self.enabled = bool(self.config.get("enabled", True))
        self.evidence_resolution = int(
            self.config.get("evidence_resolution", 37)
        )
        self.loss_resolution = int(self.config.get("loss_resolution", 68))
        self.fg_anchor_ratio = float(
            self.config.get("fg_anchor_ratio", 0.20)
        )
        self.bg_anchor_ratio = float(
            self.config.get("bg_anchor_ratio", 0.20)
        )
        self.min_fg_anchors = int(self.config.get("min_fg_anchors", 4))
        self.min_bg_anchors = int(self.config.get("min_bg_anchors", 4))
        self.max_fg_anchors = int(self.config.get("max_fg_anchors", 64))
        self.max_bg_anchors = int(self.config.get("max_bg_anchors", 64))
        self.prefer_dabe_background_seed = bool(
            self.config.get("prefer_dabe_background_seed", True)
        )
        self.tau_delta = float(self.config.get("tau_delta", 0.25))
        self.tau_margin = float(self.config.get("tau_margin", 0.50))
        self.temporal_window = int(
            self.config.get("temporal_window", 3)
        )
        self.tau_temporal = float(
            self.config.get("tau_temporal", 0.20)
        )
        self.lambda_semantic = float(
            self.config.get("lambda_semantic", 1.0)
        )
        self.lambda_temporal = float(
            self.config.get("lambda_temporal", 1.0)
        )
        self.eps = float(self.config.get("eps", 1e-6))
        if not self.enabled:
            raise RuntimeError("AP-STCR module requires enabled=True.")
        positive_values = {
            "fg_anchor_ratio": self.fg_anchor_ratio,
            "bg_anchor_ratio": self.bg_anchor_ratio,
            "tau_delta": self.tau_delta,
            "tau_margin": self.tau_margin,
            "tau_temporal": self.tau_temporal,
            "lambda_semantic": self.lambda_semantic,
            "lambda_temporal": self.lambda_temporal,
            "eps": self.eps,
        }
        invalid = {
            name: value
            for name, value in positive_values.items()
            if not math.isfinite(value) or value <= 0.0
        }
        if invalid:
            raise RuntimeError(
                f"Invalid AP-STCR positive configuration values: {invalid}."
            )
        if self.evidence_resolution != 37 or self.loss_resolution != 68:
            raise RuntimeError("AP-STCR v1 requires 37 evidence and 68 loss.")
        self.semantic_cache = APSTCRSemanticCache(
            sample_keys,
            height=self.evidence_resolution,
            width=self.evidence_resolution,
        )
        self.history_bank = APSTCRTemporalHistoryBank(
            sample_keys,
            window=self.temporal_window,
            height=self.evidence_resolution,
            width=self.evidence_resolution,
        )
        self.manifest_hash = self.semantic_cache.manifest_hash
        if self.history_bank.manifest_hash != self.manifest_hash:
            raise RuntimeError("AP-STCR cache/history manifest mismatch.")

    @torch.no_grad()
    def build_anchor_masks(
        self,
        fixed_pseudo,
        dabe_background_seed=None,
    ):
        fixed_pseudo = _require_unit_map(
            "fixed_pseudo_37", fixed_pseudo
        ).detach().float()
        if dabe_background_seed is not None:
            dabe_background_seed = _require_unit_map(
                "dabe_background_seed",
                dabe_background_seed,
                shape=fixed_pseudo.shape,
            ).detach()
        batch_size = int(fixed_pseudo.shape[0])
        fg_masks = torch.zeros_like(fixed_pseudo, dtype=torch.bool)
        bg_masks = torch.zeros_like(fixed_pseudo, dtype=torch.bool)
        bg_sources = []
        for image_index in range(batch_size):
            p0 = fixed_pseudo[image_index, 0]
            fg_candidates = p0 > 0.5
            fg_count = int(fg_candidates.sum().item())
            if fg_count >= self.min_fg_anchors:
                fg_sorted = _stable_sorted_indices(
                    p0, fg_candidates, largest=True
                )
                fg_keep = _bounded_anchor_count(
                    fg_count,
                    self.fg_anchor_ratio,
                    self.min_fg_anchors,
                    self.max_fg_anchors,
                )
            else:
                fg_sorted = _stable_sorted_indices(
                    p0,
                    torch.ones_like(fg_candidates),
                    largest=True,
                )
                fg_keep = min(self.min_fg_anchors, int(p0.numel()))
            fg_masks[image_index, 0].view(-1)[
                fg_sorted[:fg_keep]
            ] = True

            dabe_candidates = torch.zeros_like(fg_candidates)
            if (
                self.prefer_dabe_background_seed
                and dabe_background_seed is not None
            ):
                dabe_candidates = (
                    dabe_background_seed[image_index, 0] > 0.5
                ) & (~fg_masks[image_index, 0])
            bg_count = int(dabe_candidates.sum().item())
            if bg_count >= self.min_bg_anchors:
                bg_candidates = dabe_candidates
                bg_sources.append("dabe_seed:bg_anchor_37")
            else:
                bg_candidates = (p0 < 0.5) & (
                    ~fg_masks[image_index, 0]
                )
                if int(bg_candidates.sum().item()) < self.min_bg_anchors:
                    bg_candidates = ~fg_masks[image_index, 0]
                bg_count = int(bg_candidates.sum().item())
                bg_sources.append("pseudo_rank_fallback")
            bg_sorted = _stable_sorted_indices(
                p0, bg_candidates, largest=False
            )
            bg_keep = _bounded_anchor_count(
                bg_count,
                self.bg_anchor_ratio,
                self.min_bg_anchors,
                self.max_bg_anchors,
            )
            if bg_keep < self.min_bg_anchors:
                raise RuntimeError(
                    "AP-STCR could not construct the minimum background "
                    "anchor count."
                )
            bg_masks[image_index, 0].view(-1)[
                bg_sorted[:bg_keep]
            ] = True
        if bool((fg_masks & bg_masks).any().item()):
            raise RuntimeError("AP-STCR anchors must be mutually exclusive.")
        return fg_masks.detach(), bg_masks.detach(), bg_sources

    @torch.no_grad()
    def build_prototypes(
        self,
        dino_features,
        fg_anchor_mask,
        bg_anchor_mask,
    ):
        if (
            not torch.is_tensor(dino_features)
            or dino_features.ndim != 4
            or int(dino_features.shape[1]) != 384
            or tuple(dino_features.shape[-2:])
            != (self.evidence_resolution, self.evidence_resolution)
        ):
            raise RuntimeError(
                "AP-STCR requires cached DINO [B,384,37,37], got "
                f"{list(dino_features.shape)}."
            )
        features = F.normalize(
            dino_features.detach().float(), dim=1, p=2, eps=self.eps
        )
        fg = fg_anchor_mask.detach().float()
        bg = bg_anchor_mask.detach().float()
        fg_prototype = (features * fg).flatten(2).sum(dim=2) / (
            fg.flatten(2).sum(dim=2) + self.eps
        )
        bg_prototype = (features * bg).flatten(2).sum(dim=2) / (
            bg.flatten(2).sum(dim=2) + self.eps
        )
        fg_prototype = F.normalize(
            fg_prototype, dim=1, p=2, eps=self.eps
        )
        bg_prototype = F.normalize(
            bg_prototype, dim=1, p=2, eps=self.eps
        )
        return features, fg_prototype.detach(), bg_prototype.detach()

    @torch.no_grad()
    def compute_semantic_margin(
        self,
        dino_features,
        fg_prototype,
        bg_prototype,
    ):
        similarity_fg = (
            dino_features * fg_prototype[:, :, None, None]
        ).sum(dim=1, keepdim=True)
        similarity_bg = (
            dino_features * bg_prototype[:, :, None, None]
        ).sum(dim=1, keepdim=True)
        margin = similarity_fg - similarity_bg
        max_abs = margin.abs().flatten(1).amax(dim=1).view(-1, 1, 1, 1)
        normalized = margin / (max_abs + self.eps)
        return normalized.clamp(-1.0, 1.0).detach()

    @torch.no_grad()
    def _semantic_evidence(
        self,
        dino_features,
        fixed_pseudo_37,
        dabe_background_seed_37,
        sample_indices,
        datasets,
        stems,
    ):
        device = dino_features.device
        cached = self.semantic_cache.fetch(
            sample_indices, datasets, stems, device=device
        )
        missing = torch.nonzero(
            ~cached["valid"], as_tuple=False
        ).flatten()
        if int(missing.numel()) > 0:
            selected = missing.to(device=device)
            fg, bg, sources = self.build_anchor_masks(
                fixed_pseudo_37.index_select(0, selected),
                dabe_background_seed_37.index_select(0, selected),
            )
            features, fg_proto, bg_proto = self.build_prototypes(
                dino_features.index_select(0, selected),
                fg,
                bg,
            )
            margin = self.compute_semantic_margin(
                features, fg_proto, bg_proto
            )
            missing_cpu = missing.to(device="cpu")
            all_indices = torch.as_tensor(
                sample_indices, dtype=torch.long, device="cpu"
            ).flatten()
            self.semantic_cache.store(
                all_indices.index_select(0, missing_cpu),
                [datasets[int(i)] for i in missing_cpu.tolist()],
                [stems[int(i)] for i in missing_cpu.tolist()],
                margin,
                fg,
                bg,
                sources,
            )
            cached = self.semantic_cache.fetch(
                sample_indices, datasets, stems, device=device
            )
        if not bool(cached["valid"].all().item()):
            raise RuntimeError("AP-STCR semantic cache remained incomplete.")
        return cached

    @torch.no_grad()
    def compute_semantic_support(
        self,
        semantic_margin,
        teacher_binary,
        fixed_pseudo,
    ):
        delta = teacher_binary.detach().float() - fixed_pseudo.detach().float()
        correction_direction = torch.tanh(delta / self.tau_delta)
        semantic_direction = torch.tanh(
            semantic_margin.detach().float() / self.tau_margin
        )
        support = 0.5 * (
            1.0 + semantic_direction * correction_direction
        )
        return support.clamp(0.0, 1.0).detach(), delta.detach()

    @torch.no_grad()
    def compute_temporal_support(
        self,
        teacher_soft,
        fixed_pseudo,
        sample_indices,
        datasets,
        stems,
    ):
        history = self.history_bank.fetch(
            sample_indices,
            datasets,
            stems,
            device=teacher_soft.device,
        )
        history_mean = history["history_mean"]
        current_delta = teacher_soft.detach().float() - fixed_pseudo.detach().float()
        history_delta = history_mean - fixed_pseudo.detach().float()
        direction_support = 0.5 * (
            1.0
            + torch.tanh(current_delta / self.tau_delta)
            * torch.tanh(history_delta / self.tau_delta)
        )
        deviation_support = torch.exp(
            -(teacher_soft.detach().float() - history_mean).abs()
            / self.tau_temporal
        )
        support = (direction_support * deviation_support).clamp(0.0, 1.0)
        valid = history["history_valid"][:, None, None, None]
        support = torch.where(valid, support, torch.ones_like(support))
        history.update(
            {
                "temporal_support": support.detach(),
                "direction_support": direction_support.detach(),
                "deviation_support": deviation_support.detach(),
            }
        )
        return history

    @torch.no_grad()
    def compute_local_acceptance(
        self,
        semantic_support,
        temporal_support,
    ):
        evidence = (
            self.lambda_semantic * (1.0 - semantic_support.detach().float())
            + self.lambda_temporal * (1.0 - temporal_support.detach().float())
        )
        return (1.0 / (1.0 + evidence)).clamp(0.0, 1.0).detach()

    @torch.no_grad()
    def build_target(
        self,
        fixed_pseudo_68,
        teacher_binary_68,
        global_teacher_ratio,
        local_acceptance_37,
    ):
        fixed_pseudo_68 = _require_unit_map(
            "fixed_pseudo_68", fixed_pseudo_68
        ).detach().float()
        teacher_binary_68 = _require_unit_map(
            "teacher_binary_68",
            teacher_binary_68,
            shape=fixed_pseudo_68.shape,
        ).detach().float()
        alpha = float(global_teacher_ratio)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise RuntimeError(
                f"AP-STCR global teacher ratio must be in [0,1], got {alpha}."
            )
        local_acceptance_68 = F.interpolate(
            local_acceptance_37.detach().float(),
            size=(self.loss_resolution, self.loss_resolution),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        effective_teacher_weight = (
            alpha * local_acceptance_68
        ).clamp(0.0, 1.0)
        target = (
            (1.0 - effective_teacher_weight) * fixed_pseudo_68
            + effective_teacher_weight * teacher_binary_68
        ).clamp(0.0, 1.0)
        return {
            "local_acceptance_68": local_acceptance_68.detach(),
            "effective_teacher_weight_68": effective_teacher_weight.detach(),
            "fixed_weight_68": (1.0 - effective_teacher_weight).detach(),
            "mixed_target_68": target.detach(),
            "global_teacher_ratio": alpha,
        }

    @torch.no_grad()
    def build_batch(
        self,
        dino_features,
        fixed_pseudo_37,
        fixed_pseudo_68,
        dabe_background_seed_37,
        teacher_soft_68,
        teacher_binary_68,
        global_teacher_ratio,
        sample_indices,
        datasets,
        stems,
    ):
        fixed_pseudo_37 = _require_unit_map(
            "fixed_pseudo_37", fixed_pseudo_37
        ).detach().float()
        fixed_pseudo_68 = _require_unit_map(
            "fixed_pseudo_68", fixed_pseudo_68
        ).detach().float()
        teacher_soft_68 = _require_unit_map(
            "teacher_soft_68", teacher_soft_68, shape=fixed_pseudo_68.shape
        ).detach().float()
        teacher_binary_68 = _require_unit_map(
            "teacher_binary_68",
            teacher_binary_68,
            shape=fixed_pseudo_68.shape,
        ).detach().float()
        background = _require_unit_map(
            "dabe_background_seed_37",
            dabe_background_seed_37,
            shape=fixed_pseudo_37.shape,
        ).detach().float()
        semantic = self._semantic_evidence(
            dino_features.detach(),
            fixed_pseudo_37,
            background,
            sample_indices,
            datasets,
            stems,
        )
        teacher_soft_37 = F.interpolate(
            teacher_soft_68,
            size=(self.evidence_resolution, self.evidence_resolution),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        teacher_binary_37 = F.interpolate(
            teacher_binary_68,
            size=(self.evidence_resolution, self.evidence_resolution),
            mode="nearest",
        )
        semantic_support, correction_37 = self.compute_semantic_support(
            semantic["semantic_margin"],
            teacher_binary_37,
            fixed_pseudo_37,
        )
        temporal = self.compute_temporal_support(
            teacher_soft_37,
            fixed_pseudo_37,
            sample_indices,
            datasets,
            stems,
        )
        local_acceptance = self.compute_local_acceptance(
            semantic_support,
            temporal["temporal_support"],
        )
        target = self.build_target(
            fixed_pseudo_68,
            teacher_binary_68,
            global_teacher_ratio,
            local_acceptance,
        )
        result = {
            "sample_indices": torch.as_tensor(
                sample_indices, dtype=torch.long, device="cpu"
            ).flatten(),
            "datasets": [str(value) for value in datasets],
            "stems": [str(value) for value in stems],
            "fixed_pseudo_37": fixed_pseudo_37.detach(),
            "fixed_pseudo_68": fixed_pseudo_68.detach(),
            "teacher_soft_37": teacher_soft_37.detach(),
            "teacher_soft_68": teacher_soft_68.detach(),
            "teacher_binary_37": teacher_binary_37.detach(),
            "teacher_binary_68": teacher_binary_68.detach(),
            "teacher_correction_37": correction_37.detach(),
            "teacher_correction_68": (
                teacher_binary_68 - fixed_pseudo_68
            ).detach(),
            "semantic_margin_37": semantic["semantic_margin"].detach(),
            "fg_anchor_mask_37": semantic["fg_anchor_mask"].detach(),
            "bg_anchor_mask_37": semantic["bg_anchor_mask"].detach(),
            "bg_anchor_source": list(semantic["bg_anchor_source"]),
            "semantic_support_37": semantic_support.detach(),
            "temporal_support_37": temporal["temporal_support"].detach(),
            "history_mean_37": temporal["history_mean"].detach(),
            "history_count": temporal["history_count"].detach(),
            "history_valid": temporal["history_valid"].detach(),
            "local_acceptance_37": local_acceptance.detach(),
        }
        result.update(target)
        for name, value in result.items():
            if torch.is_tensor(value) and value.requires_grad:
                raise RuntimeError(
                    f"AP-STCR output {name} must be detached."
                )
        return result

    @torch.no_grad()
    def update_history(
        self,
        sample_indices,
        datasets,
        stems,
        teacher_soft_37,
        epoch,
    ):
        self.history_bank.update(
            sample_indices,
            datasets,
            stems,
            teacher_soft_37,
            epoch,
        )

    def clear_temporal_history(self):
        self.history_bank.clear()

    def state_dict(self):
        return {
            "schema_version": "ap_stcr_runtime_v1",
            "manifest_hash": self.manifest_hash,
            "semantic_cache": self.semantic_cache.state_dict(),
            "history_bank": self.history_bank.state_dict(),
        }

    def load_state_dict(self, state):
        if state.get("schema_version") != "ap_stcr_runtime_v1":
            raise RuntimeError("AP-STCR runtime schema mismatch.")
        if state.get("manifest_hash") != self.manifest_hash:
            raise RuntimeError("AP-STCR runtime manifest mismatch.")
        self.semantic_cache.load_state_dict(state["semantic_cache"])
        self.history_bank.load_state_dict(state["history_bank"])


def _new_moments():
    return {"sum": 0.0, "square_sum": 0.0, "count": 0}


def _add_moments(accumulator, tensor, mask=None):
    values = tensor.detach().float()
    if mask is not None:
        values = values[mask.detach().bool()]
    else:
        values = values.reshape(-1)
    if int(values.numel()) == 0:
        return
    values = values.double()
    accumulator["sum"] += float(values.sum().item())
    accumulator["square_sum"] += float(values.square().sum().item())
    accumulator["count"] += int(values.numel())


def _finalize_moments(value):
    count = int(value["count"])
    if count <= 0:
        return float("nan"), float("nan"), False
    mean = float(value["sum"]) / float(count)
    variance = max(
        0.0, float(value["square_sum"]) / float(count) - mean * mean
    )
    return mean, math.sqrt(variance), True


def _add_histogram(histogram, tensor):
    values = tensor.detach().float().clamp(0.0, 1.0).reshape(-1).cpu()
    if int(values.numel()) == 0:
        return
    histogram += torch.histc(
        values,
        bins=int(histogram.numel()),
        min=0.0,
        max=1.0,
    ).double()


def _histogram_quantile(histogram, quantile):
    total = float(histogram.sum().item())
    if total <= 0.0:
        return float("nan")
    target = float(quantile) * total
    index = int(
        torch.searchsorted(
            histogram.cumsum(dim=0),
            torch.tensor(target, dtype=torch.float64),
        ).item()
    )
    index = min(max(index, 0), int(histogram.numel()) - 1)
    return float(index) / float(max(1, int(histogram.numel()) - 1))


def new_ap_stcr_epoch_accumulator(histogram_bins=1000):
    moment_names = (
        "semantic_margin",
        "semantic_support",
        "temporal_support",
        "local_acceptance",
        "effective_teacher_weight",
        "fixed_weight",
        "teacher_correction_abs",
        "accepted_correction_abs",
        "target_area",
        "fixed_area",
        "teacher_area",
        "student_area",
        "semantic_support_agree",
        "semantic_support_disagree",
        "temporal_support_agree",
        "temporal_support_disagree",
        "effective_teacher_weight_agree",
        "effective_teacher_weight_disagree",
    )
    return {
        "moments": {name: _new_moments() for name in moment_names},
        "histograms": {
            name: torch.zeros(histogram_bins, dtype=torch.float64)
            for name in (
                "semantic_support",
                "temporal_support",
                "local_acceptance",
            )
        },
        "fg_anchor_count_sum": 0.0,
        "bg_anchor_count_sum": 0.0,
        "image_count": 0,
        "history_valid_images": 0,
        "disagreement_pixels": 0,
        "pixel_count": 0,
        "bg_sources": {},
        "global_teacher_ratio_sum": 0.0,
        "batch_count": 0,
    }


def accumulate_ap_stcr_epoch(accumulator, result, student_prob_68):
    batch_size = int(result["fixed_pseudo_68"].shape[0])
    fixed_binary = result["fixed_pseudo_68"] > 0.5
    teacher_binary = result["teacher_binary_68"] > 0.5
    disagree = fixed_binary != teacher_binary
    agree = ~disagree
    accepted_correction = (
        result["effective_teacher_weight_68"]
        * result["teacher_correction_68"]
    )
    values = {
        "semantic_margin": result["semantic_margin_37"],
        "semantic_support": result["semantic_support_37"],
        "temporal_support": result["temporal_support_37"],
        "local_acceptance": result["local_acceptance_37"],
        "effective_teacher_weight": result[
            "effective_teacher_weight_68"
        ],
        "fixed_weight": result["fixed_weight_68"],
        "teacher_correction_abs": result["teacher_correction_68"].abs(),
        "accepted_correction_abs": accepted_correction.abs(),
        "target_area": result["mixed_target_68"],
        "fixed_area": result["fixed_pseudo_68"],
        "teacher_area": result["teacher_binary_68"],
        "student_area": student_prob_68.detach(),
    }
    for name, value in values.items():
        _add_moments(accumulator["moments"][name], value)
    conditional = {
        "semantic_support_agree": (
            F.interpolate(
                result["semantic_support_37"],
                size=(68, 68),
                mode="bilinear",
                align_corners=False,
            ),
            agree,
        ),
        "semantic_support_disagree": (
            F.interpolate(
                result["semantic_support_37"],
                size=(68, 68),
                mode="bilinear",
                align_corners=False,
            ),
            disagree,
        ),
        "temporal_support_agree": (
            F.interpolate(
                result["temporal_support_37"],
                size=(68, 68),
                mode="bilinear",
                align_corners=False,
            ),
            agree,
        ),
        "temporal_support_disagree": (
            F.interpolate(
                result["temporal_support_37"],
                size=(68, 68),
                mode="bilinear",
                align_corners=False,
            ),
            disagree,
        ),
        "effective_teacher_weight_agree": (
            result["effective_teacher_weight_68"],
            agree,
        ),
        "effective_teacher_weight_disagree": (
            result["effective_teacher_weight_68"],
            disagree,
        ),
    }
    for name, (value, mask) in conditional.items():
        _add_moments(accumulator["moments"][name], value, mask=mask)
    for name in accumulator["histograms"]:
        _add_histogram(
            accumulator["histograms"][name],
            result[f"{name}_37"],
        )
    accumulator["fg_anchor_count_sum"] += float(
        result["fg_anchor_mask_37"].flatten(1).sum(dim=1).sum().item()
    )
    accumulator["bg_anchor_count_sum"] += float(
        result["bg_anchor_mask_37"].flatten(1).sum(dim=1).sum().item()
    )
    accumulator["image_count"] += batch_size
    accumulator["history_valid_images"] += int(
        result["history_valid"].sum().item()
    )
    accumulator["disagreement_pixels"] += int(disagree.sum().item())
    accumulator["pixel_count"] += int(disagree.numel())
    for source in result["bg_anchor_source"]:
        accumulator["bg_sources"][source] = (
            int(accumulator["bg_sources"].get(source, 0)) + 1
        )
    accumulator["global_teacher_ratio_sum"] += float(
        result["global_teacher_ratio"]
    )
    accumulator["batch_count"] += 1


def finalize_ap_stcr_epoch(accumulator):
    result = {}
    for name, value in accumulator["moments"].items():
        mean, std, valid = _finalize_moments(value)
        result[f"{name}_mean"] = mean
        result[f"{name}_std"] = std
        result[f"{name}_valid"] = valid
    for name, histogram in accumulator["histograms"].items():
        for label, quantile in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
            result[f"{name}_{label}"] = _histogram_quantile(
                histogram, quantile
            )
    image_count = max(1, int(accumulator["image_count"]))
    pixel_count = max(1, int(accumulator["pixel_count"]))
    batch_count = max(1, int(accumulator["batch_count"]))
    result.update(
        {
            "fg_anchor_count_mean": float(
                accumulator["fg_anchor_count_sum"]
            )
            / image_count,
            "bg_anchor_count_mean": float(
                accumulator["bg_anchor_count_sum"]
            )
            / image_count,
            "bg_anchor_source": dict(accumulator["bg_sources"]),
            "history_valid_ratio": float(
                accumulator["history_valid_images"]
            )
            / image_count,
            "teacher_fixed_disagreement_ratio": float(
                accumulator["disagreement_pixels"]
            )
            / pixel_count,
            "global_teacher_ratio": float(
                accumulator["global_teacher_ratio_sum"]
            )
            / batch_count,
        }
    )
    return result


def build_ap_stcr_diagnostic_payload(
    epoch,
    local_index,
    batch,
    image_68,
    student_prob_68,
    result,
):
    index = int(local_index)
    resize_bilinear = lambda value: F.interpolate(
        value[index : index + 1].detach().float(),
        size=(68, 68),
        mode="bilinear",
        align_corners=False,
    )[0].cpu()
    resize_nearest = lambda value: F.interpolate(
        value[index : index + 1].detach().float(),
        size=(68, 68),
        mode="nearest",
    )[0].cpu()
    return {
        "schema_version": "ap_stcr_diagnostic_v1",
        "epoch": int(epoch),
        "sample_index": int(batch["sample_index"][index]),
        "dataset": str(batch["dataset"][index]),
        "stem": str(batch["stem"][index]),
        "image_path": str(batch["image_path"][index]),
        "bg_anchor_source": str(result["bg_anchor_source"][index]),
        "rgb": image_68[index].detach().float().cpu().clamp(0.0, 1.0),
        "fixed_pseudo": result["fixed_pseudo_68"][index].cpu(),
        "fg_anchor_mask": resize_nearest(result["fg_anchor_mask_37"]),
        "bg_anchor_mask": resize_nearest(result["bg_anchor_mask_37"]),
        "semantic_margin": resize_bilinear(result["semantic_margin_37"]),
        "teacher_binary": result["teacher_binary_68"][index].cpu(),
        "teacher_correction": result["teacher_correction_68"][index].cpu(),
        "semantic_support": resize_bilinear(
            result["semantic_support_37"]
        ),
        "temporal_support": resize_bilinear(
            result["temporal_support_37"]
        ),
        "local_acceptance": resize_bilinear(
            result["local_acceptance_37"]
        ),
        "effective_teacher_weight": result[
            "effective_teacher_weight_68"
        ][index].cpu(),
        "mixed_target": result["mixed_target_68"][index].cpu(),
        "student_prediction": student_prob_68[index]
        .detach()
        .float()
        .cpu(),
    }


def render_ap_stcr_diagnostic(payload, output_path, gt=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("RGB image", payload["rgb"], None, 0.0, 1.0),
        ("DABE-PU fixed", payload["fixed_pseudo"], "gray", 0.0, 1.0),
        ("FG anchors", payload["fg_anchor_mask"], "gray", 0.0, 1.0),
        ("BG anchors", payload["bg_anchor_mask"], "gray", 0.0, 1.0),
        ("DINO margin", payload["semantic_margin"], "coolwarm", -1.0, 1.0),
        ("EMA teacher binary", payload["teacher_binary"], "gray", 0.0, 1.0),
        ("Teacher correction", payload["teacher_correction"], "coolwarm", -1.0, 1.0),
        ("Semantic support", payload["semantic_support"], "viridis", 0.0, 1.0),
        ("Temporal support", payload["temporal_support"], "viridis", 0.0, 1.0),
        ("Local acceptance", payload["local_acceptance"], "viridis", 0.0, 1.0),
        ("Effective teacher weight", payload["effective_teacher_weight"], "viridis", 0.0, 1.0),
        ("Mixed target", payload["mixed_target"], "gray", 0.0, 1.0),
        ("Student prediction", payload["student_prediction"], "gray", 0.0, 1.0),
    ]
    if gt is not None:
        panels.append(("GT (offline only)", gt, "gray", 0.0, 1.0))
    columns = 4
    rows = int(math.ceil(len(panels) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(4.2 * columns, 4.0 * rows)
    )
    axes = np.asarray(axes).reshape(-1)
    for axis, (title, value, cmap, vmin, vmax) in zip(axes, panels):
        tensor = torch.as_tensor(value).detach().float().cpu()
        array = tensor.squeeze().numpy()
        if title == "RGB image":
            array = tensor.permute(1, 2, 0).numpy()
            axis.imshow(np.clip(array, 0.0, 1.0))
        else:
            image = axis.imshow(
                array,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set_title(title)
        axis.axis("off")
    for axis in axes[len(panels) :]:
        axis.axis("off")
    figure.suptitle(
        f"{payload['dataset']}/{payload['stem']} | "
        f"epoch={int(payload['epoch']):03d} | "
        f"bg={payload['bg_anchor_source']}"
    )
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
