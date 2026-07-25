import math

import torch
import torch.nn.functional as F
from torch import nn

from models.cacd import CACDV1BaseHead


class SimpleConvSegHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # baseline head 只允许一个 1x1 conv，不引入 decoder 或额外非线性。
        self.proj = nn.Conv2d(in_channels, 1, kernel_size=1)

    def forward(self, x):
        return self.proj(x)


class DINOAffinityGraphPropagationHead(nn.Module):
    def __init__(
        self,
        in_channels=384,
        hidden=64,
        topk=24,
        tau=0.10,
        alpha_init=0.05,
        gamma_init=0.10,
        affinity_detach=True,
        use_ffn=False,
        use_dwconv=False,
    ):
        super().__init__()
        if int(hidden) <= 0:
            raise ValueError(f"DAGP_HIDDEN must be positive, got {hidden}")
        if int(topk) <= 0:
            raise ValueError(f"DAGP_TOPK must be positive, got {topk}")
        if float(tau) <= 0.0:
            raise ValueError(f"DAGP_TAU must be positive, got {tau}")
        if use_ffn:
            raise ValueError("DAGP_USE_FFN=True is reserved but not implemented in DAGP-Minimal.")
        if use_dwconv:
            raise ValueError("DAGP_USE_DWCONV=True is reserved but not implemented in DAGP-Minimal.")

        self.hidden = int(hidden)
        self.topk = int(topk)
        self.tau = float(tau)
        self.affinity_detach = bool(affinity_detach)
        self.use_ffn = bool(use_ffn)
        self.use_dwconv = bool(use_dwconv)

        self.base_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.proj = nn.Conv2d(in_channels, self.hidden, kernel_size=1)
        self.value = nn.Linear(self.hidden, self.hidden)
        self.graph_pred = nn.Conv2d(self.hidden, 1, kernel_size=1)

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init), dtype=torch.float32))

    @staticmethod
    def _gather_neighbors(v, idx):
        # v: [B,N,D], idx: [B,N,K] -> [B,N,K,D]
        bsz, num_nodes, dim = v.shape
        k = idx.shape[-1]
        batch_offset = torch.arange(bsz, device=idx.device, dtype=idx.dtype).view(bsz, 1, 1) * num_nodes
        flat_idx = (idx + batch_offset).reshape(-1)
        return v.reshape(bsz * num_nodes, dim).index_select(0, flat_idx).reshape(bsz, num_nodes, k, dim)

    def _topk_affinity(self, feat):
        bsz, channels, height, width = feat.shape
        del channels
        num_nodes = height * width
        if num_nodes <= 1:
            raise ValueError(f"DAGP requires at least 2 spatial nodes, got H={height}, W={width}")

        x = feat.flatten(2).transpose(1, 2)
        if self.affinity_detach:
            x = x.detach()
        x_aff = F.normalize(x.float(), dim=-1)
        sim = torch.bmm(x_aff, x_aff.transpose(1, 2))
        diag = torch.eye(num_nodes, device=sim.device, dtype=torch.bool).unsqueeze(0)
        sim = sim.masked_fill(diag, -float("inf"))
        k = min(self.topk, num_nodes - 1)
        topk_val, topk_idx = torch.topk(sim, k=k, dim=-1)
        attn = torch.softmax(topk_val / self.tau, dim=-1)
        return attn, topk_idx

    def forward(self, feat):
        bsz, _, height, width = feat.shape
        base_logits = self.base_head(feat)

        if self.affinity_detach:
            with torch.no_grad():
                attn, topk_idx = self._topk_affinity(feat)
        else:
            attn, topk_idx = self._topk_affinity(feat)

        z_map = self.proj(feat)
        z = z_map.flatten(2).transpose(1, 2)
        value = self.value(z)
        neigh_value = self._gather_neighbors(value, topk_idx)
        agg = (attn.to(dtype=value.dtype).unsqueeze(-1) * neigh_value).sum(dim=2)
        z_prop = z + self.gamma.to(dtype=z.dtype) * agg
        z_prop_map = z_prop.transpose(1, 2).reshape(bsz, self.hidden, height, width)
        graph_logits = self.graph_pred(z_prop_map)
        return base_logits + self.alpha.to(dtype=graph_logits.dtype) * graph_logits


class NativeDetailResidualBranch(nn.Module):
    def __init__(
        self,
        in_channels=5,
        hidden=32,
        num_layers=3,
        use_gn=True,
        gn_groups=4,
        act="gelu",
        zero_init_out=True,
        residual_clip=2.0,
        input_rgb=True,
        input_sobel=True,
        input_coarse_prob=True,
    ):
        super().__init__()
        if int(hidden) <= 0:
            raise ValueError(f"NDR_HIDDEN must be positive, got {hidden}")
        if int(num_layers) != 3:
            raise ValueError(f"NDR_NUM_LAYERS=3 is required for the native detail branch, got {num_layers}")
        if float(residual_clip) <= 0.0:
            raise ValueError(f"NDR_RESIDUAL_CLIP must be positive, got {residual_clip}")
        self.input_rgb = bool(input_rgb)
        self.input_sobel = bool(input_sobel)
        self.input_coarse_prob = bool(input_coarse_prob)
        expected_channels = 0
        expected_channels += 3 if self.input_rgb else 0
        expected_channels += 1 if self.input_sobel else 0
        expected_channels += 1 if self.input_coarse_prob else 0
        if int(in_channels) != expected_channels:
            raise ValueError(
                f"NDR_IN_CHANNELS={in_channels} does not match enabled inputs "
                f"({expected_channels} channels)"
            )
        self.in_channels = int(in_channels)
        self.hidden = int(hidden)
        self.mid_channels = max(1, self.hidden // 2)
        self.use_gn = bool(use_gn)
        self.gn_groups = int(gn_groups)
        self.act_name = str(act).lower()
        self.residual_clip = float(residual_clip)

        self.block1 = self._block(self.in_channels, self.hidden)
        self.block2 = self._block(self.hidden, self.hidden)
        self.block3 = self._block(self.hidden, self.mid_channels)
        self.out_conv = nn.Conv2d(self.mid_channels, 1, kernel_size=1)
        if bool(zero_init_out):
            nn.init.zeros_(self.out_conv.weight)
            nn.init.zeros_(self.out_conv.bias)

        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _activation(self):
        if self.act_name == "gelu":
            return nn.GELU()
        if self.act_name == "relu":
            return nn.ReLU(inplace=True)
        raise ValueError(f"Unsupported NDR_ACT: {self.act_name}")

    def _norm(self, channels):
        if not self.use_gn:
            return nn.Identity()
        groups = max(1, min(self.gn_groups, channels))
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)

    def _block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            self._norm(out_channels),
            self._activation(),
        )

    def compute_sobel_map(self, x):
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Sobel map input must be [B,1,H,W], got {list(x.shape)}")
        sobel_x = self.sobel_x.to(device=x.device, dtype=x.dtype)
        sobel_y = self.sobel_y.to(device=x.device, dtype=x.dtype)
        dx = F.conv2d(x, sobel_x, padding=1)
        dy = F.conv2d(x, sobel_y, padding=1)
        sobel_mag = torch.sqrt(dx * dx + dy * dy + 1e-6)
        sobel_mag = sobel_mag / (sobel_mag.amax(dim=(2, 3), keepdim=True) + 1e-6)
        return torch.clamp(sobel_mag, 0.0, 1.0)

    def compute_sobel(self, image_68):
        if image_68.ndim != 4 or image_68.shape[1] != 3:
            raise ValueError(f"NDR image_68 must be [B,3,H,W], got {list(image_68.shape)}")
        gray = (
            0.299 * image_68[:, 0:1]
            + 0.587 * image_68[:, 1:2]
            + 0.114 * image_68[:, 2:3]
        )
        return self.compute_sobel_map(gray)

    def forward(self, image_68, coarse_prob_68, return_probe_aux=False):
        sobel_68 = self.compute_sobel(image_68)
        inputs = []
        if self.input_rgb:
            inputs.append(image_68)
        if self.input_sobel:
            inputs.append(sobel_68)
        if self.input_coarse_prob:
            inputs.append(coarse_prob_68)
        ndr_input = torch.cat(inputs, dim=1)
        hidden1 = self.block1(ndr_input)
        hidden2 = self.block2(hidden1)
        hidden3 = self.block3(hidden2)
        raw_residual = self.out_conv(hidden3)
        residual = raw_residual.tanh() * raw_residual.new_tensor(self.residual_clip)
        if return_probe_aux:
            return residual, sobel_68, hidden2.detach()
        return residual, sobel_68


class _CSDConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, groups=1, dilation=1, use_gn=True):
        super().__init__()
        self.conv = nn.Conv2d(
            int(in_channels),
            int(out_channels),
            kernel_size=int(kernel_size),
            padding=int(padding),
            dilation=int(dilation),
            groups=int(groups),
            bias=True,
        )
        if bool(use_gn):
            group_count = max(1, min(8, int(out_channels)))
            while int(out_channels) % group_count != 0 and group_count > 1:
                group_count -= 1
            self.norm = nn.GroupNorm(group_count, int(out_channels))
        else:
            self.norm = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class _CSDDepthwiseSeparableBlock(nn.Module):
    def __init__(self, channels, dilation=1, use_gn=True):
        super().__init__()
        channels = int(channels)
        dilation = int(dilation)
        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=channels,
            bias=True,
        )
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        if bool(use_gn):
            group_count = max(1, min(8, channels))
            while channels % group_count != 0 and group_count > 1:
                group_count -= 1
            self.norm = nn.GroupNorm(group_count, channels)
        else:
            self.norm = nn.Identity()
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.pw(self.dw(x))))


class CSDV1Head(nn.Module):
    def __init__(
        self,
        in_channels=384,
        loss_size=68,
        sem_dim=64,
        detail_dim=32,
        fusion_dim=64,
        use_dagp_semantic=True,
        use_local_context=True,
        use_global_context=True,
        dagp_topk=12,
        dagp_tau=0.07,
        dagp_alpha_max=0.05,
        dagp_gamma_max=0.03,
        warmup_epoch=6,
        ramp_start_epoch=7,
        ramp_end_epoch=15,
        beta_max=0.10,
        residual_clip=2.0,
        bg_suppress_strength=0.70,
        use_bg_detail_lock=True,
        use_boundary_aux=True,
    ):
        super().__init__()
        if int(in_channels) <= 0:
            raise ValueError(f"CSD in_channels must be positive, got {in_channels}")
        if int(loss_size) <= 0:
            raise ValueError(f"CSD loss_size must be positive, got {loss_size}")
        if int(sem_dim) <= 0 or int(detail_dim) <= 0 or int(fusion_dim) <= 0:
            raise ValueError("CSD dims must be positive.")
        if int(dagp_topk) <= 0:
            raise ValueError(f"CSD DAGP topk must be positive, got {dagp_topk}")
        if float(dagp_tau) <= 0.0:
            raise ValueError(f"CSD DAGP tau must be positive, got {dagp_tau}")
        if float(residual_clip) <= 0.0:
            raise ValueError(f"CSD residual_clip must be positive, got {residual_clip}")
        self.loss_size = int(loss_size)
        self.sem_dim = int(sem_dim)
        self.detail_dim = int(detail_dim)
        self.fusion_dim = int(fusion_dim)
        self.use_dagp_semantic = bool(use_dagp_semantic)
        self.use_local_context = bool(use_local_context)
        self.use_global_context = bool(use_global_context)
        self.topk = int(dagp_topk)
        self.tau = float(dagp_tau)
        self.alpha_max = float(dagp_alpha_max)
        self.gamma_max = float(dagp_gamma_max)
        self.warmup_epoch = int(warmup_epoch)
        self.ramp_start_epoch = int(ramp_start_epoch)
        self.ramp_end_epoch = int(ramp_end_epoch)
        self.beta_max = float(beta_max)
        self.residual_clip = float(residual_clip)
        self.bg_suppress_strength = float(bg_suppress_strength)
        self.use_bg_detail_lock = bool(use_bg_detail_lock)
        self.use_boundary_aux = bool(use_boundary_aux)

        sem_groups = max(1, min(8, self.sem_dim))
        while self.sem_dim % sem_groups != 0 and sem_groups > 1:
            sem_groups -= 1
        self.sem_proj = nn.Sequential(
            nn.Conv2d(int(in_channels), self.sem_dim, kernel_size=1, bias=False),
            nn.GroupNorm(sem_groups, self.sem_dim),
            nn.GELU(),
        )
        self.sem_local = _CSDDepthwiseSeparableBlock(self.sem_dim, dilation=1)
        self.sem_mid = _CSDDepthwiseSeparableBlock(self.sem_dim, dilation=2)
        self.sem_global = nn.Sequential(
            nn.Conv2d(self.sem_dim, self.sem_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(self.sem_dim, self.sem_dim, kernel_size=1, bias=True),
        )
        self.value = nn.Linear(self.sem_dim, self.sem_dim)
        self.graph_proj = nn.Conv2d(self.sem_dim, self.sem_dim, kernel_size=1, bias=True)
        self.base_head = nn.Conv2d(self.sem_dim, 1, kernel_size=1)
        self.coarse_head = nn.Conv2d(self.sem_dim, 1, kernel_size=1)

        self.detail_block1 = _CSDConvBlock(5, self.detail_dim)
        self.detail_block2 = _CSDConvBlock(self.detail_dim, self.detail_dim)
        self.detail_block3 = _CSDDepthwiseSeparableBlock(self.detail_dim, dilation=1)
        self.detail_proj = nn.Conv2d(self.detail_dim, self.sem_dim, kernel_size=1, bias=True)

        fusion_in = self.sem_dim + self.detail_dim + 3
        self.detail_gate = nn.Sequential(
            _CSDConvBlock(fusion_in, self.fusion_dim),
            nn.Conv2d(self.fusion_dim, 1, kernel_size=1, bias=True),
        )
        self.residual_head = nn.Sequential(
            _CSDConvBlock(self.sem_dim, self.fusion_dim),
            nn.Conv2d(self.fusion_dim, 1, kernel_size=1, bias=True),
        )
        self.boundary_head = nn.Conv2d(self.sem_dim, 1, kernel_size=1, bias=True)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.boundary_head.weight)
        nn.init.zeros_(self.boundary_head.bias)

        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("current_epoch_tensor", torch.zeros(1, dtype=torch.float32))

    def set_epoch(self, epoch):
        self.current_epoch_tensor.fill_(float(epoch))

    def _ramp_scale(self):
        epoch = int(self.current_epoch_tensor.item())
        if epoch <= self.warmup_epoch:
            return 0.0
        if epoch < self.ramp_start_epoch:
            return 0.0
        if epoch >= self.ramp_end_epoch:
            return 1.0
        denom = max(1, self.ramp_end_epoch - self.ramp_start_epoch + 1)
        return float(epoch - self.ramp_start_epoch + 1) / float(denom)

    def _compute_sobel(self, image_68):
        if image_68.ndim != 4 or image_68.shape[1] != 3:
            raise ValueError(f"CSD image_68 must be [B,3,H,W], got {list(image_68.shape)}")
        gray = 0.299 * image_68[:, 0:1] + 0.587 * image_68[:, 1:2] + 0.114 * image_68[:, 2:3]
        sx = self.sobel_x.to(device=gray.device, dtype=gray.dtype)
        sy = self.sobel_y.to(device=gray.device, dtype=gray.dtype)
        dx = F.conv2d(gray, sx, padding=1)
        dy = F.conv2d(gray, sy, padding=1)
        edge = torch.sqrt(dx * dx + dy * dy + 1e-6)
        return self._normalize_map_per_image(edge)

    @staticmethod
    def _normalize_map_per_image(tensor, eps=1e-6):
        flat = tensor.flatten(1)
        min_val = flat.min(dim=1).values.view(-1, 1, 1, 1)
        max_val = flat.max(dim=1).values.view(-1, 1, 1, 1)
        return torch.clamp((tensor - min_val) / (max_val - min_val + float(eps)), 0.0, 1.0)

    @staticmethod
    def _gather_neighbors(v, idx):
        bsz, num_nodes, dim = v.shape
        k = idx.shape[-1]
        batch_offset = torch.arange(bsz, device=idx.device, dtype=idx.dtype).view(bsz, 1, 1) * num_nodes
        flat_idx = (idx + batch_offset).reshape(-1)
        return v.reshape(bsz * num_nodes, dim).index_select(0, flat_idx).reshape(bsz, num_nodes, k, dim)

    def _topk_affinity(self, feat):
        _, _, height, width = feat.shape
        num_nodes = height * width
        if num_nodes <= 1:
            raise ValueError(f"CSD graph requires at least 2 spatial nodes, got H={height}, W={width}")
        x = feat.flatten(2).transpose(1, 2).detach()
        x_aff = F.normalize(x.float(), dim=-1)
        sim = torch.bmm(x_aff, x_aff.transpose(1, 2))
        diag = torch.eye(num_nodes, device=sim.device, dtype=torch.bool).unsqueeze(0)
        sim = sim.masked_fill(diag, -float("inf"))
        k = min(self.topk, num_nodes - 1)
        topk_val, topk_idx = torch.topk(sim, k=k, dim=-1)
        attn = torch.softmax(topk_val / self.tau, dim=-1)
        return attn, topk_idx

    def _semantic_graph(self, raw_feat, sem, gamma_eff):
        if not self.use_dagp_semantic:
            return torch.zeros_like(sem)
        with torch.no_grad():
            attn, topk_idx = self._topk_affinity(raw_feat)
        bsz, _, height, width = sem.shape
        z = sem.flatten(2).transpose(1, 2)
        value = self.value(z)
        neigh_value = self._gather_neighbors(value, topk_idx)
        agg = (attn.to(dtype=value.dtype).unsqueeze(-1) * neigh_value).sum(dim=2)
        z_prop = z + sem.new_tensor(float(gamma_eff)) * agg
        z_prop = z_prop.transpose(1, 2).reshape(bsz, self.sem_dim, height, width)
        return self.graph_proj(z_prop)

    def _semantic_branch(self, feat, scale):
        sem = self.sem_proj(feat)
        semantic_feat = sem
        if self.use_local_context:
            semantic_feat = semantic_feat + self.sem_local(sem) + self.sem_mid(sem)
        if self.use_global_context:
            global_feat = F.adaptive_avg_pool2d(sem, output_size=1)
            semantic_feat = semantic_feat + self.sem_global(global_feat).expand_as(sem)
        alpha_eff = self.alpha_max * float(scale) if self.use_dagp_semantic else 0.0
        gamma_eff = self.gamma_max * float(scale) if self.use_dagp_semantic else 0.0
        graph_feat = self._semantic_graph(feat, sem, gamma_eff)
        semantic_feat = semantic_feat + sem.new_tensor(float(alpha_eff)) * graph_feat
        return semantic_feat, alpha_eff, gamma_eff

    def forward(self, feat, image_68=None, return_aux=False, bg_reliable_68=None):
        if feat.ndim != 4:
            raise ValueError(f"CSD feature must be [B,C,H,W], got {list(feat.shape)}")
        if image_68 is None:
            raise ValueError("CSD-v1 requires image_68.")
        target_size = (self.loss_size, self.loss_size)
        if tuple(image_68.shape[-2:]) != target_size:
            raise ValueError(f"CSD image_68 spatial size must be {target_size}, got {tuple(image_68.shape[-2:])}")
        image_68 = image_68.to(dtype=feat.dtype)
        csd_scale = float(self._ramp_scale())
        beta_eff = self.beta_max * csd_scale

        semantic_feat_37, alpha_eff, gamma_eff = self._semantic_branch(feat, csd_scale)
        base_logits_37 = self.base_head(semantic_feat_37)
        coarse_logits_37 = self.coarse_head(semantic_feat_37)
        base_logits_68 = F.interpolate(base_logits_37, size=target_size, mode="bilinear", align_corners=False)
        coarse_logits_68 = F.interpolate(coarse_logits_37, size=target_size, mode="bilinear", align_corners=False)
        semantic_feat_68 = F.interpolate(semantic_feat_37, size=target_size, mode="bilinear", align_corners=False)

        coarse_prob_68 = torch.sigmoid(coarse_logits_68.detach())
        sobel_68 = self._compute_sobel(image_68)
        detail_input = torch.cat([image_68, sobel_68, coarse_prob_68], dim=1)
        detail_feat_68 = self.detail_block3(self.detail_block2(self.detail_block1(detail_input)))
        uncertainty_68 = torch.clamp(1.0 - 2.0 * torch.abs(coarse_prob_68 - 0.5), 0.0, 1.0)
        fusion_input = torch.cat([semantic_feat_68, detail_feat_68, coarse_prob_68, uncertainty_68, sobel_68], dim=1)
        detail_gate_raw = torch.sigmoid(self.detail_gate(fusion_input))
        if self.training and self.use_bg_detail_lock and bg_reliable_68 is not None:
            bg_mask = bg_reliable_68.to(device=detail_gate_raw.device, dtype=detail_gate_raw.dtype).detach()
            detail_gate = detail_gate_raw * (1.0 - detail_gate_raw.new_tensor(self.bg_suppress_strength) * bg_mask)
            detail_gate = torch.clamp(detail_gate, 0.0, 1.0)
        else:
            bg_mask = torch.zeros_like(detail_gate_raw)
            detail_gate = detail_gate_raw
        detail_proj_68 = self.detail_proj(detail_feat_68)
        detail_injected_68 = detail_gate * detail_proj_68
        structure_feat_68 = semantic_feat_68 + semantic_feat_68.new_tensor(float(csd_scale)) * detail_injected_68
        residual_logits_68 = torch.clamp(
            self.residual_head(structure_feat_68),
            -self.residual_clip,
            self.residual_clip,
        )
        boundary_logits_68 = self.boundary_head(structure_feat_68)
        final_logits_68 = coarse_logits_68 + coarse_logits_68.new_tensor(float(beta_eff)) * residual_logits_68

        output = {
            "logits": final_logits_68,
            "final_logits": final_logits_68,
            "coarse_logits": coarse_logits_68,
            "coarse_logits_37": coarse_logits_37,
            "coarse_logits_68": coarse_logits_68,
            "base_logits": base_logits_68,
            "base_logits_37": base_logits_37,
            "boundary_logits": boundary_logits_68,
            "boundary_logits_68": boundary_logits_68,
        }
        if return_aux:
            gate_detached = detail_gate.detach()
            residual_abs = residual_logits_68.detach().abs()
            output.update(
                {
                    "semantic_feat_37": semantic_feat_37,
                    "semantic_feat_68": semantic_feat_68,
                    "detail_feat_68": detail_feat_68,
                    "coarse_prob_68": coarse_prob_68,
                    "sobel_68": sobel_68,
                    "edge_norm_68": sobel_68,
                    "uncertainty_68": uncertainty_68,
                    "csd_detail_gate": detail_gate,
                    "csd_detail_gate_raw": detail_gate_raw,
                    "csd_bg_mask": bg_mask,
                    "csd_detail_injected_68": detail_injected_68,
                    "csd_residual_logits": residual_logits_68,
                    "residual_logits_68": residual_logits_68,
                    "csd_scale": coarse_logits_68.new_tensor(float(csd_scale)),
                    "csd_alpha_eff": coarse_logits_68.new_tensor(float(alpha_eff)),
                    "csd_gamma_eff": coarse_logits_68.new_tensor(float(gamma_eff)),
                    "csd_beta_eff": coarse_logits_68.new_tensor(float(beta_eff)),
                    "csd_detail_gate_mean": gate_detached.mean(),
                    "csd_detail_gate_min": gate_detached.min(),
                    "csd_detail_gate_max": gate_detached.max(),
                    "csd_residual_abs_mean": residual_abs.mean(),
                    "csd_residual_abs_max": residual_abs.max(),
                }
            )
        return output


class CSDV1RResidual(nn.Module):
    def __init__(
        self,
        in_channels=384,
        sem_dim=64,
        detail_dim=32,
        fusion_dim=64,
        beta_max=0.05,
        residual_clip=2.0,
        warmup_epoch=6,
        ramp_start_epoch=7,
        ramp_end_epoch=15,
        bg_suppress_strength=0.70,
        use_bg_detail_lock=True,
        use_res_zero_init=True,
        use_gate_bias_init=True,
        gate_bias_init=-2.0,
    ):
        super().__init__()
        if int(in_channels) <= 0:
            raise ValueError(f"CSD_V1R input channels must be positive, got {in_channels}")
        if int(sem_dim) <= 0 or int(detail_dim) <= 0 or int(fusion_dim) <= 0:
            raise ValueError("CSD_V1R dims must be positive.")
        if float(residual_clip) <= 0.0:
            raise ValueError(f"CSD_V1R_RESIDUAL_CLIP must be positive, got {residual_clip}")

        self.sem_dim = int(sem_dim)
        self.detail_dim = int(detail_dim)
        self.fusion_dim = int(fusion_dim)
        self.beta_max = float(beta_max)
        self.residual_clip = float(residual_clip)
        self.warmup_epoch = int(warmup_epoch)
        self.ramp_start_epoch = int(ramp_start_epoch)
        self.ramp_end_epoch = int(ramp_end_epoch)
        self.bg_suppress_strength = float(bg_suppress_strength)
        self.use_bg_detail_lock = bool(use_bg_detail_lock)

        sem_groups = max(1, min(8, self.sem_dim))
        while self.sem_dim % sem_groups != 0 and sem_groups > 1:
            sem_groups -= 1
        self.csd_feat_proj = nn.Sequential(
            nn.Conv2d(int(in_channels), self.sem_dim, kernel_size=1, bias=False),
            nn.GroupNorm(sem_groups, self.sem_dim),
            nn.GELU(),
        )
        self.detail_encoder = nn.Sequential(
            _CSDConvBlock(5, self.detail_dim),
            _CSDConvBlock(self.detail_dim, self.detail_dim),
            _CSDDepthwiseSeparableBlock(self.detail_dim),
        )
        self.detail_proj = nn.Conv2d(self.detail_dim, self.sem_dim, kernel_size=1, bias=True)
        fusion_in = self.sem_dim + self.detail_dim + 3
        self.detail_gate = nn.Sequential(
            _CSDConvBlock(fusion_in, self.fusion_dim),
            nn.Conv2d(self.fusion_dim, 1, kernel_size=1, bias=True),
        )
        self.residual_head = nn.Sequential(
            _CSDConvBlock(self.sem_dim, self.fusion_dim),
            nn.Conv2d(self.fusion_dim, 1, kernel_size=1, bias=True),
        )
        if bool(use_res_zero_init):
            nn.init.zeros_(self.residual_head[-1].weight)
            nn.init.zeros_(self.residual_head[-1].bias)
        if bool(use_gate_bias_init):
            nn.init.constant_(self.detail_gate[-1].bias, float(gate_bias_init))

        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("current_epoch_tensor", torch.zeros(1, dtype=torch.float32))

    def set_epoch(self, epoch):
        self.current_epoch_tensor.fill_(float(epoch))

    def _ramp_scale(self):
        epoch = int(self.current_epoch_tensor.item())
        if epoch <= self.warmup_epoch:
            return 0.0
        if epoch < self.ramp_start_epoch:
            return 0.0
        if epoch >= self.ramp_end_epoch:
            return 1.0
        denom = max(1, self.ramp_end_epoch - self.ramp_start_epoch + 1)
        return float(epoch - self.ramp_start_epoch + 1) / float(denom)

    @staticmethod
    def _normalize_map_per_image(tensor, eps=1e-6):
        flat = tensor.detach().flatten(1)
        min_val = flat.min(dim=1).values.view(-1, 1, 1, 1)
        max_val = flat.max(dim=1).values.view(-1, 1, 1, 1)
        return torch.clamp((tensor - min_val) / (max_val - min_val + float(eps)), 0.0, 1.0)

    def _compute_sobel(self, image_68):
        if image_68.ndim != 4 or image_68.shape[1] != 3:
            raise ValueError(f"CSD-v1R image_68 must be [B,3,H,W], got {list(image_68.shape)}")
        gray = 0.299 * image_68[:, 0:1] + 0.587 * image_68[:, 1:2] + 0.114 * image_68[:, 2:3]
        sx = self.sobel_x.to(device=gray.device, dtype=gray.dtype)
        sy = self.sobel_y.to(device=gray.device, dtype=gray.dtype)
        dx = F.conv2d(gray, sx, padding=1)
        dy = F.conv2d(gray, sy, padding=1)
        edge = torch.sqrt(dx * dx + dy * dy + 1e-6)
        return self._normalize_map_per_image(edge)

    def forward(self, feature, image_68, coarse_logits_68, bg_reliable_68=None):
        if feature.ndim != 4:
            raise ValueError(f"CSD-v1R feature must be [B,C,H,W], got {list(feature.shape)}")
        if image_68 is None:
            raise ValueError("CSD-v1R requires image_68.")
        if coarse_logits_68.ndim != 4 or coarse_logits_68.shape[1] != 1:
            raise ValueError(f"CSD-v1R coarse_logits_68 must be [B,1,H,W], got {list(coarse_logits_68.shape)}")
        if tuple(image_68.shape[-2:]) != tuple(coarse_logits_68.shape[-2:]):
            raise ValueError(
                "CSD-v1R image_68 and coarse_logits_68 spatial sizes must match, got "
                f"{tuple(image_68.shape[-2:])} and {tuple(coarse_logits_68.shape[-2:])}"
            )

        image_68 = image_68.to(dtype=feature.dtype)
        coarse_logits_68 = coarse_logits_68.to(dtype=feature.dtype)
        csd_scale = float(self._ramp_scale())
        beta_eff = self.beta_max * csd_scale

        sem_feat_37 = self.csd_feat_proj(feature)
        sem_feat_68 = F.interpolate(
            sem_feat_37,
            size=coarse_logits_68.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        coarse_prob_68 = torch.sigmoid(coarse_logits_68.detach())
        sobel_68 = self._compute_sobel(image_68)
        detail_input = torch.cat([image_68, sobel_68, coarse_prob_68], dim=1)
        detail_feat_68 = self.detail_encoder(detail_input)
        uncertainty_68 = torch.clamp(1.0 - 2.0 * torch.abs(coarse_prob_68 - 0.5), 0.0, 1.0)
        fusion_input = torch.cat(
            [sem_feat_68, detail_feat_68, coarse_prob_68, uncertainty_68, sobel_68.detach()],
            dim=1,
        )
        detail_gate_raw = torch.sigmoid(self.detail_gate(fusion_input))
        if self.training and self.use_bg_detail_lock and bg_reliable_68 is not None:
            bg_mask = bg_reliable_68.to(device=detail_gate_raw.device, dtype=detail_gate_raw.dtype).detach()
            detail_gate = detail_gate_raw * (1.0 - detail_gate_raw.new_tensor(self.bg_suppress_strength) * bg_mask)
            detail_gate = torch.clamp(detail_gate, 0.0, 1.0)
        else:
            bg_mask = torch.zeros_like(detail_gate_raw)
            detail_gate = detail_gate_raw

        detail_proj_68 = self.detail_proj(detail_feat_68)
        detail_injected_68 = detail_gate * detail_proj_68
        res_feat_68 = sem_feat_68 + detail_injected_68
        residual_logits_68 = torch.clamp(
            self.residual_head(res_feat_68),
            -self.residual_clip,
            self.residual_clip,
        )

        gate_detached = detail_gate.detach()
        residual_abs = residual_logits_68.detach().abs()
        return {
            "csd_residual_logits": residual_logits_68,
            "residual_logits_68": residual_logits_68,
            "csd_detail_gate": detail_gate,
            "csd_detail_gate_raw": detail_gate_raw,
            "csd_detail_feat": detail_feat_68,
            "detail_feat_68": detail_feat_68,
            "csd_sem_feat": sem_feat_68,
            "semantic_feat_native": sem_feat_37,
            "semantic_feat_37": sem_feat_37,
            "semantic_feat_68": sem_feat_68,
            "sobel_68": sobel_68,
            "edge_norm_68": sobel_68,
            "uncertainty_68": uncertainty_68,
            "csd_bg_mask": bg_mask,
            "csd_detail_injected_68": detail_injected_68,
            "csd_scale": coarse_logits_68.new_tensor(float(csd_scale)),
            "csd_beta_eff": coarse_logits_68.new_tensor(float(beta_eff)),
            "beta_eff": coarse_logits_68.new_tensor(float(beta_eff)),
            "csd_detail_gate_mean": gate_detached.mean(),
            "csd_detail_gate_min": gate_detached.min(),
            "csd_detail_gate_max": gate_detached.max(),
            "csd_residual_abs_mean": residual_abs.mean(),
            "csd_residual_abs_max": residual_abs.max(),
        }


class HRBFRV1Branch(nn.Module):
    def __init__(
        self,
        in_channels=384,
        sem_dim=32,
        detail_dim=32,
        hidden_dim=32,
        residual_clip=2.0,
        use_rgb=True,
        use_sobel=True,
        use_anchor_prob=True,
        use_anchor_uncert=True,
        use_dino_sem=True,
        use_res_zero_init=True,
    ):
        super().__init__()
        if int(sem_dim) <= 0 or int(detail_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("HR-BFR dims must be positive.")
        if float(residual_clip) <= 0.0:
            raise ValueError(f"HR_BFR_RESIDUAL_CLIP must be positive, got {residual_clip}")
        self.sem_dim = int(sem_dim)
        self.detail_dim = int(detail_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_clip = float(residual_clip)
        self.use_rgb = bool(use_rgb)
        self.use_sobel = bool(use_sobel)
        self.use_anchor_prob = bool(use_anchor_prob)
        self.use_anchor_uncert = bool(use_anchor_uncert)
        self.use_dino_sem = bool(use_dino_sem)

        sem_groups = max(1, min(8, self.sem_dim))
        while self.sem_dim % sem_groups != 0 and sem_groups > 1:
            sem_groups -= 1
        self.dino_sem_proj = nn.Sequential(
            nn.Conv2d(int(in_channels), self.sem_dim, kernel_size=1, bias=False),
            nn.GroupNorm(sem_groups, self.sem_dim),
            nn.GELU(),
        )

        input_channels = 0
        input_channels += 3 if self.use_rgb else 0
        input_channels += 1 if self.use_sobel else 0
        input_channels += 1 if self.use_anchor_prob else 0
        input_channels += 1 if self.use_anchor_uncert else 0
        input_channels += self.sem_dim if self.use_dino_sem else 0
        input_channels += 1
        self.net = nn.Sequential(
            _CSDConvBlock(input_channels, self.detail_dim),
            _CSDConvBlock(self.detail_dim, self.hidden_dim),
            _CSDDepthwiseSeparableBlock(self.hidden_dim),
            _CSDConvBlock(self.hidden_dim, self.hidden_dim),
            nn.Conv2d(self.hidden_dim, 1, kernel_size=1, bias=True),
        )
        if bool(use_res_zero_init):
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    @staticmethod
    def _normalize_map_per_image(tensor, eps=1e-6):
        flat = tensor.detach().flatten(1)
        min_val = flat.min(dim=1).values.view(-1, 1, 1, 1)
        max_val = flat.max(dim=1).values.view(-1, 1, 1, 1)
        return torch.clamp((tensor - min_val) / (max_val - min_val + float(eps)), 0.0, 1.0)

    def compute_sobel(self, image):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"HR-BFR image must be [B,3,H,W], got {list(image.shape)}")
        gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        sx = self.sobel_x.to(device=gray.device, dtype=gray.dtype)
        sy = self.sobel_y.to(device=gray.device, dtype=gray.dtype)
        dx = F.conv2d(gray, sx, padding=1)
        dy = F.conv2d(gray, sy, padding=1)
        edge = torch.sqrt(dx * dx + dy * dy + 1e-6)
        return self._normalize_map_per_image(edge)

    def forward(self, feature, image_136, anchor_prob_136, anchor_uncert_136, band_gate_136):
        if image_136 is None:
            raise ValueError("HR-BFR requires image_136.")
        image_136 = image_136.to(dtype=feature.dtype)
        anchor_prob_136 = anchor_prob_136.to(dtype=feature.dtype)
        anchor_uncert_136 = anchor_uncert_136.to(dtype=feature.dtype)
        band_gate_136 = band_gate_136.to(dtype=feature.dtype)
        sobel_136 = self.compute_sobel(image_136)
        dino_sem_37 = self.dino_sem_proj(feature)
        dino_sem_136 = F.interpolate(dino_sem_37, size=image_136.shape[-2:], mode="bilinear", align_corners=False)

        inputs = []
        if self.use_rgb:
            inputs.append(image_136)
        if self.use_sobel:
            inputs.append(sobel_136)
        if self.use_anchor_prob:
            inputs.append(anchor_prob_136)
        if self.use_anchor_uncert:
            inputs.append(anchor_uncert_136)
        if self.use_dino_sem:
            inputs.append(dino_sem_136)
        inputs.append(band_gate_136)
        raw_residual = self.net(torch.cat(inputs, dim=1))
        residual = torch.clamp(raw_residual, -self.residual_clip, self.residual_clip) * band_gate_136
        return {
            "hr_residual_logits": residual,
            "hr_sobel_136": sobel_136,
            "hr_dino_sem_37": dino_sem_37,
            "hr_dino_sem_136": dino_sem_136,
        }


class DAGPSafeCSDV1RHead(nn.Module):
    def __init__(
        self,
        coarse_path,
        residual_branch,
        loss_size=68,
        hr_bfr_branch=None,
        use_hr_bfr=False,
        hr_size=136,
        hr_beta_max=0.05,
        hr_warmup_epoch=6,
        hr_ramp_start_epoch=7,
        hr_ramp_end_epoch=15,
        hr_boundary_thresh=0.5,
        hr_boundary_radius_68=2,
        hr_boundary_dilate_136=2,
        hr_max_band_ratio=0.35,
        hr_detach_anchor=True,
        hr_eval_res_scale=1.0,
    ):
        super().__init__()
        self.coarse_path = coarse_path
        self.csd_residual = residual_branch
        self.loss_size = int(loss_size)
        self.hr_bfr_branch = hr_bfr_branch
        self.use_hr_bfr = bool(use_hr_bfr)
        self.hr_size = int(hr_size)
        self.hr_beta_max = float(hr_beta_max)
        self.hr_warmup_epoch = int(hr_warmup_epoch)
        self.hr_ramp_start_epoch = int(hr_ramp_start_epoch)
        self.hr_ramp_end_epoch = int(hr_ramp_end_epoch)
        self.hr_boundary_thresh = float(hr_boundary_thresh)
        self.hr_boundary_radius_68 = int(hr_boundary_radius_68)
        self.hr_boundary_dilate_136 = int(hr_boundary_dilate_136)
        self.hr_max_band_ratio = float(hr_max_band_ratio)
        self.hr_detach_anchor = bool(hr_detach_anchor)
        self.set_hr_eval_res_scale(hr_eval_res_scale)
        self.topk = getattr(coarse_path, "topk", 0)
        self.tau = getattr(coarse_path, "tau", 0.0)
        self.use_prob_gate = getattr(coarse_path, "use_prob_gate", False)
        self.use_uncertainty_output_gate = getattr(coarse_path, "use_uncertainty_output_gate", False)
        self.register_buffer("current_epoch_tensor", torch.zeros(1, dtype=torch.float32))

    def set_hr_eval_res_scale(self, scale):
        scale = float(scale)
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError(f"HR_BFR_EVAL_RES_SCALE must be finite and non-negative, got {scale}")
        self.hr_eval_res_scale = scale

    def set_epoch(self, epoch):
        self.current_epoch_tensor.fill_(float(epoch))
        if hasattr(self.coarse_path, "set_epoch"):
            self.coarse_path.set_epoch(epoch)
        if hasattr(self.csd_residual, "set_epoch"):
            self.csd_residual.set_epoch(epoch)
        if hasattr(self.hr_bfr_branch, "set_epoch"):
            self.hr_bfr_branch.set_epoch(epoch)

    def _hr_scale(self):
        epoch = int(self.current_epoch_tensor.item())
        if epoch <= self.hr_warmup_epoch:
            return 0.0
        if epoch < self.hr_ramp_start_epoch:
            return 0.0
        if epoch >= self.hr_ramp_end_epoch:
            return 1.0
        denom = max(1, self.hr_ramp_end_epoch - self.hr_ramp_start_epoch + 1)
        return float(epoch - self.hr_ramp_start_epoch + 1) / float(denom)

    @staticmethod
    def _binary_band(mask, radius):
        radius = int(radius)
        if radius <= 0:
            return torch.zeros_like(mask)
        k = 2 * radius + 1
        mask = mask.float()
        dilated = F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius) > 0.5
        eroded = (1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=radius)) > 0.5
        return (dilated & ~eroded).float()

    def _build_hr_band(self, anchor_prob_68):
        anchor_bin_68 = (anchor_prob_68 >= self.hr_boundary_thresh).float()
        band_68 = self._binary_band(anchor_bin_68, self.hr_boundary_radius_68)
        band_136 = F.interpolate(band_68, size=(self.hr_size, self.hr_size), mode="nearest")
        if self.hr_boundary_dilate_136 > 0:
            k = 2 * self.hr_boundary_dilate_136 + 1
            band_136 = F.max_pool2d(band_136, kernel_size=k, stride=1, padding=self.hr_boundary_dilate_136)
        return band_68.detach(), (band_136 > 0.5).float().detach()

    def forward(
        self,
        feat,
        image_68=None,
        image_136=None,
        return_aux=False,
        bg_reliable_68=None,
        pa_compare_original=False,
    ):
        if image_68 is None:
            raise ValueError("DAGPSafeCSDV1RHead requires image_68.")
        coarse_out = self.coarse_path(
            feat,
            image_68=None,
            return_aux=True,
            pa_return_aux=return_aux,
            pa_compare_original=pa_compare_original,
        )
        if not isinstance(coarse_out, dict):
            old_coarse_logits_37 = coarse_out
            old_base_logits_37 = None
        else:
            old_coarse_logits_37 = coarse_out["logits"]
            old_base_logits_37 = coarse_out.get("base_logits")

        target_size = (self.loss_size, self.loss_size)
        old_coarse_logits_68 = F.interpolate(
            old_coarse_logits_37,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        old_base_logits_68 = (
            F.interpolate(old_base_logits_37, size=target_size, mode="bilinear", align_corners=False)
            if old_base_logits_37 is not None
            else None
        )
        residual_out = self.csd_residual(
            feat,
            image_68,
            old_coarse_logits_68,
            bg_reliable_68=bg_reliable_68,
        )
        residual_logits_68 = residual_out["csd_residual_logits"]
        beta_eff = residual_out["csd_beta_eff"].to(dtype=old_coarse_logits_68.dtype)
        final_logits_68 = old_coarse_logits_68 + beta_eff * residual_logits_68
        final_minus_coarse = (final_logits_68 - old_coarse_logits_68).detach().abs()

        output = {
            "logits": final_logits_68,
            "final_logits": final_logits_68,
            "coarse_logits": old_coarse_logits_68,
            "coarse_logits_native": old_coarse_logits_37,
            "coarse_logits_37": old_coarse_logits_37,
            "coarse_logits_68": old_coarse_logits_68,
            "old_coarse_logits_37": old_coarse_logits_37,
            "old_coarse_logits_native": old_coarse_logits_37,
            "old_coarse_logits_68": old_coarse_logits_68,
            "csd_residual_logits": residual_logits_68,
            "residual_logits_68": residual_logits_68,
            "csd_final_minus_coarse_abs_mean": final_minus_coarse.mean(),
            "final_minus_coarse_abs_mean": final_minus_coarse.mean(),
        }
        if old_base_logits_68 is not None:
            output["base_logits"] = old_base_logits_68
            output["base_logits_native"] = old_base_logits_37
            output["base_logits_37"] = old_base_logits_37
            output["old_base_logits_68"] = old_base_logits_68
        if isinstance(coarse_out, dict):
            for key in (
                "graph_logits",
                "dagp_scale",
                "dagp_alpha_eff",
                "dagp_gamma_eff",
                "uncertainty_gate_mean",
                "uncertainty_gate_min",
                "uncertainty_gate_max",
            ):
                if key in coarse_out:
                    output[key] = coarse_out[key]
        if return_aux:
            output.update(residual_out)
            if isinstance(coarse_out, dict):
                for key in (
                    "pa_raw_polarity",
                    "pa_signed_polarity",
                    "pa_base_prob",
                    "pa_anchor_valid",
                    "pa_rho",
                    "pa_diag",
                ):
                    if key in coarse_out:
                        output[key] = coarse_out[key]
        else:
            # Keep essential diagnostics available for existing train/eval extractors.
            for key in ("csd_scale", "csd_beta_eff", "beta_eff"):
                output[key] = residual_out[key]
        if pa_compare_original and isinstance(coarse_out, dict):
            original_coarse_native = coarse_out.get("pa_original_coarse_logits_native")
            if original_coarse_native is not None:
                with torch.no_grad():
                    original_coarse_68 = F.interpolate(
                        original_coarse_native,
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    )
                    original_residual_out = self.csd_residual(
                        feat.detach(),
                        image_68.detach(),
                        original_coarse_68,
                        bg_reliable_68=(
                            bg_reliable_68.detach() if bg_reliable_68 is not None else None
                        ),
                    )
                    original_final_68 = (
                        original_coarse_68
                        + original_residual_out["csd_beta_eff"].to(dtype=original_coarse_68.dtype)
                        * original_residual_out["csd_residual_logits"]
                    )
                output["pa_original_coarse_logits_native"] = original_coarse_native.detach()
                output["pa_original_coarse_logits_68"] = original_coarse_68.detach()
                output["pa_original_final_logits_68"] = original_final_68.detach()
        if self.use_hr_bfr:
            if self.hr_bfr_branch is None:
                raise RuntimeError("USE_HR_BFR=True but hr_bfr_branch is missing.")
            if image_136 is None:
                raise ValueError("HR-BFR requires image_136 from original image resize.")
            anchor_source = final_logits_68.detach() if self.hr_detach_anchor else final_logits_68
            anchor_logits_136 = F.interpolate(
                anchor_source,
                size=(self.hr_size, self.hr_size),
                mode="bilinear",
                align_corners=False,
            )
            anchor_prob_68 = torch.sigmoid(anchor_source.detach())
            anchor_prob_136 = torch.sigmoid(anchor_logits_136)
            anchor_uncert_136 = torch.clamp(1.0 - 2.0 * torch.abs(anchor_prob_136 - 0.5), 0.0, 1.0)
            band_68, band_136 = self._build_hr_band(anchor_prob_68)
            band_ratio_raw_136_per_image = band_136.flatten(1).mean(dim=1)
            valid_img_mask = (band_ratio_raw_136_per_image <= self.hr_max_band_ratio).to(dtype=band_136.dtype)
            valid_img_mask = valid_img_mask.view(-1, 1, 1, 1)
            hr_scale = float(self._hr_scale())
            beta_hr = self.hr_beta_max * hr_scale
            if hr_scale > 0.0:
                effective_band_136 = band_136 * valid_img_mask
                effective_band_68 = band_68 * valid_img_mask
            else:
                effective_band_136 = torch.zeros_like(band_136)
                effective_band_68 = torch.zeros_like(band_68)
            hr_out = self.hr_bfr_branch(
                feat,
                image_136,
                anchor_prob_136.detach(),
                anchor_uncert_136.detach(),
                effective_band_136,
            )
            hr_residual = hr_out["hr_residual_logits"]
            # The probe multiplier is eval-only. Training numerics remain exactly the
            # original HR-BFR formula regardless of the config value.
            eval_res_scale = 1.0 if self.training else float(self.hr_eval_res_scale)
            if eval_res_scale == 0.0:
                hr_logits = anchor_logits_136
            else:
                hr_logits = (
                    anchor_logits_136
                    + anchor_logits_136.new_tensor(eval_res_scale * float(beta_hr))
                    * effective_band_136
                    * hr_residual
                )
            hr_minus_anchor = (hr_logits - anchor_logits_136).detach().abs()
            hr_res_abs = hr_residual.detach().abs()
            output.update(
                {
                    "hr_logits": hr_logits,
                    "hr_anchor_logits": anchor_logits_136,
                    "hr_anchor_prob": anchor_prob_136,
                    "hr_anchor_uncert": anchor_uncert_136,
                    "hr_residual_logits": hr_residual,
                    "hr_band_gate": effective_band_136,
                    "hr_band_gate_136": effective_band_136,
                    "hr_band_gate_68": effective_band_68,
                    "hr_band_gate_raw_136": band_136,
                    "hr_band_gate_raw_68": band_68,
                    "hr_valid_img_mask": valid_img_mask,
                    "hr_valid_img_ratio": valid_img_mask.mean(),
                    "hr_skip_img_ratio": 1.0 - valid_img_mask.mean(),
                    "hr_beta_eff": hr_logits.new_tensor(float(beta_hr)),
                    "hr_scale": hr_logits.new_tensor(float(hr_scale)),
                    "hr_eval_res_scale": hr_logits.new_tensor(float(eval_res_scale)),
                    "hr_minus_anchor_abs_mean": hr_minus_anchor.mean(),
                    "hr_residual_abs_mean": hr_res_abs.mean(),
                    "hr_residual_abs_max": hr_res_abs.max(),
                    "hr_band_ratio_68": effective_band_68.detach().mean(),
                    "hr_band_ratio_136": effective_band_136.detach().mean(),
                    "hr_band_ratio_raw_136_mean": band_ratio_raw_136_per_image.detach().mean(),
                    "hr_band_ratio_raw_136_max": band_ratio_raw_136_per_image.detach().max(),
                    "hr_active_pixel_ratio": effective_band_136.detach().mean(),
                }
            )
            if return_aux:
                output.update(hr_out)
        return output


class AdaptiveDetailRouter(nn.Module):
    def __init__(
        self,
        in_channels=4,
        hidden=16,
        num_layers=2,
        act="gelu",
        init_bias=2.0,
        zero_init_out=True,
    ):
        super().__init__()
        if int(in_channels) <= 0:
            raise ValueError(f"TADR_ROUTER_IN_CHANNELS must be positive, got {in_channels}")
        if int(hidden) <= 0:
            raise ValueError(f"TADR_ROUTER_HIDDEN must be positive, got {hidden}")
        if int(num_layers) != 2:
            raise ValueError(f"TADR_ROUTER_NUM_LAYERS=2 is required, got {num_layers}")
        self.in_channels = int(in_channels)
        self.hidden = int(hidden)
        self.act_name = str(act).lower()
        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden, kernel_size=3, padding=1, bias=True),
            self._activation(),
            nn.Conv2d(self.hidden, self.hidden, kernel_size=3, padding=1, bias=True),
            self._activation(),
        )
        self.out_conv = nn.Conv2d(self.hidden, 1, kernel_size=1, bias=True)
        if bool(zero_init_out):
            nn.init.zeros_(self.out_conv.weight)
            nn.init.constant_(self.out_conv.bias, float(init_bias))

    def _activation(self):
        if self.act_name == "gelu":
            return nn.GELU()
        if self.act_name == "relu":
            return nn.ReLU(inplace=True)
        raise ValueError(f"Unsupported TADR_ROUTER_ACT: {self.act_name}")

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"TADR router input must be [B,{self.in_channels},H,W], got {list(x.shape)}"
            )
        return torch.sigmoid(self.out_conv(self.net(x)))


class PolarityEdgeGateV1(nn.Module):
    def __init__(
        self,
        in_channels=384,
        pol_dim=32,
        gn_groups=4,
        act="gelu",
        anchor_weight_power=2.0,
        anchor_eps=1e-6,
        detach_base_prob=True,
        detach_anchors=True,
        detach_rho=True,
        min_anchor_norm=1e-6,
        use_calib_head=True,
        calib_hidden=32,
        calib_zero_init=True,
        polarity_tau=0.50,
        edge_cut_max=0.50,
        edge_gate_min=0.50,
        use_same_polarity_boost=False,
        renormalize_edge=True,
        start_epoch=7,
        ramp_end_epoch=15,
        edge_stop_epoch=36,
        diag_ambig_thresh=0.20,
    ):
        super().__init__()
        self.pol_dim = int(pol_dim)
        self.anchor_weight_power = float(anchor_weight_power)
        self.anchor_eps = float(anchor_eps)
        self.detach_base_prob = bool(detach_base_prob)
        self.detach_anchors = bool(detach_anchors)
        self.detach_rho = bool(detach_rho)
        self.min_anchor_norm = float(min_anchor_norm)
        self.use_calib_head = bool(use_calib_head)
        self.polarity_tau = float(polarity_tau)
        self.edge_cut_max = float(edge_cut_max)
        self.edge_gate_min = float(edge_gate_min)
        self.use_same_polarity_boost = bool(use_same_polarity_boost)
        self.renormalize_edge = bool(renormalize_edge)
        self.start_epoch = int(start_epoch)
        self.ramp_end_epoch = int(ramp_end_epoch)
        self.edge_stop_epoch = int(edge_stop_epoch)
        self.diag_ambig_thresh = float(diag_ambig_thresh)

        if self.pol_dim <= 0:
            raise ValueError(f"PA_DAGP_POL_DIM must be positive, got {self.pol_dim}")
        if int(gn_groups) <= 0 or self.pol_dim % int(gn_groups) != 0:
            raise ValueError(
                f"PA_DAGP_POL_GN_GROUPS must divide PA_DAGP_POL_DIM, got {gn_groups} and {self.pol_dim}"
            )
        if str(act).lower() != "gelu":
            raise ValueError(f"PA-DAGP-v1 supports PA_DAGP_POL_ACT='gelu' only, got {act}")
        if self.anchor_weight_power <= 0.0 or self.anchor_eps <= 0.0:
            raise ValueError("PA-DAGP anchor power and epsilon must be positive.")
        if self.min_anchor_norm < 0.0:
            raise ValueError("PA_DAGP_MIN_ANCHOR_NORM must be non-negative.")
        if self.polarity_tau <= 0.0:
            raise ValueError("PA_DAGP_POLARITY_TAU must be positive.")
        if not 0.0 <= self.edge_cut_max <= 1.0:
            raise ValueError("PA_DAGP_EDGE_CUT_MAX must be in [0, 1].")
        if not 0.0 <= self.edge_gate_min <= 1.0:
            raise ValueError("PA_DAGP_EDGE_GATE_MIN must be in [0, 1].")
        if self.use_same_polarity_boost:
            raise ValueError("PA-DAGP-v1 forbids same-polarity edge boost.")
        if not self.renormalize_edge:
            raise ValueError("PA-DAGP-v1 requires PA_DAGP_RENORMALIZE_EDGE=True.")
        if self.start_epoch <= 0 or self.ramp_end_epoch < self.start_epoch:
            raise ValueError("Invalid PA-DAGP edge schedule.")
        if self.edge_stop_epoch <= self.ramp_end_epoch:
            raise ValueError("PA_DAGP_EDGE_STOP_EPOCH must be after the ramp end epoch.")
        if not 0.0 <= self.diag_ambig_thresh < 1.0:
            raise ValueError("PA_DAGP_DIAG_AMBIG_THRESH must be in [0, 1).")

        self.pol_proj = nn.Sequential(
            nn.Conv2d(int(in_channels), self.pol_dim, kernel_size=1, bias=False),
            nn.GroupNorm(int(gn_groups), self.pol_dim),
            nn.GELU(),
        )
        if self.use_calib_head:
            if int(calib_hidden) <= 0:
                raise ValueError("PA_DAGP_CALIB_HIDDEN must be positive.")
            self.calib_head = nn.Sequential(
                nn.Linear(self.pol_dim + 3, int(calib_hidden)),
                nn.GELU(),
                nn.Linear(int(calib_hidden), 1),
            )
            if bool(calib_zero_init):
                nn.init.zeros_(self.calib_head[-1].weight)
                nn.init.zeros_(self.calib_head[-1].bias)
        else:
            self.calib_head = None

    def edge_scale(self, epoch):
        epoch = int(epoch)
        if epoch < self.start_epoch or epoch >= self.edge_stop_epoch:
            return 0.0
        if epoch <= self.ramp_end_epoch:
            denom = max(1, self.ramp_end_epoch - self.start_epoch + 1)
            return float(epoch - self.start_epoch + 1) / float(denom)
        return 1.0

    @staticmethod
    def _gather_scalar_neighbors(values, indices):
        bsz, num_nodes = values.shape
        if indices.shape[:2] != (bsz, num_nodes):
            raise RuntimeError(
                "PA-DAGP top-k index shape mismatch: "
                f"values={list(values.shape)}, indices={list(indices.shape)}"
            )
        batch_offset = (
            torch.arange(bsz, device=indices.device, dtype=indices.dtype).view(bsz, 1, 1)
            * num_nodes
        )
        flat_indices = (indices + batch_offset).reshape(-1)
        return values.reshape(-1).index_select(0, flat_indices).reshape_as(indices)

    @staticmethod
    def _stats(values, mask=None):
        values = values.detach().float()
        if mask is not None:
            mask = mask.detach().bool()
            values = values[mask]
        if values.numel() == 0:
            zero = values.new_tensor(0.0)
            return zero, zero, zero
        return values.mean(), values.min(), values.max()

    def forward(self, feature, base_logits, topk_idx=None, edge_scale=0.0):
        if feature.ndim != 4 or base_logits.ndim != 4:
            raise RuntimeError(
                f"PA-DAGP expects 4D feature/logits, got {list(feature.shape)} and {list(base_logits.shape)}"
            )
        bsz, _, height, width = feature.shape
        if list(base_logits.shape) != [bsz, 1, height, width]:
            raise RuntimeError(
                "PA-DAGP base logits must match feature spatial shape, got "
                f"feature={list(feature.shape)}, base_logits={list(base_logits.shape)}"
            )
        num_nodes = height * width
        edge_scale = float(edge_scale)

        base_prob = torch.sigmoid(base_logits)
        base_prob_for_anchor = base_prob.detach() if self.detach_base_prob else base_prob
        pol_feat = F.normalize(self.pol_proj(feature), dim=1)
        tokens = pol_feat.flatten(2).transpose(1, 2)
        prob_tokens = base_prob_for_anchor.flatten(2).transpose(1, 2)

        weight_fg = (prob_tokens + self.anchor_eps).pow(self.anchor_weight_power)
        weight_bg = (1.0 - prob_tokens + self.anchor_eps).pow(self.anchor_weight_power)
        weight_fg = weight_fg / weight_fg.sum(dim=1, keepdim=True).clamp_min(self.anchor_eps)
        weight_bg = weight_bg / weight_bg.sum(dim=1, keepdim=True).clamp_min(self.anchor_eps)
        anchor_fg_raw = (weight_fg * tokens).sum(dim=1)
        anchor_bg_raw = (weight_bg * tokens).sum(dim=1)
        anchor_fg_norm = anchor_fg_raw.norm(dim=-1)
        anchor_bg_norm = anchor_bg_raw.norm(dim=-1)
        anchor_valid = (anchor_fg_norm > self.min_anchor_norm) & (anchor_bg_norm > self.min_anchor_norm)
        anchor_fg = F.normalize(anchor_fg_raw, dim=-1)
        anchor_bg = F.normalize(anchor_bg_raw, dim=-1)
        anchor_fg_margin = anchor_fg.detach() if self.detach_anchors else anchor_fg
        anchor_bg_margin = anchor_bg.detach() if self.detach_anchors else anchor_bg

        sim_fg = (tokens * anchor_fg_margin[:, None, :]).sum(dim=-1)
        sim_bg = (tokens * anchor_bg_margin[:, None, :]).sum(dim=-1)
        anchor_margin = sim_fg - sim_bg
        prob_scalar = prob_tokens.squeeze(-1)
        base_uncert = (1.0 - 2.0 * torch.abs(prob_scalar - 0.5)).clamp(0.0, 1.0)
        if self.calib_head is not None:
            calib_input = torch.cat(
                (
                    tokens,
                    anchor_margin.unsqueeze(-1),
                    prob_scalar.unsqueeze(-1),
                    base_uncert.unsqueeze(-1),
                ),
                dim=-1,
            )
            delta_margin = self.calib_head(calib_input).squeeze(-1)
        else:
            delta_margin = torch.zeros_like(anchor_margin)
        raw_polarity = anchor_margin + delta_margin
        raw_polarity = torch.where(anchor_valid[:, None], raw_polarity, torch.zeros_like(raw_polarity))
        signed_polarity = torch.tanh(raw_polarity / self.polarity_tau)

        anchor_cosine = (anchor_fg * anchor_bg).sum(dim=-1).clamp(-1.0, 1.0)
        rho = ((1.0 - anchor_cosine) * 0.5).clamp(0.0, 1.0)
        rho = torch.where(anchor_valid, rho, torch.zeros_like(rho))
        rho_gate = rho.detach() if self.detach_rho else rho
        edge_cut_eff = self.edge_cut_max * edge_scale

        polarity_gate = None
        cross_score = None
        if topk_idx is not None and edge_scale > 0.0:
            if topk_idx.ndim != 3 or topk_idx.shape[:2] != (bsz, num_nodes):
                raise RuntimeError(
                    f"PA-DAGP top-k indices must be [B,N,K], got {list(topk_idx.shape)}"
                )
            center_polarity = signed_polarity.unsqueeze(-1)
            neighbor_polarity = self._gather_scalar_neighbors(signed_polarity, topk_idx)
            cross_score = F.relu(-(center_polarity * neighbor_polarity))
            polarity_gate = 1.0 - edge_cut_eff * rho_gate[:, None, None] * cross_score
            polarity_gate = polarity_gate.clamp(min=self.edge_gate_min, max=1.0)
            if not bool(torch.isfinite(polarity_gate).all().item()):
                raise RuntimeError("PA-DAGP polarity gate contains NaN/Inf.")
            gate_min = float(polarity_gate.detach().min().item())
            gate_max = float(polarity_gate.detach().max().item())
            if gate_min < self.edge_gate_min - 1e-5 or gate_max > 1.0 + 1e-5:
                raise RuntimeError(
                    f"PA-DAGP polarity gate out of range: min={gate_min}, max={gate_max}"
                )

        valid_token_mask = anchor_valid[:, None].expand(-1, num_nodes)
        raw_mean, raw_min, raw_max = self._stats(raw_polarity, valid_token_mask)
        signed_mean, signed_min, signed_max = self._stats(signed_polarity, valid_token_mask)
        if bool(valid_token_mask.any().item()):
            raw_std = raw_polarity.detach().float()[valid_token_mask].std(unbiased=False)
            signed_std = signed_polarity.detach().float()[valid_token_mask].std(unbiased=False)
            valid_signed = signed_polarity.detach()[valid_token_mask]
            positive_ratio = (valid_signed > self.diag_ambig_thresh).float().mean()
            negative_ratio = (valid_signed < -self.diag_ambig_thresh).float().mean()
            ambiguous_ratio = 1.0 - positive_ratio - negative_ratio
        else:
            raw_std = raw_polarity.new_tensor(0.0)
            signed_std = raw_polarity.new_tensor(0.0)
            positive_ratio = raw_polarity.new_tensor(0.0)
            negative_ratio = raw_polarity.new_tensor(0.0)
            ambiguous_ratio = raw_polarity.new_tensor(1.0)

        if polarity_gate is not None and cross_score is not None and bool(anchor_valid.any().item()):
            valid_edge_mask = anchor_valid[:, None, None].expand_as(polarity_gate)
            gate_values = polarity_gate.detach()[valid_edge_mask]
            cross_values = cross_score.detach()[valid_edge_mask]
            gate_mean = gate_values.mean()
            gate_min_tensor = gate_values.min()
            gate_max_tensor = gate_values.max()
            cross_edge_ratio = (cross_values > 0.0).float().mean()
            edge_suppressed_ratio = (gate_values < 0.95).float().mean()
        else:
            gate_mean = raw_polarity.new_tensor(1.0)
            gate_min_tensor = raw_polarity.new_tensor(1.0)
            gate_max_tensor = raw_polarity.new_tensor(1.0)
            cross_edge_ratio = raw_polarity.new_tensor(0.0)
            edge_suppressed_ratio = raw_polarity.new_tensor(0.0)

        fg_norm_mean, fg_norm_min, fg_norm_max = self._stats(anchor_fg_norm)
        bg_norm_mean, bg_norm_min, bg_norm_max = self._stats(anchor_bg_norm)
        anchor_cos_mean, anchor_cos_min, anchor_cos_max = self._stats(anchor_cosine, anchor_valid)
        rho_mean, rho_min, rho_max = self._stats(rho, anchor_valid)
        diagnostics = {
            "edge_scale": raw_polarity.new_tensor(edge_scale).detach(),
            "edge_cut_eff": raw_polarity.new_tensor(edge_cut_eff).detach(),
            "anchor_valid_ratio": anchor_valid.float().mean().detach(),
            "anchor_fg_norm_mean": fg_norm_mean,
            "anchor_fg_norm_min": fg_norm_min,
            "anchor_fg_norm_max": fg_norm_max,
            "anchor_bg_norm_mean": bg_norm_mean,
            "anchor_bg_norm_min": bg_norm_min,
            "anchor_bg_norm_max": bg_norm_max,
            "anchor_cosine_mean": anchor_cos_mean,
            "anchor_cosine_min": anchor_cos_min,
            "anchor_cosine_max": anchor_cos_max,
            "rho_mean": rho_mean,
            "rho_min": rho_min,
            "rho_max": rho_max,
            "raw_pol_mean": raw_mean,
            "raw_pol_std": raw_std.detach(),
            "raw_pol_min": raw_min,
            "raw_pol_max": raw_max,
            "signed_pol_mean": signed_mean,
            "signed_pol_std": signed_std.detach(),
            "signed_pol_min": signed_min,
            "signed_pol_max": signed_max,
            "positive_ratio": positive_ratio.detach(),
            "negative_ratio": negative_ratio.detach(),
            "ambiguous_ratio": ambiguous_ratio.detach(),
            "cross_edge_ratio": cross_edge_ratio.detach(),
            "gate_mean": gate_mean.detach(),
            "gate_min": gate_min_tensor.detach(),
            "gate_max": gate_max_tensor.detach(),
            "edge_suppressed_ratio": edge_suppressed_ratio.detach(),
        }
        return {
            "polarity_gate": polarity_gate,
            "raw_polarity": raw_polarity.reshape(bsz, 1, height, width),
            "signed_polarity": signed_polarity.reshape(bsz, 1, height, width),
            "base_prob": base_prob.detach(),
            "anchor_valid": anchor_valid.detach(),
            "rho": rho.detach(),
            "diagnostics": diagnostics,
        }


class DAGPSafeHead(nn.Module):
    def __init__(
        self,
        in_channels=384,
        hidden=64,
        topk=12,
        tau=0.07,
        alpha_max=0.05,
        gamma_max=0.03,
        warmup_epoch=6,
        ramp_start_epoch=7,
        ramp_end_epoch=15,
        affinity_detach=True,
        exclude_self=True,
        use_prob_gate=True,
        prob_gate_sigma=0.25,
        prob_gate_eps=1e-6,
        zero_init_graph_pred=True,
        zero_init_value=False,
        use_uncertainty_output_gate=False,
        uncertainty_power=1.0,
        uncertainty_min=0.0,
        uncertainty_max=1.0,
        uncertainty_detach=True,
        use_ndr_branch=False,
        ndr_loss_size=68,
        ndr_input_rgb=True,
        ndr_input_sobel=True,
        ndr_input_coarse_prob=True,
        ndr_in_channels=5,
        ndr_hidden=32,
        ndr_num_layers=3,
        ndr_use_gn=True,
        ndr_gn_groups=4,
        ndr_act="gelu",
        ndr_zero_init_out=True,
        ndr_residual_clip=2.0,
        ndr_beta_max=0.10,
        ndr_warmup_epoch=6,
        ndr_ramp_start_epoch=7,
        ndr_ramp_end_epoch=15,
        ndr_use_uncertainty_gate=True,
        ndr_use_edge_gate=True,
        ndr_gate_mode="uncertainty_edge_boost",
        use_ndr_v2=False,
        ndr_version="v1",
        ndr_v2_use_shape_gate=False,
        ndr_v2_shape_gate_mode="soft_boundary_edge_boost",
        ndr_v2_boundary_source="coarse_prob",
        ndr_v2_boundary_radius=2,
        ndr_v2_boundary_detach=True,
        ndr_v2_shape_alpha_max=0.20,
        ndr_v2_shape_edge_mix=0.50,
        ndr_v2_shape_uncert_mix=0.50,
        ndr_v2_gate_combine="add_clamp",
        use_tadr_router=False,
        tadr_router_in_channels=4,
        tadr_router_hidden=16,
        tadr_router_num_layers=2,
        tadr_router_act="gelu",
        tadr_use_coarse_prob=True,
        tadr_use_uncertainty=True,
        tadr_use_sobel=True,
        tadr_use_coarse_boundary=True,
        tadr_router_init_bias=2.0,
        tadr_router_zero_init_out=True,
        tadr_router_detach_inputs=True,
        tadr_router_min=0.0,
        tadr_router_max=1.0,
        use_proto_contrast=False,
        proto_feature_source="dagp_semantic",
        proto_use_proj_head=True,
        proto_proj_hidden=64,
        proto_proj_dim=32,
        proto_proj_act="gelu",
        pa_dagp=None,
        use_esa_ber=False,
        return_graph_aux_for_ber=False,
        esa_ber_start_epoch=21,
        esa_ber_stop_epoch=36,
    ):
        super().__init__()
        if int(hidden) <= 0:
            raise ValueError(f"DAGP_SAFE_HIDDEN must be positive, got {hidden}")
        if int(topk) <= 0:
            raise ValueError(f"DAGP_SAFE_TOPK must be positive, got {topk}")
        if float(tau) <= 0.0:
            raise ValueError(f"DAGP_SAFE_TAU must be positive, got {tau}")
        if float(prob_gate_sigma) <= 0.0:
            raise ValueError(f"DAGP_SAFE_PROB_GATE_SIGMA must be positive, got {prob_gate_sigma}")
        if float(prob_gate_eps) <= 0.0:
            raise ValueError(f"DAGP_SAFE_PROB_GATE_EPS must be positive, got {prob_gate_eps}")
        if float(uncertainty_max) < float(uncertainty_min):
            raise ValueError(
                "DAGP_SAFE_UNCERTAINTY_MAX must be greater than or equal to "
                f"DAGP_SAFE_UNCERTAINTY_MIN, got max={uncertainty_max}, min={uncertainty_min}"
            )

        self.hidden = int(hidden)
        self.topk = int(topk)
        self.tau = float(tau)
        self.alpha_max = float(alpha_max)
        self.gamma_max = float(gamma_max)
        self.warmup_epoch = int(warmup_epoch)
        self.ramp_start_epoch = int(ramp_start_epoch)
        self.ramp_end_epoch = int(ramp_end_epoch)
        self.affinity_detach = bool(affinity_detach)
        self.exclude_self = bool(exclude_self)
        self.use_prob_gate = bool(use_prob_gate)
        self.prob_gate_sigma = float(prob_gate_sigma)
        self.prob_gate_eps = float(prob_gate_eps)
        self.use_uncertainty_output_gate = bool(use_uncertainty_output_gate)
        self.uncertainty_power = float(uncertainty_power)
        self.uncertainty_min = float(uncertainty_min)
        self.uncertainty_max = float(uncertainty_max)
        self.uncertainty_detach = bool(uncertainty_detach)
        self.use_ndr_branch = bool(use_ndr_branch)
        self.ndr_loss_size = int(ndr_loss_size)
        self.ndr_beta_max = float(ndr_beta_max)
        self.ndr_warmup_epoch = int(ndr_warmup_epoch)
        self.ndr_ramp_start_epoch = int(ndr_ramp_start_epoch)
        self.ndr_ramp_end_epoch = int(ndr_ramp_end_epoch)
        self.ndr_use_uncertainty_gate = bool(ndr_use_uncertainty_gate)
        self.ndr_use_edge_gate = bool(ndr_use_edge_gate)
        self.ndr_gate_mode = str(ndr_gate_mode)
        self.use_ndr_v2 = bool(use_ndr_v2)
        self.ndr_version = str(ndr_version)
        self.ndr_v2_use_shape_gate = bool(ndr_v2_use_shape_gate)
        self.ndr_v2_shape_gate_mode = str(ndr_v2_shape_gate_mode)
        self.ndr_v2_boundary_source = str(ndr_v2_boundary_source)
        self.ndr_v2_boundary_radius = int(ndr_v2_boundary_radius)
        self.ndr_v2_boundary_detach = bool(ndr_v2_boundary_detach)
        self.ndr_v2_shape_alpha_max = float(ndr_v2_shape_alpha_max)
        self.ndr_v2_shape_edge_mix = float(ndr_v2_shape_edge_mix)
        self.ndr_v2_shape_uncert_mix = float(ndr_v2_shape_uncert_mix)
        self.ndr_v2_gate_combine = str(ndr_v2_gate_combine)
        self.use_tadr_router = bool(use_tadr_router)
        self.tadr_router_in_channels = int(tadr_router_in_channels)
        self.tadr_use_coarse_prob = bool(tadr_use_coarse_prob)
        self.tadr_use_uncertainty = bool(tadr_use_uncertainty)
        self.tadr_use_sobel = bool(tadr_use_sobel)
        self.tadr_use_coarse_boundary = bool(tadr_use_coarse_boundary)
        self.tadr_router_detach_inputs = bool(tadr_router_detach_inputs)
        self.tadr_router_min = float(tadr_router_min)
        self.tadr_router_max = float(tadr_router_max)
        self.use_proto_contrast = bool(use_proto_contrast)
        self.proto_feature_source = str(proto_feature_source)
        self.proto_use_proj_head = bool(proto_use_proj_head)
        self.proto_proj_hidden = int(proto_proj_hidden)
        self.proto_proj_dim = int(proto_proj_dim)
        self.proto_proj_act = str(proto_proj_act).lower()
        self.pa_dagp = pa_dagp
        self.use_pa_dagp = pa_dagp is not None
        self.use_esa_ber = bool(use_esa_ber)
        self.return_graph_aux_for_ber = bool(return_graph_aux_for_ber)
        self.esa_ber_start_epoch = int(esa_ber_start_epoch)
        self.esa_ber_stop_epoch = int(esa_ber_stop_epoch)

        self.base_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.proj = nn.Conv2d(in_channels, self.hidden, kernel_size=1)
        self.value = nn.Linear(self.hidden, self.hidden)
        self.graph_pred = nn.Conv2d(self.hidden, 1, kernel_size=1)
        self.register_buffer("current_epoch_tensor", torch.zeros(1, dtype=torch.float32))
        if self.use_proto_contrast:
            if self.proto_feature_source != "dagp_semantic":
                raise ValueError(
                    f"PROTO_FEATURE_SOURCE currently supports only 'dagp_semantic', got {self.proto_feature_source}"
                )
            if not self.proto_use_proj_head:
                raise ValueError("PROTO_USE_PROJ_HEAD=False is not implemented for MVFlip-Proto.")
            if self.proto_proj_hidden <= 0:
                raise ValueError(f"PROTO_PROJ_HIDDEN must be positive, got {self.proto_proj_hidden}")
            if self.proto_proj_dim <= 0:
                raise ValueError(f"PROTO_PROJ_DIM must be positive, got {self.proto_proj_dim}")
            if self.proto_proj_act == "gelu":
                proto_act = nn.GELU()
            elif self.proto_proj_act == "relu":
                proto_act = nn.ReLU(inplace=True)
            else:
                raise ValueError(f"Unsupported PROTO_PROJ_ACT: {self.proto_proj_act}")
            self.proto_proj_head = nn.Sequential(
                nn.Conv2d(self.hidden, self.proto_proj_hidden, kernel_size=1),
                proto_act,
                nn.Conv2d(self.proto_proj_hidden, self.proto_proj_dim, kernel_size=1),
            )
        else:
            self.proto_proj_head = None
        if self.use_ndr_v2 and not self.use_ndr_branch:
            raise ValueError("USE_NDR_V2=True requires USE_NDR_BRANCH=True.")
        if self.use_ndr_v2:
            if self.ndr_v2_shape_gate_mode != "soft_boundary_edge_boost":
                raise ValueError(f"Unsupported NDR_V2_SHAPE_GATE_MODE: {self.ndr_v2_shape_gate_mode}")
            if self.ndr_v2_boundary_source != "coarse_prob":
                raise ValueError(f"Unsupported NDR_V2_BOUNDARY_SOURCE: {self.ndr_v2_boundary_source}")
            if self.ndr_v2_boundary_radius < 0:
                raise ValueError(f"NDR_V2_BOUNDARY_RADIUS must be non-negative, got {self.ndr_v2_boundary_radius}")
            if self.ndr_v2_shape_alpha_max < 0.0:
                raise ValueError(f"NDR_V2_SHAPE_ALPHA_MAX must be non-negative, got {self.ndr_v2_shape_alpha_max}")
            if not 0.0 <= self.ndr_v2_shape_edge_mix <= 1.0:
                raise ValueError(f"NDR_V2_SHAPE_EDGE_MIX must be in [0, 1], got {self.ndr_v2_shape_edge_mix}")
            if not 0.0 <= self.ndr_v2_shape_uncert_mix <= 1.0:
                raise ValueError(f"NDR_V2_SHAPE_UNCERT_MIX must be in [0, 1], got {self.ndr_v2_shape_uncert_mix}")
            if self.ndr_v2_gate_combine != "add_clamp":
                raise ValueError(f"Unsupported NDR_V2_GATE_COMBINE: {self.ndr_v2_gate_combine}")
        if self.use_ndr_branch:
            if self.ndr_loss_size <= 0:
                raise ValueError(f"LOSS_SIZE for NDR must be positive, got {self.ndr_loss_size}")
            if self.ndr_beta_max < 0.0:
                raise ValueError(f"NDR_BETA_MAX must be non-negative, got {self.ndr_beta_max}")
            if self.ndr_gate_mode not in {"uncertainty_edge_boost", "uncertainty_edge_stronger"}:
                raise ValueError(f"Unsupported NDR_GATE_MODE: {self.ndr_gate_mode}")
            if self.tadr_router_min > self.tadr_router_max:
                raise ValueError(
                    f"TADR_ROUTER_MIN must be <= TADR_ROUTER_MAX, got "
                    f"{self.tadr_router_min} > {self.tadr_router_max}"
                )
            self.ndr_branch = NativeDetailResidualBranch(
                in_channels=int(ndr_in_channels),
                hidden=int(ndr_hidden),
                num_layers=int(ndr_num_layers),
                use_gn=bool(ndr_use_gn),
                gn_groups=int(ndr_gn_groups),
                act=str(ndr_act),
                zero_init_out=bool(ndr_zero_init_out),
                residual_clip=float(ndr_residual_clip),
                input_rgb=bool(ndr_input_rgb),
                input_sobel=bool(ndr_input_sobel),
                input_coarse_prob=bool(ndr_input_coarse_prob),
            )
            if self.use_tadr_router:
                expected_tadr_channels = int(self.tadr_use_coarse_prob) + int(self.tadr_use_uncertainty)
                expected_tadr_channels += int(self.tadr_use_sobel) + int(self.tadr_use_coarse_boundary)
                if self.tadr_router_in_channels != expected_tadr_channels:
                    raise ValueError(
                        f"TADR_ROUTER_IN_CHANNELS={self.tadr_router_in_channels} does not match enabled "
                        f"TADR inputs ({expected_tadr_channels} channels)"
                    )
                self.tadr_router = AdaptiveDetailRouter(
                    in_channels=self.tadr_router_in_channels,
                    hidden=int(tadr_router_hidden),
                    num_layers=int(tadr_router_num_layers),
                    act=str(tadr_router_act),
                    init_bias=float(tadr_router_init_bias),
                    zero_init_out=bool(tadr_router_zero_init_out),
                )
            else:
                self.tadr_router = None
        else:
            if self.use_tadr_router:
                raise ValueError("USE_TADR_ROUTER=True requires USE_NDR_BRANCH=True")
            self.ndr_branch = None
            self.tadr_router = None

        if bool(zero_init_graph_pred):
            nn.init.zeros_(self.graph_pred.weight)
            nn.init.zeros_(self.graph_pred.bias)
        if bool(zero_init_value):
            nn.init.zeros_(self.value.weight)
            nn.init.zeros_(self.value.bias)

    def set_epoch(self, epoch):
        self.current_epoch_tensor.fill_(float(epoch))

    def _ramp_scale(self):
        epoch = int(self.current_epoch_tensor.item())
        if epoch <= self.warmup_epoch:
            return 0.0
        if epoch >= self.ramp_end_epoch:
            return 1.0
        if epoch < self.ramp_start_epoch:
            return 0.0
        denom = max(1, self.ramp_end_epoch - self.ramp_start_epoch + 1)
        return float(epoch - self.ramp_start_epoch + 1) / float(denom)

    def _ndr_ramp_scale(self):
        epoch = int(self.current_epoch_tensor.item())
        if epoch <= self.ndr_warmup_epoch:
            return 0.0
        if epoch >= self.ndr_ramp_end_epoch:
            return 1.0
        if epoch < self.ndr_ramp_start_epoch:
            return 0.0
        denom = max(1, self.ndr_ramp_end_epoch - self.ndr_ramp_start_epoch + 1)
        return float(epoch - self.ndr_ramp_start_epoch + 1) / float(denom)

    def _ndr_beta_eff(self):
        return self.ndr_beta_max * self._ndr_ramp_scale()

    def _ndr_v2_shape_alpha_eff(self):
        if not self.use_ndr_v2 or not self.ndr_v2_use_shape_gate:
            return 0.0
        return self.ndr_v2_shape_alpha_max * self._ndr_ramp_scale()

    @staticmethod
    def _gather_neighbors(v, idx):
        bsz, num_nodes, dim = v.shape
        k = idx.shape[-1]
        batch_offset = torch.arange(bsz, device=idx.device, dtype=idx.dtype).view(bsz, 1, 1) * num_nodes
        flat_idx = (idx + batch_offset).reshape(-1)
        return v.reshape(bsz * num_nodes, dim).index_select(0, flat_idx).reshape(bsz, num_nodes, k, dim)

    def _topk_affinity(self, feat):
        _, _, height, width = feat.shape
        num_nodes = height * width
        if num_nodes <= 1:
            raise ValueError(f"DAGP-Safe requires at least 2 spatial nodes, got H={height}, W={width}")

        x = feat.flatten(2).transpose(1, 2)
        if self.affinity_detach:
            x = x.detach()
        x_aff = F.normalize(x.float(), dim=-1)
        sim = torch.bmm(x_aff, x_aff.transpose(1, 2))
        if self.exclude_self:
            diag = torch.eye(num_nodes, device=sim.device, dtype=torch.bool).unsqueeze(0)
            sim = sim.masked_fill(diag, -float("inf"))
            k = min(self.topk, num_nodes - 1)
        else:
            k = min(self.topk, num_nodes)
        topk_val, topk_idx = torch.topk(sim, k=k, dim=-1)
        attn = torch.softmax(topk_val / self.tau, dim=-1)
        return attn, topk_idx

    def _apply_prob_gate(self, attn, topk_idx, base_logits):
        prob = torch.sigmoid(base_logits.detach()).flatten(2).transpose(1, 2)
        prob_neigh = self._gather_neighbors(prob, topk_idx)
        prob_i = prob.unsqueeze(2)
        prob_diff = torch.abs(prob_i - prob_neigh).squeeze(-1)
        prob_gate = torch.exp(-prob_diff / self.prob_gate_sigma)
        attn = attn * prob_gate
        return attn / (attn.sum(dim=-1, keepdim=True) + self.prob_gate_eps)

    def _uncertainty_output_gate(self, base_logits):
        if not self.use_uncertainty_output_gate:
            return None
        p_base = torch.sigmoid(base_logits)
        if self.uncertainty_detach:
            p_base = p_base.detach()
        gate = 1.0 - 2.0 * torch.abs(p_base - 0.5)
        gate = torch.clamp(gate, min=self.uncertainty_min, max=self.uncertainty_max)
        if self.uncertainty_power != 1.0:
            gate = gate.pow(self.uncertainty_power)
        return gate

    def _uncertainty_stats(self, base_logits, gate=None):
        if not self.use_uncertainty_output_gate:
            disabled = base_logits.new_tensor(-1.0)
            return disabled, disabled, disabled
        if gate is None:
            gate = self._uncertainty_output_gate(base_logits)
        gate_for_stats = gate.detach()
        return gate_for_stats.mean(), gate_for_stats.min(), gate_for_stats.max()

    def _proto_features(self, semantic_feat):
        if not self.use_proto_contrast:
            return None, None
        proto_feat_raw = F.interpolate(
            semantic_feat,
            size=(self.ndr_loss_size, self.ndr_loss_size),
            mode="bilinear",
            align_corners=False,
        )
        proto_feat = self.proto_proj_head(proto_feat_raw)
        proto_feat = F.normalize(proto_feat, dim=1)
        return proto_feat_raw, proto_feat

    def _attach_proto_aux(self, output, semantic_feat):
        if not self.use_proto_contrast or semantic_feat is None:
            return output
        proto_feat_raw, proto_feat = self._proto_features(semantic_feat)
        output["proto_feat_raw"] = proto_feat_raw
        output["proto_feat"] = proto_feat
        output["prob"] = torch.sigmoid(output["logits"])
        if "coarse_logits_68" in output:
            output["coarse_logits"] = output["coarse_logits_68"]
            output["coarse_prob"] = torch.sigmoid(output["coarse_logits_68"])
        elif "coarse_logits_37" in output:
            coarse_logits = F.interpolate(
                output["coarse_logits_37"],
                size=(self.ndr_loss_size, self.ndr_loss_size),
                mode="bilinear",
                align_corners=False,
            )
            output["coarse_logits"] = coarse_logits
            output["coarse_prob"] = torch.sigmoid(coarse_logits)
        else:
            output["coarse_logits"] = output["logits"]
            output["coarse_prob"] = output["prob"]
        return output

    @staticmethod
    def _attach_pa_aux(output, pa_state):
        if pa_state is None:
            return output
        output.update(
            {
                "pa_raw_polarity": pa_state["raw_polarity"],
                "pa_signed_polarity": pa_state["signed_polarity"],
                "pa_base_prob": pa_state["base_prob"],
                "pa_anchor_valid": pa_state["anchor_valid"],
                "pa_rho": pa_state["rho"],
                "pa_diag": pa_state["diagnostics"],
            }
        )
        return output

    def _ber_graph_aux_enabled(self, return_aux):
        if not bool(return_aux) or not self.use_esa_ber or not self.return_graph_aux_for_ber:
            return False
        epoch = int(self.current_epoch_tensor.item())
        return self.esa_ber_start_epoch <= epoch < self.esa_ber_stop_epoch

    @torch.no_grad()
    def forward_coarse_only(self, feat, epoch=None, return_logits_37=True):
        """Run only the native base projection and DAGP coarse path.

        This read-only interface is used by BITC counterfactual feature
        interventions.  It deliberately skips RGB/Sobel inputs, NDR and every
        final-refinement branch while sharing the current EMA-Teacher weights.
        """
        if not bool(return_logits_37):
            raise ValueError("forward_coarse_only requires return_logits_37=True")
        if self.training:
            raise RuntimeError(
                "forward_coarse_only is a read-only EMA-Teacher eval interface"
            )
        if feat.ndim != 4:
            raise RuntimeError(
                f"forward_coarse_only expects [B,C,H,W], got {list(feat.shape)}"
            )
        current_epoch = int(self.current_epoch_tensor.item())
        if epoch is not None and int(epoch) != current_epoch:
            raise RuntimeError(
                f"forward_coarse_only epoch mismatch: requested={int(epoch)}, "
                f"model={current_epoch}"
            )
        batch_size, _, height, width = feat.shape
        base_logits = self.base_head(feat)
        scale = self._ramp_scale()
        alpha_eff = self.alpha_max * scale
        gamma_eff = self.gamma_max * scale
        if alpha_eff == 0.0 or gamma_eff == 0.0:
            return base_logits.detach()

        attn, topk_idx = self._topk_affinity(feat)
        if self.use_prob_gate:
            attn = self._apply_prob_gate(attn, topk_idx, base_logits)
        if self.pa_dagp is not None:
            edge_scale = self.pa_dagp.edge_scale(current_epoch)
            if edge_scale > 0.0:
                pa_state = self.pa_dagp(
                    feat,
                    base_logits,
                    topk_idx=topk_idx,
                    edge_scale=edge_scale,
                )
                polarity_gate = pa_state["polarity_gate"]
                if polarity_gate is None or polarity_gate.shape != attn.shape:
                    raise RuntimeError(
                        "forward_coarse_only PA-DAGP polarity gate shape mismatch"
                    )
                attn = attn * polarity_gate.to(dtype=attn.dtype)
                attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(
                    self.prob_gate_eps
                )

        projected = self.proj(feat)
        projected_nodes = projected.flatten(2).transpose(1, 2)
        value = self.value(projected_nodes)
        neighbor_value = self._gather_neighbors(value, topk_idx)
        aggregate = (
            attn.to(dtype=value.dtype).unsqueeze(-1) * neighbor_value
        ).sum(dim=2)
        propagated = projected_nodes + value.new_tensor(float(gamma_eff)) * aggregate
        propagated_map = propagated.transpose(1, 2).reshape(
            batch_size, self.hidden, height, width
        )
        graph_logits = self.graph_pred(propagated_map)
        uncertainty_gate = self._uncertainty_output_gate(base_logits)
        graph_residual = (
            graph_logits
            if uncertainty_gate is None
            else uncertainty_gate * graph_logits
        )
        coarse_logits = base_logits + graph_logits.new_tensor(
            float(alpha_eff)
        ) * graph_residual
        if coarse_logits.requires_grad:
            coarse_logits = coarse_logits.detach()
        return coarse_logits

    @staticmethod
    def _attach_ber_graph_aux(output, topk_idx, semantic_weight):
        if topk_idx is None or semantic_weight is None:
            return output
        topk_idx = topk_idx.detach()
        semantic_weight = semantic_weight.detach()
        if topk_idx.ndim != 3 or semantic_weight.shape != topk_idx.shape:
            raise RuntimeError(
                "ESA-BER graph auxiliary shape mismatch: "
                f"idx={list(topk_idx.shape)}, weight={list(semantic_weight.shape)}"
            )
        if topk_idx.requires_grad or semantic_weight.requires_grad:
            raise RuntimeError("ESA-BER graph auxiliary tensors must be detached.")
        if not bool(torch.isfinite(semantic_weight).all().item()):
            raise RuntimeError("ESA-BER semantic top-k weights contain NaN/Inf.")
        sum_error = (semantic_weight.float().sum(dim=-1) - 1.0).abs().max()
        if float(sum_error.item()) > 1e-5:
            raise RuntimeError(
                "ESA-BER semantic top-k weights are not normalized: "
                f"max_error={float(sum_error.item()):.8g}"
            )
        output["dagp_topk_idx"] = topk_idx
        output["dagp_topk_sem_weight"] = semantic_weight
        output["dagp_topk_sem_weight_sum_error"] = sum_error.detach()
        return output

    def _aux_output(
        self,
        logits,
        base_logits,
        graph_logits,
        scale,
        alpha_eff,
        gamma_eff,
        uncertainty_gate=None,
        semantic_feat=None,
        pa_state=None,
        pa_original_logits=None,
    ):
        scalar = base_logits.new_tensor(float(scale))
        unc_mean, unc_min, unc_max = self._uncertainty_stats(base_logits, uncertainty_gate)
        output = {
            "logits": logits,
            "base_logits": base_logits,
            "graph_logits": graph_logits,
            "dagp_scale": scalar,
            "dagp_alpha_eff": base_logits.new_tensor(float(alpha_eff)),
            "dagp_gamma_eff": base_logits.new_tensor(float(gamma_eff)),
            "uncertainty_gate_mean": unc_mean,
            "uncertainty_gate_min": unc_min,
            "uncertainty_gate_max": unc_max,
        }
        if pa_original_logits is not None:
            output["pa_original_coarse_logits_native"] = pa_original_logits.detach()
        output = self._attach_pa_aux(output, pa_state)
        return self._attach_proto_aux(output, semantic_feat)

    @staticmethod
    def _tensor_stats(tensor):
        tensor_detached = tensor.detach()
        return tensor_detached.mean(), tensor_detached.min(), tensor_detached.max()

    @staticmethod
    def _normalize_map_per_image(tensor, eps=1e-6):
        flat = tensor.detach().flatten(1)
        min_val = flat.min(dim=1).values.view(-1, 1, 1, 1)
        max_val = flat.max(dim=1).values.view(-1, 1, 1, 1)
        return torch.clamp((tensor.detach() - min_val) / (max_val - min_val + float(eps)), 0.0, 1.0)

    @staticmethod
    def _soft_morph_boundary(prob, radius=2):
        radius = int(radius)
        if radius <= 0:
            return torch.zeros_like(prob)
        kernel_size = 2 * radius + 1
        dilated = F.max_pool2d(prob, kernel_size=kernel_size, stride=1, padding=radius)
        eroded = -F.max_pool2d(-prob, kernel_size=kernel_size, stride=1, padding=radius)
        return torch.clamp(dilated - eroded, 0.0, 1.0)

    def _apply_ndr_v2_shape_gate(self, detail_gate_v1, coarse_prob_68, sobel_68, uncertainty_68):
        if not self.use_ndr_v2 or not self.ndr_v2_use_shape_gate:
            empty = torch.zeros_like(detail_gate_v1)
            zero = detail_gate_v1.new_tensor(0.0)
            return detail_gate_v1, {
                "detail_gate_v1": detail_gate_v1,
                "detail_gate_v2": detail_gate_v1,
                "boundary_band_68": empty,
                "edge_norm_68": empty,
                "shape_boost_68": empty,
                "ndr_v2_shape_alpha_eff": zero,
            }
        boundary_source = coarse_prob_68.detach() if self.ndr_v2_boundary_detach else coarse_prob_68
        boundary_band = self._soft_morph_boundary(boundary_source, self.ndr_v2_boundary_radius)
        edge_norm = self._normalize_map_per_image(sobel_68)
        uncertainty_shape = uncertainty_68.detach()
        edge_factor = (1.0 - self.ndr_v2_shape_edge_mix) + self.ndr_v2_shape_edge_mix * edge_norm
        uncertainty_factor = (
            (1.0 - self.ndr_v2_shape_uncert_mix)
            + self.ndr_v2_shape_uncert_mix * uncertainty_shape
        )
        shape_boost = torch.clamp(boundary_band * edge_factor * uncertainty_factor, 0.0, 1.0)
        shape_alpha_eff = float(self._ndr_v2_shape_alpha_eff())
        detail_gate_v2 = torch.clamp(
            detail_gate_v1 + detail_gate_v1.new_tensor(shape_alpha_eff) * shape_boost,
            0.0,
            1.0,
        )
        return detail_gate_v2, {
            "detail_gate_v1": detail_gate_v1,
            "detail_gate_v2": detail_gate_v2,
            "boundary_band_68": boundary_band,
            "edge_norm_68": edge_norm,
            "shape_boost_68": shape_boost,
            "ndr_v2_shape_alpha_eff": detail_gate_v1.new_tensor(shape_alpha_eff),
        }

    def _build_tadr_router_input(self, coarse_prob_68, uncertainty_68, sobel_68, coarse_boundary_68):
        inputs = []
        if self.tadr_use_coarse_prob:
            inputs.append(coarse_prob_68)
        if self.tadr_use_uncertainty:
            inputs.append(uncertainty_68)
        if self.tadr_use_sobel:
            inputs.append(sobel_68)
        if self.tadr_use_coarse_boundary:
            inputs.append(coarse_boundary_68)
        if not inputs:
            raise ValueError("USE_TADR_ROUTER=True requires at least one enabled TADR input.")
        if self.tadr_router_detach_inputs:
            inputs = [tensor.detach() for tensor in inputs]
        router_input = torch.cat(inputs, dim=1)
        if router_input.shape[1] != self.tadr_router_in_channels:
            raise ValueError(
                f"TADR router input channels mismatch: got {router_input.shape[1]}, "
                f"expected {self.tadr_router_in_channels}"
            )
        return router_input

    def _apply_ndr(
        self,
        coarse_logits_37,
        image_68,
        base_logits,
        graph_logits,
        scale,
        alpha_eff,
        gamma_eff,
        uncertainty_gate,
        return_aux,
        return_probe_aux=False,
        semantic_feat=None,
    ):
        if image_68 is None:
            raise ValueError("NDR branch requires image_68")
        if image_68.ndim != 4 or image_68.shape[1] != 3:
            raise ValueError(f"NDR image_68 must be [B,3,H,W], got {list(image_68.shape)}")
        target_size = (self.ndr_loss_size, self.ndr_loss_size)
        if tuple(image_68.shape[-2:]) != target_size:
            raise ValueError(
                f"NDR image_68 spatial size must be {target_size}, got {tuple(image_68.shape[-2:])}"
            )
        coarse_logits_68 = F.interpolate(
            coarse_logits_37,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        coarse_prob_68 = torch.sigmoid(coarse_logits_68.detach())
        ndr_result = self.ndr_branch(
            image_68.to(dtype=coarse_logits_68.dtype),
            coarse_prob_68,
            return_probe_aux=bool(return_probe_aux),
        )
        if bool(return_probe_aux):
            residual_logits_68, sobel_68, probe_ndr_detail_feat = ndr_result
        else:
            residual_logits_68, sobel_68 = ndr_result
            probe_ndr_detail_feat = None

        uncertainty_68 = 1.0 - 2.0 * torch.abs(coarse_prob_68 - 0.5)
        uncertainty_68 = torch.clamp(uncertainty_68, 0.0, 1.0)
        base_uncertainty = uncertainty_68 if self.ndr_use_uncertainty_gate else torch.ones_like(coarse_prob_68)
        if not self.ndr_use_edge_gate:
            edge_gate = torch.ones_like(sobel_68)
        elif self.ndr_gate_mode == "uncertainty_edge_boost":
            edge_gate = 0.5 + 0.5 * sobel_68
        elif self.ndr_gate_mode == "uncertainty_edge_stronger":
            edge_gate = 0.25 + 0.75 * sobel_68
        else:
            raise ValueError(f"Unsupported NDR_GATE_MODE: {self.ndr_gate_mode}")
        base_gate = torch.clamp(base_uncertainty * edge_gate, 0.0, 1.0)
        detail_gate = base_gate
        coarse_boundary_68 = None
        router_input = None
        router_map_68 = None
        if self.use_tadr_router:
            coarse_boundary_68 = self.ndr_branch.compute_sobel_map(coarse_prob_68)
            router_input = self._build_tadr_router_input(
                coarse_prob_68,
                uncertainty_68,
                sobel_68,
                coarse_boundary_68,
            )
            router_map_68 = self.tadr_router(router_input)
            router_map_68 = torch.clamp(router_map_68, self.tadr_router_min, self.tadr_router_max)
            detail_gate = torch.clamp(base_gate * router_map_68, 0.0, 1.0)
        beta_eff = self._ndr_beta_eff()
        detail_gate_v1 = detail_gate
        detail_gate, ndr_v2_aux = self._apply_ndr_v2_shape_gate(
            detail_gate_v1,
            coarse_prob_68,
            sobel_68,
            uncertainty_68,
        )
        ndr_delta_logits_68 = coarse_logits_68.new_tensor(float(beta_eff)) * detail_gate * residual_logits_68
        logits = coarse_logits_68 + ndr_delta_logits_68

        if not return_aux and not return_probe_aux:
            return logits

        output = self._aux_output(
            logits,
            base_logits,
            graph_logits,
            scale,
            alpha_eff,
            gamma_eff,
            uncertainty_gate,
        )
        gate_mean, gate_min, gate_max = self._tensor_stats(detail_gate)
        residual_abs = residual_logits_68.detach().abs()
        output.update(
            {
                "coarse_logits_37": coarse_logits_37,
                "coarse_logits_68": coarse_logits_68,
                "coarse_prob_68": coarse_prob_68,
                "uncertainty_68": uncertainty_68,
                "residual_logits_68": residual_logits_68,
                "ndr_delta_logits_68": ndr_delta_logits_68,
                "base_gate": base_gate,
                "detail_gate": detail_gate,
                "sobel_68": sobel_68,
                "ndr_beta_eff": coarse_logits_68.new_tensor(float(beta_eff)),
                "ndr_detail_gate_mean": gate_mean,
                "ndr_detail_gate_min": gate_min,
                "ndr_detail_gate_max": gate_max,
                "ndr_residual_abs_mean": residual_abs.mean(),
                "ndr_residual_abs_max": residual_abs.max(),
            }
        )
        if return_probe_aux:
            output["probe_ndr_detail_feat"] = probe_ndr_detail_feat
        if self.use_ndr_v2:
            output.update(ndr_v2_aux)
            gate_v1_mean, gate_v1_min, gate_v1_max = self._tensor_stats(ndr_v2_aux["detail_gate_v1"])
            gate_v2_mean, gate_v2_min, gate_v2_max = self._tensor_stats(ndr_v2_aux["detail_gate_v2"])
            boundary_mean, boundary_min, boundary_max = self._tensor_stats(ndr_v2_aux["boundary_band_68"])
            edge_norm_mean, edge_norm_min, edge_norm_max = self._tensor_stats(ndr_v2_aux["edge_norm_68"])
            shape_boost_mean, shape_boost_min, shape_boost_max = self._tensor_stats(ndr_v2_aux["shape_boost_68"])
            output.update(
                {
                    "boundary_band": ndr_v2_aux["boundary_band_68"],
                    "edge_norm": ndr_v2_aux["edge_norm_68"],
                    "shape_boost": ndr_v2_aux["shape_boost_68"],
                    "ndr_v2_detail_gate_v1_mean": gate_v1_mean,
                    "ndr_v2_detail_gate_v1_min": gate_v1_min,
                    "ndr_v2_detail_gate_v1_max": gate_v1_max,
                    "ndr_v2_detail_gate_v2_mean": gate_v2_mean,
                    "ndr_v2_detail_gate_v2_min": gate_v2_min,
                    "ndr_v2_detail_gate_v2_max": gate_v2_max,
                    "ndr_v2_boundary_mean": boundary_mean,
                    "ndr_v2_boundary_min": boundary_min,
                    "ndr_v2_boundary_max": boundary_max,
                    "ndr_v2_edge_norm_mean": edge_norm_mean,
                    "ndr_v2_edge_norm_min": edge_norm_min,
                    "ndr_v2_edge_norm_max": edge_norm_max,
                    "ndr_v2_uncertainty_mean": uncertainty_68.detach().mean(),
                    "ndr_v2_uncertainty_min": uncertainty_68.detach().min(),
                    "ndr_v2_uncertainty_max": uncertainty_68.detach().max(),
                    "ndr_v2_shape_boost_mean": shape_boost_mean,
                    "ndr_v2_shape_boost_min": shape_boost_min,
                    "ndr_v2_shape_boost_max": shape_boost_max,
                }
            )
        if self.use_tadr_router:
            router_mean, router_min, router_max = self._tensor_stats(router_map_68)
            base_gate_mean, base_gate_min, base_gate_max = self._tensor_stats(base_gate)
            output.update(
                {
                    "coarse_boundary_68": coarse_boundary_68,
                    "router_input": router_input,
                    "router_map_68": router_map_68,
                    "tadr_router_mean": router_mean,
                    "tadr_router_min": router_min,
                    "tadr_router_max": router_max,
                    "tadr_base_gate_mean": base_gate_mean,
                    "tadr_base_gate_min": base_gate_min,
                    "tadr_base_gate_max": base_gate_max,
                    "tadr_final_gate_mean": gate_mean,
                    "tadr_final_gate_min": gate_min,
                    "tadr_final_gate_max": gate_max,
                }
            )
        return self._attach_proto_aux(output, semantic_feat)

    def forward(
        self,
        feat,
        image_68=None,
        return_aux=False,
        return_probe_aux=False,
        pa_return_aux=None,
        pa_compare_original=False,
    ):
        bsz, _, height, width = feat.shape
        base_logits = self.base_head(feat)
        scale = self._ramp_scale()
        alpha_eff = self.alpha_max * scale
        gamma_eff = self.gamma_max * scale
        pa_return_aux = bool(return_aux) if pa_return_aux is None else bool(pa_return_aux)
        current_epoch = int(self.current_epoch_tensor.item())
        pa_edge_scale = self.pa_dagp.edge_scale(current_epoch) if self.pa_dagp is not None else 0.0
        return_ber_graph_aux = self._ber_graph_aux_enabled(return_aux)

        if alpha_eff == 0.0 or gamma_eff == 0.0:
            graph_logits = torch.zeros_like(base_logits)
            semantic_feat = self.proj(feat) if return_aux and self.use_proto_contrast else None
            pa_state = (
                self.pa_dagp(feat, base_logits, topk_idx=None, edge_scale=pa_edge_scale)
                if self.pa_dagp is not None and pa_return_aux
                else None
            )
            if self.use_ndr_branch:
                output = self._apply_ndr(
                    base_logits,
                    image_68,
                    base_logits,
                    graph_logits,
                    scale,
                    alpha_eff,
                    gamma_eff,
                    self._uncertainty_output_gate(base_logits),
                    return_aux,
                    return_probe_aux=return_probe_aux,
                    semantic_feat=semantic_feat,
                )
                if return_aux and isinstance(output, dict):
                    output = self._attach_pa_aux(output, pa_state)
                    if pa_compare_original:
                        output["pa_original_coarse_logits_native"] = base_logits.detach()
                if return_aux and isinstance(output, dict) and return_ber_graph_aux:
                    with torch.no_grad():
                        semantic_weight, ber_topk_idx = self._topk_affinity(feat)
                    output = self._attach_ber_graph_aux(output, ber_topk_idx, semantic_weight)
                return output
            if return_aux:
                uncertainty_gate = self._uncertainty_output_gate(base_logits)
                output = self._aux_output(
                    base_logits,
                    base_logits,
                    graph_logits,
                    scale,
                    alpha_eff,
                    gamma_eff,
                    uncertainty_gate,
                    semantic_feat=semantic_feat,
                    pa_state=pa_state,
                    pa_original_logits=base_logits if pa_compare_original else None,
                )
                if return_ber_graph_aux:
                    with torch.no_grad():
                        semantic_weight, ber_topk_idx = self._topk_affinity(feat)
                    output = self._attach_ber_graph_aux(output, ber_topk_idx, semantic_weight)
                return output
            return base_logits

        if self.affinity_detach:
            with torch.no_grad():
                attn, topk_idx = self._topk_affinity(feat)
        else:
            attn, topk_idx = self._topk_affinity(feat)
        semantic_topk_weight = attn.detach() if return_ber_graph_aux else None
        if self.use_prob_gate:
            attn = self._apply_prob_gate(attn, topk_idx, base_logits)
        attn_original = attn
        pa_state = None
        if self.pa_dagp is not None:
            pa_state = self.pa_dagp(
                feat,
                base_logits,
                topk_idx=topk_idx if pa_edge_scale > 0.0 else None,
                edge_scale=pa_edge_scale,
            )
            if pa_edge_scale > 0.0:
                polarity_gate = pa_state["polarity_gate"]
                if polarity_gate is None or polarity_gate.shape != attn.shape:
                    raise RuntimeError(
                        "PA-DAGP polarity gate shape mismatch: "
                        f"gate={None if polarity_gate is None else list(polarity_gate.shape)}, "
                        f"edge={list(attn.shape)}"
                    )
                attn = attn * polarity_gate.to(dtype=attn.dtype)
                attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(self.prob_gate_eps)
                if not bool(torch.isfinite(attn).all().item()):
                    raise RuntimeError("PA-DAGP normalized edge contains NaN/Inf.")
                edge_sum_error = (attn.float().sum(dim=-1) - 1.0).abs().max()
                if float(edge_sum_error.detach().item()) > 1e-5:
                    raise RuntimeError(
                        f"PA-DAGP normalized edge sum error is too large: {float(edge_sum_error.item()):.8g}"
                    )
                pa_state["diagnostics"]["edge_normalization_max_error"] = edge_sum_error.detach()
            else:
                pa_state["diagnostics"]["edge_normalization_max_error"] = base_logits.new_tensor(0.0)

        z_map = self.proj(feat)
        z = z_map.flatten(2).transpose(1, 2)
        value = self.value(z)
        neigh_value = self._gather_neighbors(value, topk_idx)
        agg = (attn.to(dtype=value.dtype).unsqueeze(-1) * neigh_value).sum(dim=2)
        z_prop = z + value.new_tensor(float(gamma_eff)) * agg
        z_prop_map = z_prop.transpose(1, 2).reshape(bsz, self.hidden, height, width)
        graph_logits = self.graph_pred(z_prop_map)
        uncertainty_gate = self._uncertainty_output_gate(base_logits)
        graph_residual = graph_logits if uncertainty_gate is None else uncertainty_gate * graph_logits
        logits = base_logits + graph_logits.new_tensor(float(alpha_eff)) * graph_residual
        pa_original_logits = None
        if pa_compare_original:
            if self.pa_dagp is not None and pa_edge_scale > 0.0:
                with torch.no_grad():
                    original_agg = (
                        attn_original.to(dtype=value.dtype).unsqueeze(-1) * neigh_value.detach()
                    ).sum(dim=2)
                    original_z_prop = z.detach() + value.new_tensor(float(gamma_eff)) * original_agg
                    original_z_prop_map = original_z_prop.transpose(1, 2).reshape(
                        bsz, self.hidden, height, width
                    )
                    original_graph_logits = self.graph_pred(original_z_prop_map)
                    original_uncertainty = self._uncertainty_output_gate(base_logits.detach())
                    original_residual = (
                        original_graph_logits
                        if original_uncertainty is None
                        else original_uncertainty * original_graph_logits
                    )
                    pa_original_logits = (
                        base_logits.detach()
                        + original_graph_logits.new_tensor(float(alpha_eff)) * original_residual
                    )
            else:
                pa_original_logits = logits.detach()
        if self.use_ndr_branch:
            output = self._apply_ndr(
                logits,
                image_68,
                base_logits,
                graph_logits,
                scale,
                alpha_eff,
                gamma_eff,
                uncertainty_gate,
                return_aux,
                return_probe_aux=return_probe_aux,
                semantic_feat=z_prop_map if return_aux and self.use_proto_contrast else None,
            )
            if return_aux and isinstance(output, dict):
                output = self._attach_pa_aux(output, pa_state if pa_return_aux else None)
                if pa_original_logits is not None:
                    output["pa_original_coarse_logits_native"] = pa_original_logits.detach()
                if return_ber_graph_aux:
                    output = self._attach_ber_graph_aux(
                        output,
                        topk_idx,
                        semantic_topk_weight,
                    )
            return output
        if return_aux:
            output = self._aux_output(
                logits,
                base_logits,
                graph_logits,
                scale,
                alpha_eff,
                gamma_eff,
                uncertainty_gate,
                semantic_feat=z_prop_map if self.use_proto_contrast else None,
                pa_state=pa_state if pa_return_aux else None,
                pa_original_logits=pa_original_logits,
            )
            if return_ber_graph_aux:
                output = self._attach_ber_graph_aux(
                    output,
                    topk_idx,
                    semantic_topk_weight,
                )
            return output
        return logits


def _group_count(channels):
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvGNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class SceLite(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Conv2d(channels, channels, kernel_size=1)
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)

    def forward(self, x):
        key = torch.sigmoid(self.gate(x))
        uncertainty = key * (1.0 - key)
        return x + self.dw(x) * uncertainty


class FuseBlock(nn.Module):
    def __init__(self, hidden, use_sce=True):
        super().__init__()
        self.reduce = ConvGNAct(hidden * 2, hidden, kernel_size=1)
        self.dw = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.out = nn.Conv2d(hidden, hidden, kernel_size=1)
        self.sce = SceLite(hidden) if use_sce else nn.Identity()

    def forward(self, low, high):
        if high.shape[-2:] != low.shape[-2:]:
            high = F.interpolate(high, size=low.shape[-2:], mode="bilinear", align_corners=False)
        residual = low
        x = torch.cat([low, high], dim=1)
        x = self.reduce(x)
        x = self.act(self.dw(x))
        x = self.out(x)
        x = self.sce(x)
        return residual + x


class PCFLite(nn.Module):
    def __init__(self, hidden, dilations):
        super().__init__()
        self.dilations = [int(value) for value in dilations]
        if hidden % len(self.dilations) != 0:
            raise ValueError(f"MLC_HIDDEN must be divisible by len(MLC_PCF_DILATIONS), got {hidden}")
        group_channels = hidden // len(self.dilations)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    group_channels,
                    group_channels,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    groups=group_channels,
                )
                for dilation in self.dilations
            ]
        )
        self.act = nn.GELU()
        self.out = ConvGNAct(hidden, hidden, kernel_size=1)

    def forward(self, x):
        chunks = torch.chunk(x, len(self.dilations), dim=1)
        outputs = []
        running = None
        for chunk, conv in zip(chunks, self.convs):
            current = chunk if running is None else chunk + running
            running = self.act(conv(current))
            outputs.append(running)
        return x + self.out(torch.cat(outputs, dim=1))


class MultiLevelContextFusionDecoder(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        hidden = int(getattr(cfg, "MLC_HIDDEN", 128))
        self.loss_size = int(getattr(cfg, "LOSS_SIZE", 68))
        self.use_semantic_gate = bool(getattr(cfg, "MLC_USE_SEMANTIC_GATE", True))
        use_sce = bool(getattr(cfg, "MLC_USE_SCE_LITE", True))
        use_pcf = bool(getattr(cfg, "MLC_USE_PCF", True))
        dilations = list(getattr(cfg, "MLC_PCF_DILATIONS", [1, 2, 3, 5]))

        self.base_head = SimpleConvSegHead(in_channels)
        self.proj4 = ConvGNAct(in_channels, hidden, kernel_size=1)
        self.proj8 = ConvGNAct(in_channels, hidden, kernel_size=1)
        self.proj12 = ConvGNAct(in_channels, hidden, kernel_size=1)
        self.gate_conv = nn.Conv2d(hidden, hidden, kernel_size=1) if self.use_semantic_gate else None
        self.fuse8 = FuseBlock(hidden, use_sce=use_sce)
        self.fuse4 = FuseBlock(hidden, use_sce=use_sce)
        self.pcf = PCFLite(hidden, dilations) if use_pcf else nn.Identity()
        self.context_pred = nn.Conv2d(hidden, 1, kernel_size=1)

        init = float(getattr(cfg, "MLC_RES_SCALE_INIT", 0.1))
        init = max(1e-6, min(1.0 - 1e-6, init))
        self.res_alpha = nn.Parameter(torch.tensor(math.log(init / (1.0 - init)), dtype=torch.float32))

    def _unpack(self, x):
        if isinstance(x, dict):
            return x["l4"], x["l8"], x["l12"]
        if isinstance(x, (tuple, list)) and len(x) == 3:
            return x[0], x[1], x[2]
        raise TypeError("MultiLevelContextFusionDecoder expects dict {'l4','l8','l12'} or a 3-tuple/list.")

    def forward(self, x):
        f4, f8, f12 = self._unpack(x)
        p4 = self.proj4(f4)
        p8 = self.proj8(f8)
        p12 = self.proj12(f12)

        if self.gate_conv is not None:
            gate = torch.sigmoid(self.gate_conv(p12))
            p4 = p4 * (1.0 + gate)
            p8 = p8 * (1.0 + gate)

        x8 = self.fuse8(p8, p12)
        x4 = self.fuse4(p4, x8)
        context = self.pcf(x4)
        context = F.interpolate(context, size=(self.loss_size, self.loss_size), mode="bilinear", align_corners=False)
        context_logits = self.context_pred(context)

        base_logits = self.base_head(f12)
        base_logits = F.interpolate(base_logits, size=(self.loss_size, self.loss_size), mode="bilinear", align_corners=False)
        scale = torch.sigmoid(self.res_alpha)
        final_logits = base_logits + scale * context_logits
        return {
            "logits": final_logits,
            "base_logits": base_logits,
            "context_logits": context_logits,
            "context_scale": scale,
        }


class SAPConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        dilation=1,
        bias=False,
        activation="relu",
    ):
        super().__init__()
        if activation == "relu":
            act = nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            act = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            act,
        )

    def forward(self, x):
        return self.block(x)


class SAPCNNTrans(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_f1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn_f1 = nn.BatchNorm2d(out_channels)
        self.conv_f2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn_f2 = nn.BatchNorm2d(out_channels)

    def forward(self, a, b):
        a = F.relu(self.bn_f1(self.conv_f1(a)), inplace=True)
        b = F.relu(self.bn_f2(self.conv_f2(b)), inplace=True)
        return a, b


class SAPCAFF(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_fuse = nn.Sequential(
            SAPConvBNAct(channels * 2, channels, kernel_size=3, padding=1, bias=True, activation="leaky_relu"),
            SAPConvBNAct(channels, channels, kernel_size=3, padding=1, bias=True, activation="leaky_relu"),
            SAPConvBNAct(channels, channels, kernel_size=3, padding=1, bias=True, activation="leaky_relu"),
        )
        self.conv_p1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.conv_matt = SAPConvBNAct(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=True,
            activation="leaky_relu",
        )
        self.conv_p2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.bn_p2 = nn.BatchNorm2d(channels)

    def forward(self, x, global_feature):
        x = self.conv_fuse(torch.cat([x, global_feature], dim=1))
        attention = torch.sigmoid(self.conv_p1(x))
        uncertainty = attention * (1.0 - attention)
        modulation = self.conv_matt(uncertainty)
        x = x * (1.0 + modulation)
        x = self.conv_p2(x)
        return F.relu(self.bn_p2(x), inplace=True)


class SAPConvDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            SAPConvBNAct(in_channels, in_channels, kernel_size=1, bias=False),
            SAPConvBNAct(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            SAPConvBNAct(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            SAPConvBNAct(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return self.block(x)


class SAPReceptiveConv(nn.Module):
    def __init__(self, channels, width, dilations=(1, 2, 2, 4), aggregation=True):
        super().__init__()
        self.width = int(width)
        self.dilations = [int(value) for value in dilations]
        self.scale = len(self.dilations)
        self.aggregation = bool(aggregation)
        if self.width <= 0:
            raise ValueError("SAP_RCIM_WIDTH must be positive.")
        self.conv_in = nn.Conv2d(channels, self.width * self.scale, kernel_size=1, bias=False)
        self.bn_in = nn.BatchNorm2d(self.width * self.scale)
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    self.width,
                    self.width,
                    kernel_size=3,
                    padding=dilation,
                    dilation=dilation,
                    bias=False,
                )
                for dilation in self.dilations
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm2d(self.width) for _ in self.dilations])
        self.conv_out = nn.Conv2d(self.width * self.scale, channels, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn_in(self.conv_in(x)))
        chunks = torch.split(x, self.width, dim=1)
        outputs = []
        previous = None
        for chunk, conv, bn in zip(chunks, self.convs, self.bns):
            current = chunk if previous is None or not self.aggregation else chunk + previous
            previous = self.relu(bn(conv(current)))
            outputs.append(previous)
        x = torch.cat(outputs, dim=1)
        x = self.bn_out(self.conv_out(x))
        return self.relu(x + residual)


class SAPFusion(nn.Module):
    def __init__(self, channels, width, use_gap_guide=True):
        super().__init__()
        self.use_gap_guide = bool(use_gap_guide)
        inter_channels = max(1, channels // 4)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.global_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, bias=True),
        )
        self.fu1 = SAPReceptiveConv(channels, width=width, dilations=(1, 2, 2, 4))
        self.fu2 = SAPReceptiveConv(channels * 2, width=width * 2, dilations=(1, 2, 2, 4))

    def forward(self, x, residual):
        if self.use_gap_guide:
            gate = torch.sigmoid(self.global_att(self.gap(x + residual)))
            x = x * gate
            residual = residual * gate
        x = self.fu1(x)
        x = torch.cat([x, residual], dim=1)
        return self.fu2(x)


class SAPOutPut(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pred = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(self, x, out_size):
        x = self.pred(x)
        return F.interpolate(x, size=(out_size, out_size), mode="bilinear", align_corners=False)


def _debug_abs_mean(x):
    return x.detach().abs().mean()


class SAPRCIMAdaptDecoder(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        self.layers = [int(layer) for layer in getattr(cfg, "MULTI_LEVEL_LAYERS", [10, 11, 12])]
        if len(self.layers) != 3:
            raise ValueError(f"SAP-RCIM expects exactly 3 layers, got {self.layers}")
        self.mode = str(getattr(cfg, "SAP_RCIM_MODE", "full"))
        if self.mode not in {"base_only", "caff_min", "full"}:
            raise ValueError(f"Unknown SAP_RCIM_MODE: {self.mode}")
        channels = int(getattr(cfg, "SAP_RCIM_CHANNEL", 64))
        width = int(getattr(cfg, "SAP_RCIM_WIDTH", 32))
        self.out_size = int(getattr(cfg, "SAP_RCIM_OUT_SIZE", getattr(cfg, "LOSS_SIZE", 68)))
        self.base_head = SimpleConvSegHead(in_channels)

        self.trans_10_11 = SAPCNNTrans(in_channels, channels)
        self.trans_11_12 = SAPCNNTrans(in_channels, channels)
        self.caff_10_11 = SAPCAFF(channels)
        self.caff_11_12 = SAPCAFF(channels)
        self.pair_decoder_10_11 = SAPConvDecoder(channels, channels)
        self.pair_decoder_11_12 = SAPConvDecoder(channels, channels)
        self.caff_min_decoder = SAPConvDecoder(channels * 2, channels)
        self.fusion = SAPFusion(
            channels,
            width=width,
            use_gap_guide=bool(getattr(cfg, "SAP_RCIM_USE_GAP_GUIDE", True)),
        )
        self.post_fusion_decoder = SAPConvDecoder(channels * 2, channels)
        self.sap_output = SAPOutPut(channels)

    def _unpack(self, x):
        if isinstance(x, dict):
            values = []
            for layer in self.layers:
                key = f"l{layer}"
                fallback_key = f"feature_l{layer}"
                if key in x:
                    values.append(x[key])
                elif fallback_key in x:
                    values.append(x[fallback_key])
                else:
                    raise KeyError(f"SAP-RCIM input missing {key}/{fallback_key}")
            return values
        if isinstance(x, (tuple, list)) and len(x) == 3:
            return list(x)
        raise TypeError("SAPRCIMAdaptDecoder expects a 3-level feature dict or a 3-tuple/list.")

    def _base_output(self, f10, f11, f12):
        base_logits = self.base_head(f12)
        base_logits = F.interpolate(base_logits, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        zero = base_logits.detach().sum() * 0.0
        return {
            "logits": base_logits,
            "base_logits": base_logits,
            "sap_logits": base_logits,
            "debug": {
                "x10_abs_mean": _debug_abs_mean(f10),
                "x11_abs_mean": _debug_abs_mean(f11),
                "x12_abs_mean": _debug_abs_mean(f12),
                "caff_10_11_abs_mean": zero,
                "caff_11_12_abs_mean": zero,
                "fusion_abs_mean": zero,
                "final_logits_abs_mean": _debug_abs_mean(base_logits),
            },
        }

    def forward(self, x):
        f10, f11, f12 = self._unpack(x)
        if self.mode == "base_only":
            return self._base_output(f10, f11, f12)

        base_logits = self.base_head(f12)
        base_logits = F.interpolate(base_logits, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)

        x10, x11_for_10 = self.trans_10_11(f10, f11)
        x11_for_12, x12 = self.trans_11_12(f11, f12)
        caff_10_11 = self.caff_10_11(x10, x11_for_10)
        caff_11_12 = self.caff_11_12(x11_for_12, x12)

        if self.mode == "caff_min":
            fused = self.caff_min_decoder(torch.cat([caff_10_11, caff_11_12], dim=1))
        else:
            x10_11 = self.pair_decoder_10_11(caff_10_11)
            x11_12 = self.pair_decoder_11_12(caff_11_12)
            fused = self.fusion(x10_11, x11_12)
            fused = self.post_fusion_decoder(fused)

        sap_logits = self.sap_output(fused, self.out_size)
        return {
            "logits": sap_logits,
            "base_logits": base_logits,
            "sap_logits": sap_logits,
            "debug": {
                "x10_abs_mean": _debug_abs_mean(x10),
                "x11_abs_mean": _debug_abs_mean(x11_for_10),
                "x12_abs_mean": _debug_abs_mean(x12),
                "caff_10_11_abs_mean": _debug_abs_mean(caff_10_11),
                "caff_11_12_abs_mean": _debug_abs_mean(caff_11_12),
                "fusion_abs_mean": _debug_abs_mean(fused),
                "final_logits_abs_mean": _debug_abs_mean(sap_logits),
            },
        }


class ContextResidualHead(nn.Module):
    def __init__(self, in_channels, hidden=64, gamma_init=0.0):
        super().__init__()
        self.base = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.reduce = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.dw = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=3,
            padding=1,
            groups=hidden,
            bias=True,
        )
        self.act = nn.GELU()
        self.out = nn.Conv2d(hidden, 1, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x):
        base_logits = self.base(x)
        z = self.reduce(x)
        z = self.dw(z)
        z = self.act(z)
        res_logits = self.out(z)
        return base_logits + self.gamma * res_logits


class GatedContextSegHead(nn.Module):
    def __init__(self, in_channels, hidden=64, gamma_init=0.0):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.reduce = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.dwconv = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=3,
            padding=1,
            groups=hidden,
            bias=True,
        )
        self.act = nn.GELU()
        self.out = nn.Conv2d(hidden, 1, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x):
        base_logits = self.proj(x)
        z = self.reduce(x)
        z = self.dwconv(z)
        z = self.act(z)
        ctx_logits = self.out(z)
        return base_logits + self.gamma * ctx_logits


def _build_pa_dagp(in_channels, cfg):
    if not bool(getattr(cfg, "USE_PA_DAGP", False)):
        return None
    # PA is an optional ablation branch. Preserve the caller's RNG state so
    # enabling it cannot change initialization of the existing DAGP/CSD path.
    with torch.random.fork_rng(devices=[]):
        return PolarityEdgeGateV1(
            in_channels=in_channels,
            pol_dim=int(getattr(cfg, "PA_DAGP_POL_DIM", 32)),
            gn_groups=int(getattr(cfg, "PA_DAGP_POL_GN_GROUPS", 4)),
            act=str(getattr(cfg, "PA_DAGP_POL_ACT", "gelu")),
            anchor_weight_power=float(getattr(cfg, "PA_DAGP_ANCHOR_WEIGHT_POWER", 2.0)),
            anchor_eps=float(getattr(cfg, "PA_DAGP_ANCHOR_EPS", 1e-6)),
            detach_base_prob=bool(getattr(cfg, "PA_DAGP_DETACH_BASE_PROB", True)),
            detach_anchors=bool(getattr(cfg, "PA_DAGP_DETACH_ANCHORS", True)),
            detach_rho=bool(getattr(cfg, "PA_DAGP_DETACH_RHO", True)),
            min_anchor_norm=float(getattr(cfg, "PA_DAGP_MIN_ANCHOR_NORM", 1e-6)),
            use_calib_head=bool(getattr(cfg, "PA_DAGP_USE_CALIB_HEAD", True)),
            calib_hidden=int(getattr(cfg, "PA_DAGP_CALIB_HIDDEN", 32)),
            calib_zero_init=bool(getattr(cfg, "PA_DAGP_CALIB_ZERO_INIT", True)),
            polarity_tau=float(getattr(cfg, "PA_DAGP_POLARITY_TAU", 0.50)),
            edge_cut_max=float(getattr(cfg, "PA_DAGP_EDGE_CUT_MAX", 0.50)),
            edge_gate_min=float(getattr(cfg, "PA_DAGP_EDGE_GATE_MIN", 0.50)),
            use_same_polarity_boost=bool(getattr(cfg, "PA_DAGP_USE_SAME_POLARITY_BOOST", False)),
            renormalize_edge=bool(getattr(cfg, "PA_DAGP_RENORMALIZE_EDGE", True)),
            start_epoch=int(getattr(cfg, "PA_DAGP_START_EPOCH", 7)),
            ramp_end_epoch=int(getattr(cfg, "PA_DAGP_RAMP_END_EPOCH", 15)),
            edge_stop_epoch=int(getattr(cfg, "PA_DAGP_EDGE_STOP_EPOCH", 36)),
            diag_ambig_thresh=float(getattr(cfg, "PA_DAGP_DIAG_AMBIG_THRESH", 0.20)),
        )


def build_seg_head(in_channels, cfg):
    head_type = str(getattr(cfg, "HEAD_TYPE", "simple")).lower()
    if head_type == "cacd_v1_base":
        return CACDV1BaseHead(in_channels=in_channels, cfg=cfg)
    if head_type == "simple":
        return SimpleConvSegHead(in_channels)
    if head_type == "dagp":
        num_layers = int(getattr(cfg, "DAGP_NUM_LAYERS", 1))
        if num_layers != 1:
            raise ValueError(f"DAGP-Minimal supports DAGP_NUM_LAYERS=1 only, got {num_layers}")
        return DINOAffinityGraphPropagationHead(
            in_channels=in_channels,
            hidden=int(getattr(cfg, "DAGP_HIDDEN", 64)),
            topk=int(getattr(cfg, "DAGP_TOPK", 24)),
            tau=float(getattr(cfg, "DAGP_TAU", 0.10)),
            alpha_init=float(getattr(cfg, "DAGP_ALPHA_INIT", 0.05)),
            gamma_init=float(getattr(cfg, "DAGP_GAMMA_INIT", 0.10)),
            affinity_detach=bool(getattr(cfg, "DAGP_AFFINITY_DETACH", True)),
            use_ffn=bool(getattr(cfg, "DAGP_USE_FFN", False)),
            use_dwconv=bool(getattr(cfg, "DAGP_USE_DWCONV", False)),
        )
    if head_type == "dagp_safe":
        return DAGPSafeHead(
            in_channels=in_channels,
            hidden=int(getattr(cfg, "DAGP_SAFE_HIDDEN", 64)),
            topk=int(getattr(cfg, "DAGP_SAFE_TOPK", 12)),
            tau=float(getattr(cfg, "DAGP_SAFE_TAU", 0.07)),
            alpha_max=float(getattr(cfg, "DAGP_SAFE_ALPHA_MAX", 0.05)),
            gamma_max=float(getattr(cfg, "DAGP_SAFE_GAMMA_MAX", 0.03)),
            warmup_epoch=int(getattr(cfg, "DAGP_SAFE_WARMUP_EPOCH", 6)),
            ramp_start_epoch=int(getattr(cfg, "DAGP_SAFE_RAMP_START_EPOCH", 7)),
            ramp_end_epoch=int(getattr(cfg, "DAGP_SAFE_RAMP_END_EPOCH", 15)),
            affinity_detach=bool(getattr(cfg, "DAGP_SAFE_AFFINITY_DETACH", True)),
            exclude_self=bool(getattr(cfg, "DAGP_SAFE_EXCLUDE_SELF", True)),
            use_prob_gate=bool(getattr(cfg, "DAGP_SAFE_USE_PROB_GATE", True)),
            prob_gate_sigma=float(getattr(cfg, "DAGP_SAFE_PROB_GATE_SIGMA", 0.25)),
            prob_gate_eps=float(getattr(cfg, "DAGP_SAFE_PROB_GATE_EPS", 1e-6)),
            zero_init_graph_pred=bool(getattr(cfg, "DAGP_SAFE_ZERO_INIT_GRAPH_PRED", True)),
            zero_init_value=bool(getattr(cfg, "DAGP_SAFE_ZERO_INIT_VALUE", False)),
            use_uncertainty_output_gate=bool(
                getattr(cfg, "DAGP_SAFE_USE_UNCERTAINTY_OUTPUT_GATE", False)
            ),
            uncertainty_power=float(getattr(cfg, "DAGP_SAFE_UNCERTAINTY_POWER", 1.0)),
            uncertainty_min=float(getattr(cfg, "DAGP_SAFE_UNCERTAINTY_MIN", 0.0)),
            uncertainty_max=float(getattr(cfg, "DAGP_SAFE_UNCERTAINTY_MAX", 1.0)),
            uncertainty_detach=bool(getattr(cfg, "DAGP_SAFE_UNCERTAINTY_DETACH", True)),
            use_ndr_branch=bool(getattr(cfg, "USE_NDR_BRANCH", False)),
            ndr_loss_size=int(getattr(cfg, "LOSS_SIZE", 68)),
            ndr_input_rgb=bool(getattr(cfg, "NDR_INPUT_RGB", True)),
            ndr_input_sobel=bool(getattr(cfg, "NDR_INPUT_SOBEL", True)),
            ndr_input_coarse_prob=bool(getattr(cfg, "NDR_INPUT_COARSE_PROB", True)),
            ndr_in_channels=int(getattr(cfg, "NDR_IN_CHANNELS", 5)),
            ndr_hidden=int(getattr(cfg, "NDR_HIDDEN", 32)),
            ndr_num_layers=int(getattr(cfg, "NDR_NUM_LAYERS", 3)),
            ndr_use_gn=bool(getattr(cfg, "NDR_USE_GN", True)),
            ndr_gn_groups=int(getattr(cfg, "NDR_GN_GROUPS", 4)),
            ndr_act=str(getattr(cfg, "NDR_ACT", "gelu")),
            ndr_zero_init_out=bool(getattr(cfg, "NDR_ZERO_INIT_OUT", True)),
            ndr_residual_clip=float(getattr(cfg, "NDR_RESIDUAL_CLIP", 2.0)),
            ndr_beta_max=float(getattr(cfg, "NDR_BETA_MAX", 0.10)),
            ndr_warmup_epoch=int(getattr(cfg, "NDR_WARMUP_EPOCH", 6)),
            ndr_ramp_start_epoch=int(getattr(cfg, "NDR_RAMP_START_EPOCH", 7)),
            ndr_ramp_end_epoch=int(getattr(cfg, "NDR_RAMP_END_EPOCH", 15)),
            ndr_use_uncertainty_gate=bool(getattr(cfg, "NDR_USE_UNCERTAINTY_GATE", True)),
            ndr_use_edge_gate=bool(getattr(cfg, "NDR_USE_EDGE_GATE", True)),
            ndr_gate_mode=str(getattr(cfg, "NDR_GATE_MODE", "uncertainty_edge_boost")),
            use_ndr_v2=bool(getattr(cfg, "USE_NDR_V2", False)),
            ndr_version=str(getattr(cfg, "NDR_VERSION", "v1")),
            ndr_v2_use_shape_gate=bool(getattr(cfg, "NDR_V2_USE_SHAPE_GATE", False)),
            ndr_v2_shape_gate_mode=str(getattr(cfg, "NDR_V2_SHAPE_GATE_MODE", "soft_boundary_edge_boost")),
            ndr_v2_boundary_source=str(getattr(cfg, "NDR_V2_BOUNDARY_SOURCE", "coarse_prob")),
            ndr_v2_boundary_radius=int(getattr(cfg, "NDR_V2_BOUNDARY_RADIUS", 2)),
            ndr_v2_boundary_detach=bool(getattr(cfg, "NDR_V2_BOUNDARY_DETACH", True)),
            ndr_v2_shape_alpha_max=float(getattr(cfg, "NDR_V2_SHAPE_ALPHA_MAX", 0.20)),
            ndr_v2_shape_edge_mix=float(getattr(cfg, "NDR_V2_SHAPE_EDGE_MIX", 0.50)),
            ndr_v2_shape_uncert_mix=float(getattr(cfg, "NDR_V2_SHAPE_UNCERT_MIX", 0.50)),
            ndr_v2_gate_combine=str(getattr(cfg, "NDR_V2_GATE_COMBINE", "add_clamp")),
            use_tadr_router=bool(getattr(cfg, "USE_TADR_ROUTER", False)),
            tadr_router_in_channels=int(getattr(cfg, "TADR_ROUTER_IN_CHANNELS", 4)),
            tadr_router_hidden=int(getattr(cfg, "TADR_ROUTER_HIDDEN", 16)),
            tadr_router_num_layers=int(getattr(cfg, "TADR_ROUTER_NUM_LAYERS", 2)),
            tadr_router_act=str(getattr(cfg, "TADR_ROUTER_ACT", "gelu")),
            tadr_use_coarse_prob=bool(getattr(cfg, "TADR_USE_COARSE_PROB", True)),
            tadr_use_uncertainty=bool(getattr(cfg, "TADR_USE_UNCERTAINTY", True)),
            tadr_use_sobel=bool(getattr(cfg, "TADR_USE_SOBEL", True)),
            tadr_use_coarse_boundary=bool(getattr(cfg, "TADR_USE_COARSE_BOUNDARY", True)),
            tadr_router_init_bias=float(getattr(cfg, "TADR_ROUTER_INIT_BIAS", 2.0)),
            tadr_router_zero_init_out=bool(getattr(cfg, "TADR_ROUTER_ZERO_INIT_OUT", True)),
            tadr_router_detach_inputs=bool(getattr(cfg, "TADR_ROUTER_DETACH_INPUTS", True)),
            tadr_router_min=float(getattr(cfg, "TADR_ROUTER_MIN", 0.0)),
            tadr_router_max=float(getattr(cfg, "TADR_ROUTER_MAX", 1.0)),
            use_proto_contrast=bool(getattr(cfg, "USE_PROTO_CONTRAST", False)),
            proto_feature_source=str(getattr(cfg, "PROTO_FEATURE_SOURCE", "dagp_semantic")),
            proto_use_proj_head=bool(getattr(cfg, "PROTO_USE_PROJ_HEAD", True)),
            proto_proj_hidden=int(getattr(cfg, "PROTO_PROJ_HIDDEN", 64)),
            proto_proj_dim=int(getattr(cfg, "PROTO_PROJ_DIM", 32)),
            proto_proj_act=str(getattr(cfg, "PROTO_PROJ_ACT", "gelu")),
            pa_dagp=_build_pa_dagp(in_channels, cfg),
            use_esa_ber=bool(getattr(cfg, "USE_ESA_BER", False)),
            return_graph_aux_for_ber=bool(
                getattr(cfg, "DAGP_SAFE_RETURN_GRAPH_AUX_FOR_BER", False)
            ),
            esa_ber_start_epoch=int(getattr(cfg, "ESA_BER_START_EPOCH", 21)),
            esa_ber_stop_epoch=int(getattr(cfg, "ESA_BER_STOP_EPOCH", 36)),
        )
    if head_type == "dagp_safe_csd_v1r":
        coarse_path = DAGPSafeHead(
            in_channels=in_channels,
            hidden=int(getattr(cfg, "DAGP_SAFE_HIDDEN", 64)),
            topk=int(getattr(cfg, "DAGP_SAFE_TOPK", 12)),
            tau=float(getattr(cfg, "DAGP_SAFE_TAU", 0.07)),
            alpha_max=float(getattr(cfg, "DAGP_SAFE_ALPHA_MAX", 0.05)),
            gamma_max=float(getattr(cfg, "DAGP_SAFE_GAMMA_MAX", 0.03)),
            warmup_epoch=int(getattr(cfg, "DAGP_SAFE_WARMUP_EPOCH", 6)),
            ramp_start_epoch=int(getattr(cfg, "DAGP_SAFE_RAMP_START_EPOCH", 7)),
            ramp_end_epoch=int(getattr(cfg, "DAGP_SAFE_RAMP_END_EPOCH", 15)),
            affinity_detach=bool(getattr(cfg, "DAGP_SAFE_AFFINITY_DETACH", True)),
            exclude_self=bool(getattr(cfg, "DAGP_SAFE_EXCLUDE_SELF", True)),
            use_prob_gate=bool(getattr(cfg, "DAGP_SAFE_USE_PROB_GATE", True)),
            prob_gate_sigma=float(getattr(cfg, "DAGP_SAFE_PROB_GATE_SIGMA", 0.25)),
            prob_gate_eps=float(getattr(cfg, "DAGP_SAFE_PROB_GATE_EPS", 1e-6)),
            zero_init_graph_pred=bool(getattr(cfg, "DAGP_SAFE_ZERO_INIT_GRAPH_PRED", True)),
            zero_init_value=bool(getattr(cfg, "DAGP_SAFE_ZERO_INIT_VALUE", False)),
            use_uncertainty_output_gate=bool(
                getattr(cfg, "DAGP_SAFE_USE_UNCERTAINTY_OUTPUT_GATE", False)
            ),
            uncertainty_power=float(getattr(cfg, "DAGP_SAFE_UNCERTAINTY_POWER", 1.0)),
            uncertainty_min=float(getattr(cfg, "DAGP_SAFE_UNCERTAINTY_MIN", 0.0)),
            uncertainty_max=float(getattr(cfg, "DAGP_SAFE_UNCERTAINTY_MAX", 1.0)),
            uncertainty_detach=bool(getattr(cfg, "DAGP_SAFE_UNCERTAINTY_DETACH", True)),
            use_ndr_branch=False,
            use_ndr_v2=False,
            use_tadr_router=False,
            use_proto_contrast=False,
            pa_dagp=_build_pa_dagp(in_channels, cfg),
        )
        residual_branch = CSDV1RResidual(
            in_channels=in_channels,
            sem_dim=int(getattr(cfg, "CSD_V1R_SEM_DIM", 64)),
            detail_dim=int(getattr(cfg, "CSD_V1R_DETAIL_DIM", 32)),
            fusion_dim=int(getattr(cfg, "CSD_V1R_FUSION_DIM", 64)),
            beta_max=float(getattr(cfg, "CSD_V1R_BETA_MAX", 0.05)),
            residual_clip=float(getattr(cfg, "CSD_V1R_RESIDUAL_CLIP", 2.0)),
            warmup_epoch=int(getattr(cfg, "CSD_V1R_WARMUP_EPOCH", 6)),
            ramp_start_epoch=int(getattr(cfg, "CSD_V1R_RAMP_START_EPOCH", 7)),
            ramp_end_epoch=int(getattr(cfg, "CSD_V1R_RAMP_END_EPOCH", 15)),
            bg_suppress_strength=float(getattr(cfg, "CSD_V1R_BG_SUPPRESS_STRENGTH", 0.70)),
            use_bg_detail_lock=bool(getattr(cfg, "CSD_V1R_USE_BG_DETAIL_LOCK", True)),
            use_res_zero_init=bool(getattr(cfg, "CSD_V1R_ZERO_INIT_RESIDUAL", True)),
            use_gate_bias_init=bool(getattr(cfg, "CSD_V1R_USE_GATE_BIAS_INIT", True)),
            gate_bias_init=float(getattr(cfg, "CSD_V1R_GATE_BIAS_INIT", -2.0)),
        )
        use_hr_bfr = bool(getattr(cfg, "USE_HR_BFR", False))
        hr_bfr_branch = (
            HRBFRV1Branch(
                in_channels=in_channels,
                sem_dim=int(getattr(cfg, "HR_BFR_SEM_DIM", 32)),
                detail_dim=int(getattr(cfg, "HR_BFR_DETAIL_DIM", 32)),
                hidden_dim=int(getattr(cfg, "HR_BFR_HIDDEN_DIM", 32)),
                residual_clip=float(getattr(cfg, "HR_BFR_RESIDUAL_CLIP", 2.0)),
                use_rgb=bool(getattr(cfg, "HR_BFR_USE_RGB", True)),
                use_sobel=bool(getattr(cfg, "HR_BFR_USE_SOBEL", True)),
                use_anchor_prob=bool(getattr(cfg, "HR_BFR_USE_ANCHOR_PROB", True)),
                use_anchor_uncert=bool(getattr(cfg, "HR_BFR_USE_ANCHOR_UNCERT", True)),
                use_dino_sem=bool(getattr(cfg, "HR_BFR_USE_DINO_SEM", True)),
                use_res_zero_init=bool(getattr(cfg, "HR_BFR_USE_RES_ZERO_INIT", True)),
            )
            if use_hr_bfr
            else None
        )
        return DAGPSafeCSDV1RHead(
            coarse_path=coarse_path,
            residual_branch=residual_branch,
            loss_size=int(getattr(cfg, "LOSS_SIZE", 68)),
            hr_bfr_branch=hr_bfr_branch,
            use_hr_bfr=use_hr_bfr,
            hr_size=int(getattr(cfg, "HR_BFR_SIZE", 136)),
            hr_beta_max=float(getattr(cfg, "HR_BFR_BETA_MAX", 0.05)),
            hr_warmup_epoch=int(getattr(cfg, "HR_BFR_WARMUP_EPOCH", 6)),
            hr_ramp_start_epoch=int(getattr(cfg, "HR_BFR_RAMP_START_EPOCH", 7)),
            hr_ramp_end_epoch=int(getattr(cfg, "HR_BFR_RAMP_END_EPOCH", 15)),
            hr_boundary_thresh=float(getattr(cfg, "HR_BFR_BOUNDARY_THRESH", 0.5)),
            hr_boundary_radius_68=int(getattr(cfg, "HR_BFR_BOUNDARY_RADIUS_68", 2)),
            hr_boundary_dilate_136=int(getattr(cfg, "HR_BFR_BOUNDARY_DILATE_136", 2)),
            hr_max_band_ratio=float(getattr(cfg, "HR_BFR_MAX_BAND_RATIO", 0.35)),
            hr_detach_anchor=bool(getattr(cfg, "HR_BFR_DETACH_ANCHOR", True)),
            hr_eval_res_scale=float(getattr(cfg, "HR_BFR_EVAL_RES_SCALE", 1.0)),
        )
    if head_type == "csd_v1":
        return CSDV1Head(
            in_channels=in_channels,
            loss_size=int(getattr(cfg, "LOSS_SIZE", 68)),
            sem_dim=int(getattr(cfg, "CSD_SEM_DIM", 64)),
            detail_dim=int(getattr(cfg, "CSD_DETAIL_DIM", 32)),
            fusion_dim=int(getattr(cfg, "CSD_FUSION_DIM", 64)),
            use_dagp_semantic=bool(getattr(cfg, "CSD_USE_DAGP_SEMANTIC", True)),
            use_local_context=bool(getattr(cfg, "CSD_USE_LOCAL_CONTEXT", True)),
            use_global_context=bool(getattr(cfg, "CSD_USE_GLOBAL_CONTEXT", True)),
            dagp_topk=int(getattr(cfg, "CSD_DAGP_TOPK", getattr(cfg, "DAGP_SAFE_TOPK", 12))),
            dagp_tau=float(getattr(cfg, "CSD_DAGP_TAU", getattr(cfg, "DAGP_SAFE_TAU", 0.07))),
            dagp_alpha_max=float(getattr(cfg, "CSD_DAGP_ALPHA_MAX", getattr(cfg, "DAGP_SAFE_ALPHA_MAX", 0.05))),
            dagp_gamma_max=float(getattr(cfg, "CSD_DAGP_GAMMA_MAX", getattr(cfg, "DAGP_SAFE_GAMMA_MAX", 0.03))),
            warmup_epoch=int(getattr(cfg, "CSD_WARMUP_EPOCH", 6)),
            ramp_start_epoch=int(getattr(cfg, "CSD_RAMP_START_EPOCH", 7)),
            ramp_end_epoch=int(getattr(cfg, "CSD_RAMP_END_EPOCH", 15)),
            beta_max=float(getattr(cfg, "CSD_BETA_MAX", 0.10)),
            residual_clip=float(getattr(cfg, "CSD_RESIDUAL_CLIP", 2.0)),
            bg_suppress_strength=float(getattr(cfg, "CSD_BG_SUPPRESS_STRENGTH", 0.70)),
            use_bg_detail_lock=bool(getattr(cfg, "CSD_USE_BG_DETAIL_LOCK", True)),
            use_boundary_aux=bool(getattr(cfg, "CSD_USE_BOUNDARY_AUX", True)),
        )
    if head_type == "context_residual":
        hidden = int(getattr(cfg, "CONTEXT_HEAD_HIDDEN", 64))
        gamma_init = float(getattr(cfg, "CONTEXT_HEAD_GAMMA_INIT", 0.0))
        return ContextResidualHead(in_channels, hidden=hidden, gamma_init=gamma_init)
    if head_type == "gated_context":
        hidden = int(getattr(cfg, "GATED_HEAD_HIDDEN", 64))
        gamma_init = float(getattr(cfg, "GATED_HEAD_GAMMA_INIT", 0.0))
        return GatedContextSegHead(in_channels, hidden=hidden, gamma_init=gamma_init)
    if head_type == "ml_context":
        return MultiLevelContextFusionDecoder(in_channels, cfg)
    if head_type == "sap_rcim":
        return SAPRCIMAdaptDecoder(in_channels, cfg)
    raise ValueError(f"Unknown HEAD_TYPE: {head_type}")


@torch.no_grad()
def update_ema(student, teacher, global_step, ema_weight=0.99):
    # global_step=0 时 alpha=0，可在 finetune reset 后让 teacher 首步对齐 student。
    alpha = min(1.0 - 1.0 / float(global_step + 1), ema_weight)
    for ema_p, p in zip(teacher.parameters(), student.parameters()):
        ema_p.data.mul_(alpha).add_(p.data, alpha=1.0 - alpha)
    for ema_b, b in zip(teacher.buffers(), student.buffers()):
        ema_b.copy_(b)
