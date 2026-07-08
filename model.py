import math

import torch
import torch.nn.functional as F
from torch import nn


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

    def forward(self, image_68, coarse_prob_68):
        sobel_68 = self.compute_sobel(image_68)
        inputs = []
        if self.input_rgb:
            inputs.append(image_68)
        if self.input_sobel:
            inputs.append(sobel_68)
        if self.input_coarse_prob:
            inputs.append(coarse_prob_68)
        ndr_input = torch.cat(inputs, dim=1)
        raw_residual = self.out_conv(self.block3(self.block2(self.block1(ndr_input))))
        residual = raw_residual.tanh() * raw_residual.new_tensor(self.residual_clip)
        return residual, sobel_68


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
        residual_logits_68, sobel_68 = self.ndr_branch(image_68.to(dtype=coarse_logits_68.dtype), coarse_prob_68)

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

        if not return_aux:
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

    def forward(self, feat, image_68=None, return_aux=False):
        bsz, _, height, width = feat.shape
        base_logits = self.base_head(feat)
        scale = self._ramp_scale()
        alpha_eff = self.alpha_max * scale
        gamma_eff = self.gamma_max * scale

        if alpha_eff == 0.0 or gamma_eff == 0.0:
            graph_logits = torch.zeros_like(base_logits)
            semantic_feat = self.proj(feat) if return_aux and self.use_proto_contrast else None
            if self.use_ndr_branch:
                return self._apply_ndr(
                    base_logits,
                    image_68,
                    base_logits,
                    graph_logits,
                    scale,
                    alpha_eff,
                    gamma_eff,
                    self._uncertainty_output_gate(base_logits),
                    return_aux,
                    semantic_feat=semantic_feat,
                )
            if return_aux:
                uncertainty_gate = self._uncertainty_output_gate(base_logits)
                return self._aux_output(
                    base_logits,
                    base_logits,
                    graph_logits,
                    scale,
                    alpha_eff,
                    gamma_eff,
                    uncertainty_gate,
                    semantic_feat=semantic_feat,
                )
            return base_logits

        if self.affinity_detach:
            with torch.no_grad():
                attn, topk_idx = self._topk_affinity(feat)
        else:
            attn, topk_idx = self._topk_affinity(feat)
        if self.use_prob_gate:
            attn = self._apply_prob_gate(attn, topk_idx, base_logits)

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
        if self.use_ndr_branch:
            return self._apply_ndr(
                logits,
                image_68,
                base_logits,
                graph_logits,
                scale,
                alpha_eff,
                gamma_eff,
                uncertainty_gate,
                return_aux,
                semantic_feat=z_prop_map if return_aux and self.use_proto_contrast else None,
            )
        if return_aux:
            return self._aux_output(
                logits,
                base_logits,
                graph_logits,
                scale,
                alpha_eff,
                gamma_eff,
                uncertainty_gate,
                semantic_feat=z_prop_map if self.use_proto_contrast else None,
            )
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


def build_seg_head(in_channels, cfg):
    head_type = str(getattr(cfg, "HEAD_TYPE", "simple")).lower()
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
