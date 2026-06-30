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

        self.base_head = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.proj = nn.Conv2d(in_channels, self.hidden, kernel_size=1)
        self.value = nn.Linear(self.hidden, self.hidden)
        self.graph_pred = nn.Conv2d(self.hidden, 1, kernel_size=1)
        self.register_buffer("current_epoch_tensor", torch.zeros(1, dtype=torch.float32))

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

    def _aux_output(
        self,
        logits,
        base_logits,
        graph_logits,
        scale,
        alpha_eff,
        gamma_eff,
        uncertainty_gate=None,
    ):
        scalar = base_logits.new_tensor(float(scale))
        unc_mean, unc_min, unc_max = self._uncertainty_stats(base_logits, uncertainty_gate)
        return {
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

    def forward(self, feat, return_aux=False):
        bsz, _, height, width = feat.shape
        base_logits = self.base_head(feat)
        scale = self._ramp_scale()
        alpha_eff = self.alpha_max * scale
        gamma_eff = self.gamma_max * scale

        if alpha_eff == 0.0 or gamma_eff == 0.0:
            if return_aux:
                graph_logits = torch.zeros_like(base_logits)
                uncertainty_gate = self._uncertainty_output_gate(base_logits)
                return self._aux_output(
                    base_logits,
                    base_logits,
                    graph_logits,
                    scale,
                    alpha_eff,
                    gamma_eff,
                    uncertainty_gate,
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
        if return_aux:
            return self._aux_output(
                logits,
                base_logits,
                graph_logits,
                scale,
                alpha_eff,
                gamma_eff,
                uncertainty_gate,
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
