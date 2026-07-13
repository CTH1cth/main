import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_projection(in_channels, channels, groups):
    return nn.Sequential(
        nn.Conv2d(in_channels, channels, kernel_size=1, bias=False),
        nn.GroupNorm(groups, channels),
        nn.GELU(),
    )


def _stats(prefix, value):
    detached = value.detach()
    return {
        f"{prefix}_mean": detached.mean(),
        f"{prefix}_min": detached.min(),
        f"{prefix}_max": detached.max(),
    }


def _off_diagonal_stats(vectors):
    """Return mean/max pairwise cosine, excluding each slot's diagonal."""
    if vectors.ndim != 3:
        raise RuntimeError(f"Slot vectors must be [B,K,D], got {list(vectors.shape)}")
    slots = int(vectors.shape[1])
    if slots < 2:
        zero = vectors.new_zeros(())
        return zero, zero
    normalized = F.normalize(vectors, dim=-1, eps=1e-6)
    similarity = torch.matmul(normalized, normalized.transpose(1, 2))
    mask = ~torch.eye(slots, dtype=torch.bool, device=vectors.device).unsqueeze(0)
    values = similarity.masked_select(mask.expand_as(similarity))
    return values.mean(), values.max()


class CrossLayerConsensusEncoder(nn.Module):
    def __init__(
        self,
        in_channels=384,
        dim=96,
        groups=8,
        gate_hidden=96,
        alpha_max=0.25,
        alpha_init_logit=-4.0,
        gate_bias_init=0.0,
        qc_detach=True,
        qc_eps=1e-6,
    ):
        super().__init__()
        self.dim = int(dim)
        self.alpha_max = float(alpha_max)
        self.qc_detach = bool(qc_detach)
        self.qc_eps = float(qc_eps)
        self.proj10 = _make_projection(in_channels, dim, groups)
        self.proj11 = _make_projection(in_channels, dim, groups)
        self.proj12 = _make_projection(in_channels, dim, groups)

        def make_gate():
            gate = nn.Sequential(
                nn.Conv2d(3 * dim + 1, gate_hidden, kernel_size=1, bias=False),
                nn.GroupNorm(groups, gate_hidden),
                nn.GELU(),
                nn.Conv2d(gate_hidden, dim, kernel_size=1, bias=True),
            )
            nn.init.constant_(gate[-1].bias, float(gate_bias_init))
            return gate

        self.gate10 = make_gate()
        self.gate11 = make_gate()
        self.alpha10_raw = nn.Parameter(torch.tensor(float(alpha_init_logit)))
        self.alpha11_raw = nn.Parameter(torch.tensor(float(alpha_init_logit)))
        self.fusion_refine = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.GroupNorm(groups, dim),
        )

    def forward(self, f10, f11, f12):
        z10 = self.proj10(f10)
        z11 = self.proj11(f11)
        z12 = self.proj12(f12)
        z10_norm = F.normalize(z10, dim=1, eps=self.qc_eps)
        z11_norm = F.normalize(z11, dim=1, eps=self.qc_eps)
        z12_norm = F.normalize(z12, dim=1, eps=self.qc_eps)
        cos10 = (z10_norm * z12_norm).sum(dim=1, keepdim=True)
        cos11 = (z11_norm * z12_norm).sum(dim=1, keepdim=True)
        gate10 = torch.sigmoid(self.gate10(torch.cat([z10, z12, (z10 - z12).abs(), cos10], dim=1)))
        gate11 = torch.sigmoid(self.gate11(torch.cat([z11, z12, (z11 - z12).abs(), cos11], dim=1)))
        alpha10 = self.alpha_max * torch.sigmoid(self.alpha10_raw)
        alpha11 = self.alpha_max * torch.sigmoid(self.alpha11_raw)
        consensus_input = z12 + alpha10 * gate10 * z10 + alpha11 * gate11 * z11
        consensus = F.gelu(consensus_input + self.fusion_refine(consensus_input))
        q_consensus = 0.5 * (((cos10 + 1.0) * 0.5) + ((cos11 + 1.0) * 0.5))
        q_consensus = q_consensus.clamp(0.0, 1.0)
        q_context = q_consensus.detach() if self.qc_detach else q_consensus
        aux = {
            "alpha10": alpha10.detach(),
            "alpha11": alpha11.detach(),
            **_stats("gate10", gate10),
            **_stats("gate11", gate11),
            **_stats("cos10", cos10),
            **_stats("cos11", cos11),
            **_stats("q_consensus", q_consensus),
        }
        return consensus, z10, z11, z12, q_consensus, q_context, aux


class ReliableAnchorEstimator(nn.Module):
    def __init__(self, dim=96, hidden=64, groups=8, classes=3):
        super().__init__()
        self.anchor_head = nn.Sequential(
            nn.Conv2d(dim, hidden, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, classes, kernel_size=1, bias=True),
        )

    def forward(self, feature):
        logits = self.anchor_head(feature)
        probability = torch.softmax(logits, dim=1)
        return logits, probability


class MultiAnchorContextReasoner(nn.Module):
    def __init__(
        self,
        dim=96,
        num_fg_slots=4,
        num_bg_slots=4,
        slot_seed=13579,
        attn_dropout=0.0,
        anchor_prior_eps=1e-4,
        context_hidden=192,
        gate_hidden=64,
        context_dropout=0.0,
        final_zero_init=True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.num_fg_slots = int(num_fg_slots)
        self.num_bg_slots = int(num_bg_slots)
        self.attn_dropout = float(attn_dropout)
        self.anchor_prior_eps = float(anchor_prior_eps)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.token_query_proj = nn.Linear(dim, dim)
        self.fg_slot_queries = nn.Parameter(torch.empty(num_fg_slots, dim))
        self.bg_slot_queries = nn.Parameter(torch.empty(num_bg_slots, dim))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(slot_seed))
            nn.init.orthogonal_(self.fg_slot_queries)
            nn.init.orthogonal_(self.bg_slot_queries)
        relation_layers = [
            nn.Linear(5 * dim, context_hidden),
            nn.LayerNorm(context_hidden),
            nn.GELU(),
        ]
        if float(context_dropout) > 0.0:
            relation_layers.append(nn.Dropout(float(context_dropout)))
        relation_layers.append(nn.Linear(context_hidden, dim))
        self.relation_mlp = nn.Sequential(*relation_layers)
        self.context_gate = nn.Sequential(
            nn.Linear(dim + 3, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        if final_zero_init:
            nn.init.zeros_(self.relation_mlp[-1].weight)
            nn.init.zeros_(self.relation_mlp[-1].bias)

    def _slot_pool(self, queries, key_tokens, value_tokens, prior):
        queries = F.normalize(queries, dim=-1, eps=1e-6)
        scores = torch.einsum("kd,bnd->bkn", queries, key_tokens) / math.sqrt(self.dim)
        scores = scores + torch.log(prior)
        attention = torch.softmax(scores, dim=-1)
        if self.attn_dropout > 0.0:
            attention = F.dropout(attention, p=self.attn_dropout, training=self.training)
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        slots = torch.einsum("bkn,bnd->bkd", attention, value_tokens)
        return F.normalize(slots, dim=-1, eps=1e-6), attention

    def forward(self, feature, anchor_prob, q_consensus):
        batch, channels, height, width = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        key_tokens = self.key_proj(tokens)
        value_tokens = self.value_proj(tokens)
        query_tokens = self.token_query_proj(tokens)
        a_fg = anchor_prob[:, 0:1]
        a_bg = anchor_prob[:, 1:2]
        a_amb = anchor_prob[:, 2:3]
        qc_flat = q_consensus.flatten(2)
        fg_prior = (a_fg.flatten(2) * qc_flat).clamp_min(self.anchor_prior_eps)
        bg_prior = (a_bg.flatten(2) * qc_flat).clamp_min(self.anchor_prior_eps)
        slots_fg, slot_attn_fg = self._slot_pool(
            self.fg_slot_queries, key_tokens, value_tokens, fg_prior
        )
        slots_bg, slot_attn_bg = self._slot_pool(
            self.bg_slot_queries, key_tokens, value_tokens, bg_prior
        )
        token_fg_score = torch.einsum("bnd,bkd->bnk", query_tokens, slots_fg) / math.sqrt(self.dim)
        token_bg_score = torch.einsum("bnd,bkd->bnk", query_tokens, slots_bg) / math.sqrt(self.dim)
        token_fg_attn = torch.softmax(token_fg_score, dim=-1)
        token_bg_attn = torch.softmax(token_bg_score, dim=-1)
        context_fg = torch.einsum("bnk,bkd->bnd", token_fg_attn, slots_fg)
        context_bg = torch.einsum("bnk,bkd->bnd", token_bg_attn, slots_bg)
        difference = context_fg - context_bg
        relation_input = torch.cat(
            [tokens, context_fg, context_bg, difference, difference.abs()], dim=-1
        )
        relation = self.relation_mlp(relation_input)
        a_amb_flat = a_amb.flatten(2).transpose(1, 2)
        qc_token = q_consensus.flatten(2).transpose(1, 2)
        anchor_gap = (a_fg - a_bg).abs().flatten(2).transpose(1, 2)
        gate_input = torch.cat([tokens, a_amb_flat, 1.0 - qc_token, anchor_gap], dim=-1)
        context_gate = torch.sigmoid(self.context_gate(gate_input))
        context_delta = context_gate * relation
        tokens_context = tokens + context_delta

        fg_slot_cos_mean, fg_slot_cos_max = _off_diagonal_stats(slots_fg)
        bg_slot_cos_mean, bg_slot_cos_max = _off_diagonal_stats(slots_bg)
        fg_attn_overlap_mean, fg_attn_overlap_max = _off_diagonal_stats(slot_attn_fg)
        bg_attn_overlap_mean, bg_attn_overlap_max = _off_diagonal_stats(slot_attn_bg)
        aux = {
            "fg_slot_cos_mean": fg_slot_cos_mean.detach(),
            "fg_slot_cos_max": fg_slot_cos_max.detach(),
            "bg_slot_cos_mean": bg_slot_cos_mean.detach(),
            "bg_slot_cos_max": bg_slot_cos_max.detach(),
            "fg_attn_overlap_mean": fg_attn_overlap_mean.detach(),
            "fg_attn_overlap_max": fg_attn_overlap_max.detach(),
            "bg_attn_overlap_mean": bg_attn_overlap_mean.detach(),
            "bg_attn_overlap_max": bg_attn_overlap_max.detach(),
            "fg_slot_attention_sum_error": (slot_attn_fg.sum(-1) - 1.0).abs().max().detach(),
            "bg_slot_attention_sum_error": (slot_attn_bg.sum(-1) - 1.0).abs().max().detach(),
            **_stats("context_gate", context_gate),
            "context_relation_abs_mean": relation.detach().abs().mean(),
            "context_relation_abs_max": relation.detach().abs().max(),
            "context_delta_abs_mean": context_delta.detach().abs().mean(),
            "context_delta_abs_max": context_delta.detach().abs().max(),
            "context_delta_norm_ratio": (
                context_delta.detach().pow(2).mean().sqrt()
                / tokens.detach().pow(2).mean().sqrt().clamp_min(1e-6)
            ),
        }
        context_feature = tokens_context.transpose(1, 2).reshape(batch, channels, height, width)
        return context_feature, aux


class ResidualDWContextBlock(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.gn1 = nn.GroupNorm(groups, channels)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.gn2 = nn.GroupNorm(groups, channels)

    def forward(self, value):
        residual = F.gelu(self.gn1(self.dw(value)))
        residual = self.gn2(self.pw(residual))
        return F.gelu(value + residual)


class CACDSemanticDetailDecoder(nn.Module):
    def __init__(self, dim=96, detail_dim=32, out_dim=64, groups=8, context_blocks=2):
        super().__init__()
        self.context_blocks = nn.Sequential(
            *[ResidualDWContextBlock(dim, groups=groups) for _ in range(int(context_blocks))]
        )
        self.coarse_head = nn.Conv2d(dim, 1, kernel_size=1)
        self.detail_encoder = nn.Sequential(
            nn.Conv2d(4, detail_dim, 3, padding=1, bias=False),
            nn.GroupNorm(4, detail_dim),
            nn.GELU(),
            nn.Conv2d(detail_dim, detail_dim, 3, padding=1, bias=False),
            nn.GroupNorm(4, detail_dim),
            nn.GELU(),
        )
        self.detail_gate = nn.Conv2d(dim, detail_dim, kernel_size=1)
        self.final_decoder = nn.Sequential(
            nn.Conv2d(dim + detail_dim + 3 + 1, dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, dim),
            nn.GELU(),
            ResidualDWContextBlock(dim, groups=groups),
            nn.Conv2d(dim, out_dim, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_dim),
            nn.GELU(),
        )
        self.final_head = nn.Conv2d(out_dim, 1, kernel_size=1)

    def forward(self, context_37, image_68, sobel_68, anchor_prob_37, q_consensus_37):
        semantic_37 = self.context_blocks(context_37)
        coarse_logits_37 = self.coarse_head(semantic_37)
        coarse_logits_68 = F.interpolate(
            coarse_logits_37, size=(68, 68), mode="bilinear", align_corners=False
        )
        semantic_68 = F.interpolate(
            semantic_37, size=(68, 68), mode="bilinear", align_corners=False
        )
        detail_68 = self.detail_encoder(torch.cat([image_68, sobel_68], dim=1))
        detail_gate = torch.sigmoid(self.detail_gate(semantic_68))
        detail_guided = detail_68 * detail_gate
        anchor_prob_68 = F.interpolate(
            anchor_prob_37, size=(68, 68), mode="bilinear", align_corners=False
        )
        qc_68 = F.interpolate(
            q_consensus_37, size=(68, 68), mode="bilinear", align_corners=False
        )
        decoder_input = torch.cat(
            [semantic_68, detail_guided, anchor_prob_68, qc_68], dim=1
        )
        final_feature_68 = self.final_decoder(decoder_input)
        final_logits_68 = self.final_head(final_feature_68)
        aux = {
            **_stats("detail_gate", detail_gate),
            "detail_feature_abs_mean": detail_68.detach().abs().mean(),
            "semantic_feature_abs_mean": semantic_37.detach().abs().mean(),
        }
        return final_logits_68, coarse_logits_37, coarse_logits_68, aux


class CACDV1BaseHead(nn.Module):
    def __init__(self, in_channels, cfg):
        super().__init__()
        self.version = str(getattr(cfg, "CACD_VERSION", "v1_base_last3_consensus_anchor_context_68"))
        self.feature_size = int(getattr(cfg, "CACD_FEATURE_SIZE", 37))
        self.output_size = int(getattr(cfg, "CACD_OUTPUT_SIZE", 68))
        if self.output_size != 68:
            raise ValueError(f"CACD-v1-Base requires output size 68, got {self.output_size}.")
        dim = int(getattr(cfg, "CACD_DIM", 96))
        groups = int(getattr(cfg, "CACD_GN_GROUPS", 8))
        self.consensus_encoder = CrossLayerConsensusEncoder(
            in_channels=int(in_channels),
            dim=dim,
            groups=groups,
            gate_hidden=int(getattr(cfg, "CACD_GATE_HIDDEN", 96)),
            alpha_max=float(getattr(cfg, "CACD_FUSION_ALPHA_MAX", 0.25)),
            alpha_init_logit=float(getattr(cfg, "CACD_FUSION_ALPHA_INIT_LOGIT", -4.0)),
            gate_bias_init=float(getattr(cfg, "CACD_GATE_BIAS_INIT", 0.0)),
            qc_detach=bool(getattr(cfg, "CACD_QC_DETACH", True)),
            qc_eps=float(getattr(cfg, "CACD_QC_EPS", 1e-6)),
        )
        self.base_head = nn.Conv2d(dim, 1, kernel_size=1)
        self.anchor_estimator = ReliableAnchorEstimator(
            dim=dim,
            hidden=int(getattr(cfg, "CACD_ANCHOR_HIDDEN", 64)),
            groups=groups,
            classes=int(getattr(cfg, "CACD_ANCHOR_CLASSES", 3)),
        )
        self.context_reasoner = MultiAnchorContextReasoner(
            dim=dim,
            num_fg_slots=int(getattr(cfg, "CACD_NUM_FG_SLOTS", 4)),
            num_bg_slots=int(getattr(cfg, "CACD_NUM_BG_SLOTS", 4)),
            slot_seed=int(getattr(cfg, "CACD_SLOT_INIT_SEED", 13579)),
            attn_dropout=float(getattr(cfg, "CACD_SLOT_ATTN_DROPOUT", 0.0)),
            anchor_prior_eps=float(getattr(cfg, "CACD_ANCHOR_PRIOR_EPS", 1e-4)),
            context_hidden=int(getattr(cfg, "CACD_CONTEXT_HIDDEN", 192)),
            gate_hidden=int(getattr(cfg, "CACD_CONTEXT_GATE_HIDDEN", 64)),
            context_dropout=float(getattr(cfg, "CACD_CONTEXT_DROPOUT", 0.0)),
            final_zero_init=bool(getattr(cfg, "CACD_CONTEXT_FINAL_ZERO_INIT", True)),
        )
        self.decoder = CACDSemanticDetailDecoder(
            dim=dim,
            detail_dim=int(getattr(cfg, "CACD_DETAIL_DIM", 32)),
            out_dim=int(getattr(cfg, "CACD_DECODER_OUT_DIM", 64)),
            groups=groups,
            context_blocks=int(getattr(cfg, "CACD_CONTEXT_BLOCKS", 2)),
        )
        self.register_buffer("current_epoch_tensor", torch.tensor(1, dtype=torch.long), persistent=True)

    def set_epoch(self, epoch):
        self.current_epoch_tensor.fill_(int(epoch))

    def _validate_inputs(self, features, image_68, sobel_68):
        if not isinstance(features, dict) or set(features) != {"f10", "f11", "f12"}:
            raise RuntimeError(
                f"CACD input must contain exactly f10/f11/f12, got "
                f"{sorted(features) if isinstance(features, dict) else type(features)}"
            )
        batch = None
        for name in ("f10", "f11", "f12"):
            value = features[name]
            if value.ndim != 4 or list(value.shape[1:]) != [384, self.feature_size, self.feature_size]:
                raise RuntimeError(f"CACD {name} shape mismatch: {list(value.shape)}")
            if not bool(torch.isfinite(value).all().item()):
                raise RuntimeError(f"CACD {name} contains NaN/Inf.")
            batch = int(value.shape[0]) if batch is None else batch
            if int(value.shape[0]) != batch:
                raise RuntimeError("CACD feature batch sizes differ.")
        if image_68 is None or list(image_68.shape) != [batch, 3, 68, 68]:
            raise RuntimeError(f"CACD image_68 shape mismatch: {None if image_68 is None else list(image_68.shape)}")
        if sobel_68 is None or list(sobel_68.shape) != [batch, 1, 68, 68]:
            raise RuntimeError(f"CACD sobel_68 shape mismatch: {None if sobel_68 is None else list(sobel_68.shape)}")
        if not bool(torch.isfinite(image_68).all().item()) or not bool(torch.isfinite(sobel_68).all().item()):
            raise RuntimeError("CACD image_68/sobel_68 contains NaN/Inf.")

    def forward(self, features, image_68=None, sobel_68=None, return_aux=False):
        self._validate_inputs(features, image_68, sobel_68)
        consensus, z10, z11, z12, q_consensus, q_context, consensus_aux = self.consensus_encoder(
            features["f10"], features["f11"], features["f12"]
        )
        base_logits_37 = self.base_head(z12)
        base_logits_68 = F.interpolate(
            base_logits_37, size=(68, 68), mode="bilinear", align_corners=False
        )
        anchor_logits_37, anchor_prob_37 = self.anchor_estimator(consensus)
        probability_error = (anchor_prob_37.sum(dim=1) - 1.0).abs().max()
        if float(probability_error.detach().item()) > 1e-5:
            raise RuntimeError(
                f"CACD anchor probability sum error exceeds 1e-5: {float(probability_error):.8g}"
            )
        context_37, context_aux = self.context_reasoner(
            consensus, anchor_prob_37, q_context
        )
        for key in ("fg_slot_attention_sum_error", "bg_slot_attention_sum_error"):
            if float(context_aux[key].item()) > 1e-5:
                raise RuntimeError(f"CACD {key} exceeds 1e-5: {float(context_aux[key]):.8g}")
        final_logits_68, coarse_logits_37, coarse_logits_68, decoder_aux = self.decoder(
            context_37, image_68, sobel_68, anchor_prob_37, q_context
        )
        tensors = (
            base_logits_37,
            base_logits_68,
            anchor_logits_37,
            anchor_prob_37,
            context_37,
            coarse_logits_37,
            coarse_logits_68,
            final_logits_68,
        )
        if not all(bool(torch.isfinite(value).all().item()) for value in tensors):
            raise RuntimeError("CACD forward produced NaN/Inf.")
        cacd_aux = {
            **consensus_aux,
            **context_aux,
            **decoder_aux,
            "anchor_probability_sum_error": probability_error.detach(),
            "anchor_fg_mean": anchor_prob_37[:, 0:1].detach().mean(),
            "anchor_bg_mean": anchor_prob_37[:, 1:2].detach().mean(),
            "anchor_amb_mean": anchor_prob_37[:, 2:3].detach().mean(),
            "base_coarse_abs_diff": (
                base_logits_68.detach() - coarse_logits_68.detach()
            ).abs().mean(),
            "coarse_final_abs_diff": (
                coarse_logits_68.detach() - final_logits_68.detach()
            ).abs().mean(),
            "z10_abs_mean": z10.detach().abs().mean(),
            "z11_abs_mean": z11.detach().abs().mean(),
            "z12_abs_mean": z12.detach().abs().mean(),
        }
        return {
            "logits": final_logits_68,
            "final_logits": final_logits_68,
            "coarse_logits": coarse_logits_68,
            "coarse_logits_37": coarse_logits_37,
            "coarse_logits_68": coarse_logits_68,
            "base_logits": base_logits_68,
            "base_logits_37": base_logits_37,
            "anchor_logits": anchor_logits_37,
            "anchor_prob": anchor_prob_37,
            "consensus_conf": q_consensus,
            "cacd_aux": cacd_aux,
        }
