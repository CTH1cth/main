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
        self.version = str(
            self.config.get("version", "ap_stcr_v1_full_pixel_37_to_68")
        ).strip()
        supported_versions = {
            "ap_stcr_v1_full_pixel_37_to_68",
            "ap_stcr_v2_conflict_only_pass_through",
            "ap_stcr_v3_soft_disagreement_bounded_continuation",
            "ap_stcr_v4_transition_envelope_non_compensatory",
            "ap_stcr_v4_semantic_only_ablation",
        }
        if self.version not in supported_versions:
            raise RuntimeError(
                f"Unsupported AP-STCR version={self.version!r}; "
                f"expected one of {sorted(supported_versions)}."
            )
        self.conflict_only = bool(
            self.config.get("conflict_only", False)
        )
        self.is_v2 = (
            self.version == "ap_stcr_v2_conflict_only_pass_through"
        )
        self.is_v3 = (
            self.version
            == "ap_stcr_v3_soft_disagreement_bounded_continuation"
        )
        self.is_v4 = (
            self.version
            == "ap_stcr_v4_transition_envelope_non_compensatory"
        )
        self.is_semantic_only = (
            self.version == "ap_stcr_v4_semantic_only_ablation"
        )
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
        self.temporal_empty_support = float(
            self.config.get("temporal_empty_support", 1.0)
        )
        self.rejection_max = float(
            self.config.get("rejection_max", 0.35)
        )
        self.support_neutral_point = float(
            self.config.get("support_neutral_point", 0.50)
        )
        self.use_soft_deviation = True
        self.use_soft_correction_for_semantic = True
        self.soft_rejection_strength = 0.35
        self.evidence_fusion = "negative_soft_or"
        self.use_temporal_evidence = True
        self.temporal_history_enabled = True
        self.use_transition_envelope = True
        self.evidence_rejection_strength = 0.35
        self.min_local_acceptance = 0.65
        if self.is_v3:
            self.use_soft_deviation = bool(
                self.config.get("use_soft_deviation", True)
            )
            self.use_soft_correction_for_semantic = bool(
                self.config.get("use_soft_correction_for_semantic", True)
            )
            self.soft_rejection_strength = float(
                self.config.get("soft_rejection_strength", 0.35)
            )
            self.min_local_acceptance = float(
                self.config.get("min_local_acceptance", 0.65)
            )
        elif self.is_v4:
            self.use_soft_deviation = bool(
                self.config.get("use_soft_deviation", True)
            )
            self.use_soft_correction_for_semantic = bool(
                self.config.get("use_soft_correction_for_semantic", True)
            )
            self.evidence_fusion = str(
                self.config.get("evidence_fusion", "negative_soft_or")
            ).strip()
            self.use_transition_envelope = bool(
                self.config.get("use_transition_envelope", True)
            )
            self.evidence_rejection_strength = float(
                self.config.get("evidence_rejection_strength", 0.35)
            )
            self.min_local_acceptance = float(
                self.config.get("min_local_acceptance", 0.65)
            )
        elif self.is_semantic_only:
            self.use_soft_deviation = bool(
                self.config.get("use_soft_deviation", True)
            )
            self.use_soft_correction_for_semantic = bool(
                self.config.get("use_soft_correction_for_semantic", True)
            )
            self.evidence_fusion = str(
                self.config.get("evidence_fusion", "semantic_only")
            ).strip()
            self.use_temporal_evidence = bool(
                self.config.get("use_temporal_evidence", False)
            )
            self.temporal_history_enabled = bool(
                self.config.get("temporal_history_enabled", False)
            )
            self.use_transition_envelope = bool(
                self.config.get("use_transition_envelope", True)
            )
            self.evidence_rejection_strength = float(
                self.config.get("evidence_rejection_strength", 0.35)
            )
            self.min_local_acceptance = float(
                self.config.get("min_local_acceptance", 0.65)
            )
        self.lambda_semantic = None
        self.lambda_temporal = None
        if not (
            self.is_v2
            or self.is_v3
            or self.is_v4
            or self.is_semantic_only
        ):
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
            "eps": self.eps,
        }
        if not (
            self.is_v2
            or self.is_v3
            or self.is_v4
            or self.is_semantic_only
        ):
            positive_values.update(
                {
                    "lambda_semantic": self.lambda_semantic,
                    "lambda_temporal": self.lambda_temporal,
                }
            )
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
            raise RuntimeError("AP-STCR requires 37 evidence and 68 loss.")
        if not 0.0 <= self.temporal_empty_support <= 1.0:
            raise RuntimeError(
                "AP-STCR temporal_empty_support must be in [0,1]."
            )
        if self.is_v2:
            if not self.conflict_only:
                raise RuntimeError(
                    "AP-STCR v2 requires conflict_only=True."
                )
            if "lambda_semantic" in self.config or "lambda_temporal" in self.config:
                raise RuntimeError(
                    "AP-STCR v2 must not configure lambda_semantic or "
                    "lambda_temporal."
                )
            if (
                not math.isfinite(self.rejection_max)
                or not 0.0 <= self.rejection_max <= 1.0
            ):
                raise RuntimeError(
                    "AP-STCR v2 rejection_max must be in [0,1]."
                )
            if (
                not math.isfinite(self.support_neutral_point)
                or not 0.0 < self.support_neutral_point < 1.0
            ):
                raise RuntimeError(
                    "AP-STCR v2 support_neutral_point must be in (0,1)."
                )
        elif self.is_v3:
            forbidden = {
                "conflict_only",
                "rejection_max",
                "support_neutral_point",
                "lambda_semantic",
                "lambda_temporal",
            }.intersection(self.config)
            if forbidden:
                raise RuntimeError(
                    "AP-STCR v3 must not configure legacy conflict/lambda "
                    f"fields: {sorted(forbidden)}."
                )
            if not self.use_soft_deviation:
                raise RuntimeError(
                    "AP-STCR v3 requires use_soft_deviation=True."
                )
            if not self.use_soft_correction_for_semantic:
                raise RuntimeError(
                    "AP-STCR v3 requires "
                    "use_soft_correction_for_semantic=True."
                )
            if (
                not math.isfinite(self.soft_rejection_strength)
                or not 0.0 <= self.soft_rejection_strength <= 1.0
            ):
                raise RuntimeError(
                    "AP-STCR v3 soft_rejection_strength must be in [0,1]."
                )
            if (
                not math.isfinite(self.min_local_acceptance)
                or not 0.0 <= self.min_local_acceptance <= 1.0
            ):
                raise RuntimeError(
                    "AP-STCR v3 min_local_acceptance must be in [0,1]."
                )
            if not math.isclose(
                self.min_local_acceptance,
                1.0 - self.soft_rejection_strength,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    "AP-STCR v3 requires min_local_acceptance == "
                    "1 - soft_rejection_strength."
                )
        elif self.is_v4:
            forbidden = {
                "conflict_only",
                "rejection_max",
                "support_neutral_point",
                "lambda_semantic",
                "lambda_temporal",
                "soft_rejection_strength",
                "fused_support",
                "support_deficiency",
                "soft_inertia",
            }.intersection(self.config)
            if forbidden:
                raise RuntimeError(
                    "AP-STCR v4 must not configure legacy conflict/fused "
                    f"fields: {sorted(forbidden)}."
                )
            if not self.use_soft_deviation:
                raise RuntimeError(
                    "AP-STCR v4 requires use_soft_deviation=True."
                )
            if not self.use_soft_correction_for_semantic:
                raise RuntimeError(
                    "AP-STCR v4 requires "
                    "use_soft_correction_for_semantic=True."
                )
            if self.evidence_fusion != "negative_soft_or":
                raise RuntimeError(
                    "AP-STCR v4 requires "
                    "evidence_fusion='negative_soft_or'."
                )
            if not self.use_transition_envelope:
                raise RuntimeError(
                    "AP-STCR v4 requires use_transition_envelope=True."
                )
            if (
                not math.isfinite(self.evidence_rejection_strength)
                or not 0.0 <= self.evidence_rejection_strength <= 1.0
            ):
                raise RuntimeError(
                    "AP-STCR v4 evidence_rejection_strength must be in "
                    "[0,1]."
                )
            if (
                not math.isfinite(self.min_local_acceptance)
                or not 0.0 <= self.min_local_acceptance <= 1.0
            ):
                raise RuntimeError(
                    "AP-STCR v4 min_local_acceptance must be in [0,1]."
                )
            if not math.isclose(
                self.min_local_acceptance,
                1.0 - self.evidence_rejection_strength,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    "AP-STCR v4 requires min_local_acceptance == "
                    "1 - evidence_rejection_strength."
                )
        elif self.is_semantic_only:
            forbidden = {
                "conflict_only",
                "rejection_max",
                "support_neutral_point",
                "lambda_semantic",
                "lambda_temporal",
                "soft_rejection_strength",
                "fused_support",
                "support_deficiency",
                "soft_inertia",
            }.intersection(self.config)
            if forbidden:
                raise RuntimeError(
                    "AP-STCR A1 must not configure legacy conflict/fused "
                    f"fields: {sorted(forbidden)}."
                )
            if not self.use_soft_deviation:
                raise RuntimeError(
                    "AP-STCR A1 requires use_soft_deviation=True."
                )
            if not self.use_soft_correction_for_semantic:
                raise RuntimeError(
                    "AP-STCR A1 requires "
                    "use_soft_correction_for_semantic=True."
                )
            if self.evidence_fusion != "semantic_only":
                raise RuntimeError(
                    "AP-STCR A1 requires evidence_fusion='semantic_only'."
                )
            if self.use_temporal_evidence:
                raise RuntimeError(
                    "AP-STCR A1 requires use_temporal_evidence=False."
                )
            if self.temporal_history_enabled:
                raise RuntimeError(
                    "AP-STCR A1 requires temporal_history_enabled=False."
                )
            if not self.use_transition_envelope:
                raise RuntimeError(
                    "AP-STCR A1 requires use_transition_envelope=True."
                )
            if (
                not math.isfinite(self.evidence_rejection_strength)
                or not 0.0 <= self.evidence_rejection_strength <= 1.0
            ):
                raise RuntimeError(
                    "AP-STCR A1 evidence_rejection_strength must be in "
                    "[0,1]."
                )
            if (
                not math.isfinite(self.min_local_acceptance)
                or not 0.0 <= self.min_local_acceptance <= 1.0
            ):
                raise RuntimeError(
                    "AP-STCR A1 min_local_acceptance must be in [0,1]."
                )
            if not math.isclose(
                self.min_local_acceptance,
                1.0 - self.evidence_rejection_strength,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    "AP-STCR A1 requires min_local_acceptance == "
                    "1 - evidence_rejection_strength."
                )
        elif self.conflict_only:
            raise RuntimeError(
                "AP-STCR v1 cannot enable conflict_only routing."
            )
        self.semantic_cache = APSTCRSemanticCache(
            sample_keys,
            height=self.evidence_resolution,
            width=self.evidence_resolution,
        )
        self.manifest_hash = self.semantic_cache.manifest_hash
        self.history_bank = None
        if not self.is_semantic_only:
            self.history_bank = APSTCRTemporalHistoryBank(
                sample_keys,
                window=self.temporal_window,
                height=self.evidence_resolution,
                width=self.evidence_resolution,
            )
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
    def compute_soft_semantic_support(
        self,
        semantic_margin,
        teacher_soft,
        fixed_pseudo,
    ):
        semantic_margin = _require_map(
            "semantic_margin", semantic_margin
        ).detach().float()
        teacher_soft = _require_unit_map(
            "teacher_soft",
            teacher_soft,
            shape=semantic_margin.shape,
        ).detach().float()
        fixed_pseudo = _require_unit_map(
            "fixed_pseudo",
            fixed_pseudo,
            shape=semantic_margin.shape,
        ).detach().float()
        soft_correction = teacher_soft - fixed_pseudo
        correction_direction = torch.tanh(
            soft_correction / self.tau_delta
        )
        semantic_direction = torch.tanh(
            semantic_margin / self.tau_margin
        )
        support = 0.5 * (
            1.0 + semantic_direction * correction_direction
        )
        return support.clamp(0.0, 1.0).detach(), soft_correction.detach()

    @torch.no_grad()
    def compute_semantic_contradiction(
        self,
        semantic_margin,
        soft_correction,
    ):
        semantic_margin = _require_map(
            "semantic_margin", semantic_margin
        ).detach().float()
        soft_correction = _require_map(
            "soft_correction",
            soft_correction,
            shape=semantic_margin.shape,
        ).detach().float()
        semantic_direction = torch.tanh(
            semantic_margin / self.tau_margin
        )
        correction_direction = torch.tanh(
            soft_correction / self.tau_delta
        )
        semantic_contradiction = torch.relu(
            -semantic_direction * correction_direction
        ).clamp(0.0, 1.0)
        return semantic_contradiction.detach()

    @torch.no_grad()
    def compute_temporal_support(
        self,
        teacher_soft,
        fixed_pseudo,
        sample_indices,
        datasets,
        stems,
    ):
        if self.history_bank is None:
            raise RuntimeError(
                "AP-STCR A1 has temporal evidence/history disabled."
            )
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
        empty_support = torch.full_like(
            support, self.temporal_empty_support
        )
        support = torch.where(valid, support, empty_support)
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
        source_conflict=None,
    ):
        if self.is_v2:
            if source_conflict is None:
                raise RuntimeError(
                    "AP-STCR v2 local acceptance requires source_conflict."
                )
            return self.compute_conflict_only_acceptance(
                semantic_support,
                temporal_support,
                source_conflict,
            )["local_acceptance_37"]
        evidence = (
            self.lambda_semantic * (1.0 - semantic_support.detach().float())
            + self.lambda_temporal * (1.0 - temporal_support.detach().float())
        )
        return (1.0 / (1.0 + evidence)).clamp(0.0, 1.0).detach()

    @torch.no_grad()
    def compute_source_conflict(
        self,
        fixed_pseudo_37,
        teacher_binary_37,
    ):
        fixed_pseudo_37 = _require_unit_map(
            "fixed_pseudo_37", fixed_pseudo_37
        ).detach()
        teacher_binary_37 = _require_unit_map(
            "teacher_binary_37",
            teacher_binary_37,
            shape=fixed_pseudo_37.shape,
        ).detach()
        fixed_hard_37 = (fixed_pseudo_37.float() > 0.5).float()
        teacher_hard_37 = (teacher_binary_37.float() > 0.5).float()
        source_conflict_37 = (
            fixed_hard_37 != teacher_hard_37
        ).float()
        return {
            "fixed_hard_37": fixed_hard_37.detach(),
            "teacher_binary_37": teacher_hard_37.detach(),
            "source_conflict_37": source_conflict_37.detach(),
        }

    @torch.no_grad()
    def compute_conflict_only_acceptance(
        self,
        semantic_support,
        temporal_support,
        source_conflict,
    ):
        semantic_support = _require_unit_map(
            "semantic_support", semantic_support
        ).detach().float()
        temporal_support = _require_unit_map(
            "temporal_support",
            temporal_support,
            shape=semantic_support.shape,
        ).detach().float()
        source_conflict = _require_unit_map(
            "source_conflict",
            source_conflict,
            shape=semantic_support.shape,
        ).detach().float()
        if not bool(
            ((source_conflict == 0.0) | (source_conflict == 1.0))
            .all()
            .item()
        ):
            raise RuntimeError(
                "AP-STCR v2 source_conflict must be binary."
            )
        neutral = self.support_neutral_point
        semantic_negative = (
            (neutral - semantic_support) / neutral
        ).clamp(0.0, 1.0)
        temporal_negative = (
            (neutral - temporal_support) / neutral
        ).clamp(0.0, 1.0)
        negative_evidence = 0.5 * (
            semantic_negative + temporal_negative
        )
        local_rejection = (
            self.rejection_max * negative_evidence
        ).clamp(0.0, self.rejection_max)
        local_acceptance = (
            1.0 - source_conflict * local_rejection
        ).clamp(0.0, 1.0)

        minimum = 1.0 - self.rejection_max
        if (
            float(local_acceptance.min().item()) < minimum - 1e-6
            or float(local_acceptance.max().item()) > 1.0 + 1e-6
        ):
            raise RuntimeError(
                "AP-STCR v2 local acceptance escaped its configured bounds."
            )
        non_conflict = source_conflict == 0.0
        if bool(non_conflict.any().item()) and not torch.equal(
            local_acceptance[non_conflict],
            torch.ones_like(local_acceptance[non_conflict]),
        ):
            raise RuntimeError(
                "AP-STCR v2 must pass every non-conflict pixel unchanged."
            )
        return {
            "semantic_negative_37": semantic_negative.detach(),
            "temporal_negative_37": temporal_negative.detach(),
            "negative_evidence_37": negative_evidence.detach(),
            "local_rejection_37": local_rejection.detach(),
            "local_acceptance_37": local_acceptance.detach(),
        }

    @torch.no_grad()
    def compute_soft_disagreement_acceptance(
        self,
        semantic_support,
        temporal_support,
        history_valid,
        soft_deviation,
    ):
        semantic_support = _require_unit_map(
            "semantic_support", semantic_support
        ).detach().float()
        temporal_support = _require_unit_map(
            "temporal_support",
            temporal_support,
            shape=semantic_support.shape,
        ).detach().float()
        soft_deviation = _require_unit_map(
            "soft_deviation",
            soft_deviation,
            shape=semantic_support.shape,
        ).detach().float()
        if not torch.is_tensor(history_valid):
            raise TypeError("AP-STCR history_valid must be a tensor.")
        history_valid = history_valid.detach().to(
            device=semantic_support.device
        )
        batch_size = int(semantic_support.shape[0])
        if tuple(history_valid.shape) == (batch_size,):
            history_valid = history_valid[:, None, None, None]
        elif tuple(history_valid.shape) != (batch_size, 1, 1, 1):
            raise RuntimeError(
                "AP-STCR history_valid must be [B] or [B,1,1,1], got "
                f"{list(history_valid.shape)}."
            )
        history_valid = history_valid.bool()
        fused_support = torch.where(
            history_valid,
            0.5 * (semantic_support + temporal_support),
            semantic_support,
        ).clamp(0.0, 1.0)
        support_deficiency = (1.0 - fused_support).clamp(0.0, 1.0)
        soft_inertia = (
            soft_deviation * support_deficiency
        ).clamp(0.0, 1.0)
        local_acceptance = (
            1.0 - self.soft_rejection_strength * soft_inertia
        ).clamp(self.min_local_acceptance, 1.0)
        if (
            float(local_acceptance.min().item())
            < self.min_local_acceptance - 1e-6
            or float(local_acceptance.max().item()) > 1.0 + 1e-6
        ):
            raise RuntimeError(
                "AP-STCR v3 local acceptance escaped its configured bounds."
            )
        return {
            "fused_support_37": fused_support.detach(),
            "support_deficiency_37": support_deficiency.detach(),
            "soft_inertia_37": soft_inertia.detach(),
            "local_acceptance_37": local_acceptance.detach(),
        }

    @torch.no_grad()
    def compute_non_compensatory_acceptance(
        self,
        semantic_contradiction,
        temporal_support,
        history_valid,
        soft_deviation,
        global_teacher_ratio,
    ):
        semantic_contradiction = _require_unit_map(
            "semantic_contradiction", semantic_contradiction
        ).detach().float()
        temporal_support = _require_unit_map(
            "temporal_support",
            temporal_support,
            shape=semantic_contradiction.shape,
        ).detach().float()
        soft_deviation = _require_unit_map(
            "soft_deviation",
            soft_deviation,
            shape=semantic_contradiction.shape,
        ).detach().float()
        if not torch.is_tensor(history_valid):
            raise TypeError("AP-STCR history_valid must be a tensor.")
        history_valid = history_valid.detach().to(
            device=semantic_contradiction.device
        )
        batch_size = int(semantic_contradiction.shape[0])
        if tuple(history_valid.shape) == (batch_size,):
            history_valid = history_valid[:, None, None, None]
        elif tuple(history_valid.shape) != (batch_size, 1, 1, 1):
            raise RuntimeError(
                "AP-STCR history_valid must be [B] or [B,1,1,1], got "
                f"{list(history_valid.shape)}."
            )
        history_valid = history_valid.bool()
        temporal_instability = torch.where(
            history_valid,
            (1.0 - temporal_support).clamp(0.0, 1.0),
            torch.zeros_like(temporal_support),
        )
        combined_negative_evidence = (
            1.0
            - (1.0 - semantic_contradiction)
            * (1.0 - temporal_instability)
        ).clamp(0.0, 1.0)
        alpha = float(global_teacher_ratio)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise RuntimeError(
                "AP-STCR global teacher ratio must be in [0,1], got "
                f"{alpha}."
            )
        transition_envelope = float(
            max(0.0, min(1.0, 4.0 * alpha * (1.0 - alpha)))
        )
        transition_penalty = (
            transition_envelope
            * soft_deviation
            * combined_negative_evidence
        ).clamp(0.0, 1.0)
        local_acceptance = (
            1.0
            - self.evidence_rejection_strength * transition_penalty
        ).clamp(self.min_local_acceptance, 1.0)
        if (
            float(local_acceptance.min().item())
            < self.min_local_acceptance - 1e-6
            or float(local_acceptance.max().item()) > 1.0 + 1e-6
        ):
            raise RuntimeError(
                "AP-STCR v4 local acceptance escaped its configured bounds."
            )
        return {
            "temporal_instability_37": temporal_instability.detach(),
            "combined_negative_evidence_37": (
                combined_negative_evidence.detach()
            ),
            "transition_envelope": transition_envelope,
            "transition_penalty_37": transition_penalty.detach(),
            "local_acceptance_37": local_acceptance.detach(),
        }

    @torch.no_grad()
    def compute_semantic_only_acceptance(
        self,
        semantic_contradiction,
        soft_deviation,
        global_teacher_ratio,
    ):
        semantic_contradiction = _require_unit_map(
            "semantic_contradiction", semantic_contradiction
        ).detach().float()
        soft_deviation = _require_unit_map(
            "soft_deviation",
            soft_deviation,
            shape=semantic_contradiction.shape,
        ).detach().float()
        alpha = float(global_teacher_ratio)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise RuntimeError(
                "AP-STCR global teacher ratio must be in [0,1], got "
                f"{alpha}."
            )
        combined_negative_evidence = semantic_contradiction
        transition_envelope = float(
            max(0.0, min(1.0, 4.0 * alpha * (1.0 - alpha)))
        )
        transition_penalty = (
            transition_envelope
            * soft_deviation
            * combined_negative_evidence
        ).clamp(0.0, 1.0)
        local_acceptance = (
            1.0
            - self.evidence_rejection_strength * transition_penalty
        ).clamp(self.min_local_acceptance, 1.0)
        if (
            float(local_acceptance.min().item())
            < self.min_local_acceptance - 1e-6
            or float(local_acceptance.max().item()) > 1.0 + 1e-6
        ):
            raise RuntimeError(
                "AP-STCR A1 local acceptance escaped its configured bounds."
            )
        return {
            "combined_negative_evidence_37": (
                combined_negative_evidence.detach()
            ),
            "transition_envelope": transition_envelope,
            "transition_penalty_37": transition_penalty.detach(),
            "local_acceptance_37": local_acceptance.detach(),
        }

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
        if (self.is_v3 or self.is_v4 or self.is_semantic_only) and not bool(
            ((teacher_binary_68 == 0.0) | (teacher_binary_68 == 1.0))
            .all()
            .item()
        ):
            raise RuntimeError(
                "AP-STCR "
                f"{'A1' if self.is_semantic_only else ('v4' if self.is_v4 else 'v3')} "
                "final teacher target "
                "must be binary."
            )
        alpha = float(global_teacher_ratio)
        if not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise RuntimeError(
                f"AP-STCR global teacher ratio must be in [0,1], got {alpha}."
            )
        local_acceptance_37 = _require_unit_map(
            "local_acceptance_37", local_acceptance_37
        ).detach().float()
        local_acceptance_68 = F.interpolate(
            local_acceptance_37,
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
            "effective_teacher_weight_37": (
                alpha * local_acceptance_37
            ).clamp(0.0, 1.0).detach(),
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
        if (
            self.is_v2
            or self.is_v3
            or self.is_v4
            or self.is_semantic_only
        ):
            teacher_binary_37 = strict_teacher_binary(
                teacher_soft_37,
                threshold=0.5,
            )
        else:
            teacher_binary_37 = F.interpolate(
                teacher_binary_68,
                size=(self.evidence_resolution, self.evidence_resolution),
                mode="nearest",
            )
        conflict = self.compute_source_conflict(
            fixed_pseudo_37,
            teacher_binary_37,
        )
        teacher_binary_37 = conflict["teacher_binary_37"]
        if self.is_v3 or self.is_v4 or self.is_semantic_only:
            semantic_support, soft_correction_37 = (
                self.compute_soft_semantic_support(
                    semantic["semantic_margin"],
                    teacher_soft_37,
                    fixed_pseudo_37,
                )
            )
        else:
            semantic_support, correction_37 = self.compute_semantic_support(
                semantic["semantic_margin"],
                teacher_binary_37,
                fixed_pseudo_37,
            )
            soft_correction_37 = None
        correction_37 = (
            teacher_binary_37.detach().float()
            - fixed_pseudo_37.detach().float()
        )
        temporal = None
        if not self.is_semantic_only:
            temporal = self.compute_temporal_support(
                teacher_soft_37,
                fixed_pseudo_37,
                sample_indices,
                datasets,
                stems,
            )
        if self.is_v2:
            acceptance = self.compute_conflict_only_acceptance(
                semantic_support,
                temporal["temporal_support"],
                conflict["source_conflict_37"],
            )
            local_acceptance = acceptance["local_acceptance_37"]
        elif self.is_v3:
            soft_deviation_37 = soft_correction_37.abs().clamp(0.0, 1.0)
            acceptance = self.compute_soft_disagreement_acceptance(
                semantic_support,
                temporal["temporal_support"],
                temporal["history_valid"],
                soft_deviation_37,
            )
            local_acceptance = acceptance["local_acceptance_37"]
        elif self.is_v4:
            soft_deviation_37 = soft_correction_37.abs().clamp(0.0, 1.0)
            semantic_contradiction_37 = (
                self.compute_semantic_contradiction(
                    semantic["semantic_margin"],
                    soft_correction_37,
                )
            )
            acceptance = self.compute_non_compensatory_acceptance(
                semantic_contradiction_37,
                temporal["temporal_support"],
                temporal["history_valid"],
                soft_deviation_37,
                global_teacher_ratio,
            )
            local_acceptance = acceptance["local_acceptance_37"]
        elif self.is_semantic_only:
            soft_deviation_37 = soft_correction_37.abs().clamp(0.0, 1.0)
            semantic_contradiction_37 = (
                self.compute_semantic_contradiction(
                    semantic["semantic_margin"],
                    soft_correction_37,
                )
            )
            acceptance = self.compute_semantic_only_acceptance(
                semantic_contradiction_37,
                soft_deviation_37,
                global_teacher_ratio,
            )
            local_acceptance = acceptance["local_acceptance_37"]
        else:
            local_acceptance = self.compute_local_acceptance(
                semantic_support,
                temporal["temporal_support"],
            )
            semantic_negative = (
                (0.5 - semantic_support.detach().float()) / 0.5
            ).clamp(0.0, 1.0)
            temporal_negative = (
                (0.5 - temporal["temporal_support"].detach().float()) / 0.5
            ).clamp(0.0, 1.0)
            acceptance = {
                "semantic_negative_37": semantic_negative.detach(),
                "temporal_negative_37": temporal_negative.detach(),
                "negative_evidence_37": (
                    0.5 * (semantic_negative + temporal_negative)
                ).detach(),
                "local_rejection_37": (
                    1.0 - local_acceptance
                ).clamp(0.0, 1.0).detach(),
                "local_acceptance_37": local_acceptance.detach(),
            }
        target = self.build_target(
            fixed_pseudo_68,
            teacher_binary_68,
            global_teacher_ratio,
            local_acceptance,
        )
        result = {
            "version": self.version,
            "conflict_only": self.conflict_only,
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
            "fixed_hard_37": conflict["fixed_hard_37"].detach(),
            "source_conflict_37": conflict[
                "source_conflict_37"
            ].detach(),
            "teacher_correction_37": correction_37.detach(),
            "teacher_correction_68": (
                teacher_binary_68 - fixed_pseudo_68
            ).detach(),
            "semantic_margin_37": semantic["semantic_margin"].detach(),
            "fg_anchor_mask_37": semantic["fg_anchor_mask"].detach(),
            "bg_anchor_mask_37": semantic["bg_anchor_mask"].detach(),
            "bg_anchor_source": list(semantic["bg_anchor_source"]),
            "semantic_support_37": semantic_support.detach(),
        }
        if not self.is_semantic_only:
            result.update(
                {
                    "temporal_support_37": temporal[
                        "temporal_support"
                    ].detach(),
                    "history_mean_37": temporal["history_mean"].detach(),
                    "history_count": temporal["history_count"].detach(),
                    "history_valid": temporal["history_valid"].detach(),
                }
            )
        result["local_acceptance_37"] = local_acceptance.detach()
        if self.is_v3:
            result.update(
                {
                    "soft_correction_37": soft_correction_37.detach(),
                    "soft_deviation_37": soft_deviation_37.detach(),
                    "fused_support_37": acceptance[
                        "fused_support_37"
                    ].detach(),
                    "support_deficiency_37": acceptance[
                        "support_deficiency_37"
                    ].detach(),
                    "soft_inertia_37": acceptance[
                        "soft_inertia_37"
                    ].detach(),
                }
            )
        elif self.is_v4:
            result.update(
                {
                    "soft_correction_37": soft_correction_37.detach(),
                    "soft_deviation_37": soft_deviation_37.detach(),
                    "semantic_contradiction_37": (
                        semantic_contradiction_37.detach()
                    ),
                    "temporal_instability_37": acceptance[
                        "temporal_instability_37"
                    ].detach(),
                    "combined_negative_evidence_37": acceptance[
                        "combined_negative_evidence_37"
                    ].detach(),
                    "transition_envelope": acceptance[
                        "transition_envelope"
                    ],
                    "transition_penalty_37": acceptance[
                        "transition_penalty_37"
                    ].detach(),
                }
            )
        elif self.is_semantic_only:
            result.update(
                {
                    "soft_correction_37": soft_correction_37.detach(),
                    "soft_deviation_37": soft_deviation_37.detach(),
                    "semantic_contradiction_37": (
                        semantic_contradiction_37.detach()
                    ),
                    "combined_negative_evidence_37": acceptance[
                        "combined_negative_evidence_37"
                    ].detach(),
                    "transition_envelope": acceptance[
                        "transition_envelope"
                    ],
                    "transition_penalty_37": acceptance[
                        "transition_penalty_37"
                    ].detach(),
                }
            )
        else:
            result.update(
                {
                    "semantic_negative_37": acceptance[
                        "semantic_negative_37"
                    ].detach(),
                    "temporal_negative_37": acceptance[
                        "temporal_negative_37"
                    ].detach(),
                    "negative_evidence_37": acceptance[
                        "negative_evidence_37"
                    ].detach(),
                    "local_rejection_37": acceptance[
                        "local_rejection_37"
                    ].detach(),
                }
            )
        result.update(target)
        if (
            self.is_v2
            or self.is_v3
            or self.is_v4
            or self.is_semantic_only
        ):
            version_label = (
                "A1"
                if self.is_semantic_only
                else ("v4" if self.is_v4 else ("v3" if self.is_v3 else "v2"))
            )
            if (
                float(result["effective_teacher_weight_68"].min().item())
                < -1e-6
                or float(
                    result["effective_teacher_weight_68"].max().item()
                )
                > 1.0 + 1e-6
            ):
                raise RuntimeError(
                    "AP-STCR "
                    f"{version_label} effective teacher weight is outside "
                    "[0,1]."
                )
            if (
                float(result["mixed_target_68"].min().item()) < -1e-6
                or float(result["mixed_target_68"].max().item())
                > 1.0 + 1e-6
            ):
                raise RuntimeError(
                    f"AP-STCR {version_label} mixed target is outside [0,1]."
                )
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
        if self.history_bank is None:
            return
        self.history_bank.update(
            sample_indices,
            datasets,
            stems,
            teacher_soft_37,
            epoch,
        )

    def clear_temporal_history(self):
        if self.history_bank is None:
            return
        self.history_bank.clear()

    def state_dict(self):
        return {
            "schema_version": "ap_stcr_runtime_v1",
            "manifest_hash": self.manifest_hash,
            "semantic_cache": self.semantic_cache.state_dict(),
            "history_bank": (
                None
                if self.history_bank is None
                else self.history_bank.state_dict()
            ),
        }

    def load_state_dict(self, state):
        if state.get("schema_version") != "ap_stcr_runtime_v1":
            raise RuntimeError("AP-STCR runtime schema mismatch.")
        if state.get("manifest_hash") != self.manifest_hash:
            raise RuntimeError("AP-STCR runtime manifest mismatch.")
        self.semantic_cache.load_state_dict(state["semantic_cache"])
        history_state = state.get("history_bank")
        if self.history_bank is None:
            if history_state is not None:
                raise RuntimeError(
                    "AP-STCR A1 runtime must not contain temporal history."
                )
            return
        if history_state is None:
            raise RuntimeError("AP-STCR runtime temporal history is missing.")
        self.history_bank.load_state_dict(history_state)


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
        "semantic_negative",
        "temporal_negative",
        "local_rejection",
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
        "local_acceptance_on_conflict",
        "local_acceptance_on_non_conflict",
        "effective_teacher_weight_on_conflict",
        "effective_teacher_weight_on_non_conflict",
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
        "source_conflict_pixels": 0,
        "evidence_pixel_count": 0,
        "pixel_count": 0,
        "bg_sources": {},
        "global_teacher_ratio_sum": 0.0,
        "batch_count": 0,
    }


def accumulate_ap_stcr_epoch(accumulator, result, student_prob_68):
    version = str(result.get("version", ""))
    is_v3 = (
        version == "ap_stcr_v3_soft_disagreement_bounded_continuation"
    )
    is_v4 = (
        version == "ap_stcr_v4_transition_envelope_non_compensatory"
    )
    is_semantic_only = (
        version == "ap_stcr_v4_semantic_only_ablation"
    )
    if is_v3:
        for name in (
            "soft_deviation",
            "fused_support",
            "support_deficiency",
            "soft_inertia",
            "effective_teacher_ratio",
            "effective_fixed_ratio",
        ):
            accumulator["moments"].setdefault(name, _new_moments())
        histogram_template = accumulator["histograms"][
            "local_acceptance"
        ]
        for name in ("soft_deviation", "fused_support"):
            accumulator["histograms"].setdefault(
                name, torch.zeros_like(histogram_template)
            )
    if is_v4:
        for name in (
            "soft_deviation",
            "semantic_contradiction",
            "temporal_instability",
            "combined_negative_evidence",
            "transition_penalty",
            "effective_teacher_ratio",
            "effective_fixed_ratio",
            "mixed_target_area",
            "semantic_contradiction_when_agree",
            "semantic_contradiction_when_disagree",
            "temporal_instability_when_agree",
            "temporal_instability_when_disagree",
            "combined_negative_evidence_when_agree",
            "combined_negative_evidence_when_disagree",
        ):
            accumulator["moments"].setdefault(name, _new_moments())
        histogram_template = accumulator["histograms"][
            "local_acceptance"
        ]
        for name in (
            "semantic_contradiction",
            "temporal_instability",
        ):
            accumulator["histograms"].setdefault(
                name, torch.zeros_like(histogram_template)
            )
        accumulator.setdefault("transition_envelope_sum", 0.0)
        accumulator.setdefault("transition_envelope_count", 0)
    if is_semantic_only:
        for name in (
            "temporal_support",
            "temporal_negative",
            "temporal_support_agree",
            "temporal_support_disagree",
            "semantic_negative",
            "local_rejection",
        ):
            accumulator["moments"].pop(name, None)
        accumulator["histograms"].pop("temporal_support", None)
        for name in (
            "soft_deviation",
            "semantic_contradiction",
            "combined_negative_evidence",
            "transition_penalty",
            "effective_teacher_ratio",
            "effective_fixed_ratio",
            "mixed_target_area",
            "semantic_contradiction_when_agree",
            "semantic_contradiction_when_disagree",
            "combined_negative_evidence_when_agree",
            "combined_negative_evidence_when_disagree",
        ):
            accumulator["moments"].setdefault(name, _new_moments())
        histogram_template = accumulator["histograms"][
            "local_acceptance"
        ]
        accumulator["histograms"].setdefault(
            "semantic_contradiction",
            torch.zeros_like(histogram_template),
        )
        accumulator["report_history_stats"] = False
        accumulator.setdefault("transition_envelope_sum", 0.0)
        accumulator.setdefault("transition_envelope_count", 0)
    batch_size = int(result["fixed_pseudo_68"].shape[0])
    fixed_binary = result["fixed_pseudo_68"] > 0.5
    teacher_binary = result["teacher_binary_68"] > 0.5
    disagree = fixed_binary != teacher_binary
    agree = ~disagree
    source_conflict = result["source_conflict_37"].detach().bool()
    source_non_conflict = ~source_conflict
    accepted_correction = (
        result["effective_teacher_weight_68"]
        * result["teacher_correction_68"]
    )
    values = {
        "semantic_margin": result["semantic_margin_37"],
        "semantic_support": result["semantic_support_37"],
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
    if not is_semantic_only:
        values["temporal_support"] = result["temporal_support_37"]
    if is_v3:
        values.update(
            {
                "soft_deviation": result["soft_deviation_37"],
                "fused_support": result["fused_support_37"],
                "support_deficiency": result[
                    "support_deficiency_37"
                ],
                "soft_inertia": result["soft_inertia_37"],
                "effective_teacher_ratio": result[
                    "effective_teacher_weight_37"
                ],
                "effective_fixed_ratio": (
                    1.0 - result["effective_teacher_weight_37"]
                ),
            }
        )
    elif is_v4:
        values.update(
            {
                "soft_deviation": result["soft_deviation_37"],
                "semantic_contradiction": result[
                    "semantic_contradiction_37"
                ],
                "temporal_instability": result[
                    "temporal_instability_37"
                ],
                "combined_negative_evidence": result[
                    "combined_negative_evidence_37"
                ],
                "transition_penalty": result["transition_penalty_37"],
                "effective_teacher_ratio": result[
                    "effective_teacher_weight_37"
                ],
                "effective_fixed_ratio": (
                    1.0 - result["effective_teacher_weight_37"]
                ),
                "mixed_target_area": result["mixed_target_68"],
            }
        )
    elif is_semantic_only:
        values.update(
            {
                "soft_deviation": result["soft_deviation_37"],
                "semantic_contradiction": result[
                    "semantic_contradiction_37"
                ],
                "combined_negative_evidence": result[
                    "combined_negative_evidence_37"
                ],
                "transition_penalty": result["transition_penalty_37"],
                "effective_teacher_ratio": result[
                    "effective_teacher_weight_37"
                ],
                "effective_fixed_ratio": (
                    1.0 - result["effective_teacher_weight_37"]
                ),
                "mixed_target_area": result["mixed_target_68"],
            }
        )
    else:
        values.update(
            {
                "semantic_negative": result["semantic_negative_37"],
                "temporal_negative": result["temporal_negative_37"],
                "local_rejection": result["local_rejection_37"],
            }
        )
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
        "effective_teacher_weight_agree": (
            result["effective_teacher_weight_68"],
            agree,
        ),
        "effective_teacher_weight_disagree": (
            result["effective_teacher_weight_68"],
            disagree,
        ),
        "local_acceptance_on_conflict": (
            result["local_acceptance_37"],
            source_conflict,
        ),
        "local_acceptance_on_non_conflict": (
            result["local_acceptance_37"],
            source_non_conflict,
        ),
        "effective_teacher_weight_on_conflict": (
            result["effective_teacher_weight_37"],
            source_conflict,
        ),
        "effective_teacher_weight_on_non_conflict": (
            result["effective_teacher_weight_37"],
            source_non_conflict,
        ),
    }
    if not is_semantic_only:
        temporal_support_68 = F.interpolate(
            result["temporal_support_37"],
            size=(68, 68),
            mode="bilinear",
            align_corners=False,
        )
        conditional.update(
            {
                "temporal_support_agree": (
                    temporal_support_68,
                    agree,
                ),
                "temporal_support_disagree": (
                    temporal_support_68,
                    disagree,
                ),
            }
        )
    if is_v4:
        v4_conditionals = {
            "semantic_contradiction_when_agree": result[
                "semantic_contradiction_37"
            ],
            "semantic_contradiction_when_disagree": result[
                "semantic_contradiction_37"
            ],
            "temporal_instability_when_agree": result[
                "temporal_instability_37"
            ],
            "temporal_instability_when_disagree": result[
                "temporal_instability_37"
            ],
            "combined_negative_evidence_when_agree": result[
                "combined_negative_evidence_37"
            ],
            "combined_negative_evidence_when_disagree": result[
                "combined_negative_evidence_37"
            ],
        }
        for name, value in v4_conditionals.items():
            value_68 = F.interpolate(
                value,
                size=(68, 68),
                mode="bilinear",
                align_corners=False,
            )
            conditional[name] = (
                value_68,
                disagree if name.endswith("_disagree") else agree,
            )
    elif is_semantic_only:
        semantic_only_conditionals = {
            "semantic_contradiction_when_agree": result[
                "semantic_contradiction_37"
            ],
            "semantic_contradiction_when_disagree": result[
                "semantic_contradiction_37"
            ],
            "combined_negative_evidence_when_agree": result[
                "combined_negative_evidence_37"
            ],
            "combined_negative_evidence_when_disagree": result[
                "combined_negative_evidence_37"
            ],
        }
        for name, value in semantic_only_conditionals.items():
            value_68 = F.interpolate(
                value,
                size=(68, 68),
                mode="bilinear",
                align_corners=False,
            )
            conditional[name] = (
                value_68,
                disagree if name.endswith("_disagree") else agree,
            )
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
    if not is_semantic_only:
        accumulator["history_valid_images"] += int(
            result["history_valid"].sum().item()
        )
    accumulator["disagreement_pixels"] += int(disagree.sum().item())
    accumulator["source_conflict_pixels"] += int(
        source_conflict.sum().item()
    )
    accumulator["evidence_pixel_count"] += int(source_conflict.numel())
    accumulator["pixel_count"] += int(disagree.numel())
    for source in result["bg_anchor_source"]:
        accumulator["bg_sources"][source] = (
            int(accumulator["bg_sources"].get(source, 0)) + 1
        )
    accumulator["global_teacher_ratio_sum"] += float(
        result["global_teacher_ratio"]
    )
    if is_v4 or is_semantic_only:
        accumulator["transition_envelope_sum"] += float(
            result["transition_envelope"]
        )
        accumulator["transition_envelope_count"] += 1
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
    evidence_pixel_count = max(
        1, int(accumulator["evidence_pixel_count"])
    )
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
            "source_conflict_ratio": float(
                accumulator["source_conflict_pixels"]
            )
            / evidence_pixel_count,
            "global_teacher_ratio": float(
                accumulator["global_teacher_ratio_sum"]
            )
            / batch_count,
        }
    )
    transition_envelope_count = int(
        accumulator.get("transition_envelope_count", 0)
    )
    if transition_envelope_count > 0:
        result["transition_envelope"] = float(
            accumulator["transition_envelope_sum"]
        ) / float(transition_envelope_count)
    if not bool(accumulator.get("report_history_stats", True)):
        result.pop("history_valid_ratio", None)
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
    ap_stcr_version = str(
        result.get("version", "ap_stcr_v1_full_pixel_37_to_68")
    )
    is_v2 = bool(result.get("conflict_only", False))
    is_v3 = (
        ap_stcr_version
        == "ap_stcr_v3_soft_disagreement_bounded_continuation"
    )
    is_v4 = (
        ap_stcr_version
        == "ap_stcr_v4_transition_envelope_non_compensatory"
    )
    is_semantic_only = (
        ap_stcr_version == "ap_stcr_v4_semantic_only_ablation"
    )
    payload = {
        "schema_version": (
            "ap_stcr_diagnostic_v4_semantic_only"
            if is_semantic_only
            else (
                "ap_stcr_diagnostic_v4"
                if is_v4
                else (
                    "ap_stcr_diagnostic_v3"
                    if is_v3
                    else (
                        "ap_stcr_diagnostic_v2"
                        if is_v2
                        else "ap_stcr_diagnostic_v1"
                    )
                )
            )
        ),
        "ap_stcr_version": ap_stcr_version,
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
    }
    if not is_semantic_only:
        payload.update(
            {
                "teacher_correction": result[
                    "teacher_correction_68"
                ][index].cpu(),
                "semantic_support": resize_bilinear(
                    result["semantic_support_37"]
                ),
                "temporal_support": resize_bilinear(
                    result["temporal_support_37"]
                ),
            }
        )
    payload.update(
        {
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
    )
    if is_v2 or is_v3 or is_v4:
        payload["source_conflict"] = resize_nearest(
            result["source_conflict_37"]
        )
    if is_v3:
        payload.update(
            {
                "teacher_soft": result["teacher_soft_68"][index].cpu(),
                "soft_deviation": resize_bilinear(
                    result["soft_deviation_37"]
                ),
                "fused_support": resize_bilinear(
                    result["fused_support_37"]
                ),
                "support_deficiency": resize_bilinear(
                    result["support_deficiency_37"]
                ),
                "soft_inertia": resize_bilinear(
                    result["soft_inertia_37"]
                ),
            }
        )
    elif is_v4:
        payload.update(
            {
                "teacher_soft": result["teacher_soft_68"][index].cpu(),
                "soft_deviation": resize_bilinear(
                    result["soft_deviation_37"]
                ),
                "semantic_contradiction": resize_bilinear(
                    result["semantic_contradiction_37"]
                ),
                "temporal_instability": resize_bilinear(
                    result["temporal_instability_37"]
                ),
                "combined_negative_evidence": resize_bilinear(
                    result["combined_negative_evidence_37"]
                ),
                "transition_penalty": resize_bilinear(
                    result["transition_penalty_37"]
                ),
                "global_teacher_ratio": float(
                    result["global_teacher_ratio"]
                ),
                "transition_envelope": float(
                    result["transition_envelope"]
                ),
            }
        )
    elif is_semantic_only:
        payload.update(
            {
                "teacher_soft": result["teacher_soft_68"][index].cpu(),
                "soft_deviation": resize_bilinear(
                    result["soft_deviation_37"]
                ),
                "semantic_contradiction": resize_bilinear(
                    result["semantic_contradiction_37"]
                ),
                "transition_penalty": resize_bilinear(
                    result["transition_penalty_37"]
                ),
                "global_teacher_ratio": float(
                    result["global_teacher_ratio"]
                ),
                "transition_envelope": float(
                    result["transition_envelope"]
                ),
            }
        )
    return payload


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
    ]
    schema_version = payload.get("schema_version")
    is_v3 = schema_version == "ap_stcr_diagnostic_v3"
    is_v4 = schema_version == "ap_stcr_diagnostic_v4"
    is_semantic_only = (
        schema_version == "ap_stcr_diagnostic_v4_semantic_only"
    )
    if is_semantic_only:
        panels.extend(
            [
                ("EMA teacher soft probability", payload["teacher_soft"], "gray", 0.0, 1.0),
                ("EMA teacher binary", payload["teacher_binary"], "gray", 0.0, 1.0),
                ("Soft teacher deviation", payload["soft_deviation"], "magma", 0.0, 1.0),
                ("Semantic contradiction", payload["semantic_contradiction"], "magma", 0.0, 1.0),
                ("Transition penalty", payload["transition_penalty"], "magma", 0.0, 1.0),
                ("Local acceptance", payload["local_acceptance"], "viridis", 0.0, 1.0),
                ("Effective teacher weight", payload["effective_teacher_weight"], "viridis", 0.0, 1.0),
                ("Mixed target", payload["mixed_target"], "gray", 0.0, 1.0),
                ("Student prediction", payload["student_prediction"], "gray", 0.0, 1.0),
            ]
        )
    elif is_v4:
        panels.extend(
            [
                ("EMA teacher soft probability", payload["teacher_soft"], "gray", 0.0, 1.0),
                ("EMA teacher binary", payload["teacher_binary"], "gray", 0.0, 1.0),
                ("Soft teacher deviation", payload["soft_deviation"], "magma", 0.0, 1.0),
                ("Semantic contradiction", payload["semantic_contradiction"], "magma", 0.0, 1.0),
                ("Temporal instability", payload["temporal_instability"], "magma", 0.0, 1.0),
                ("Combined negative evidence", payload["combined_negative_evidence"], "magma", 0.0, 1.0),
                ("Transition penalty", payload["transition_penalty"], "magma", 0.0, 1.0),
                ("Local acceptance", payload["local_acceptance"], "viridis", 0.0, 1.0),
                ("Effective teacher weight", payload["effective_teacher_weight"], "viridis", 0.0, 1.0),
                ("Mixed target", payload["mixed_target"], "gray", 0.0, 1.0),
                ("Student prediction", payload["student_prediction"], "gray", 0.0, 1.0),
            ]
        )
    else:
        if is_v3:
            panels.append(
                ("EMA teacher soft probability", payload["teacher_soft"], "gray", 0.0, 1.0)
            )
        panels.extend(
            [
                ("EMA teacher binary", payload["teacher_binary"], "gray", 0.0, 1.0),
                ("Teacher correction", payload["teacher_correction"], "coolwarm", -1.0, 1.0),
            ]
        )
        if "source_conflict" in payload:
            panels.append(
                (
                    (
                        "Source conflict (diagnostic only)"
                        if is_v3
                        else "Source conflict"
                    ),
                    payload["source_conflict"],
                    "gray",
                    0.0,
                    1.0,
                )
            )
        panels.extend(
            [
                ("Semantic support", payload["semantic_support"], "viridis", 0.0, 1.0),
                ("Temporal support", payload["temporal_support"], "viridis", 0.0, 1.0),
                ("Local acceptance", payload["local_acceptance"], "viridis", 0.0, 1.0),
                ("Effective teacher weight", payload["effective_teacher_weight"], "viridis", 0.0, 1.0),
                ("Mixed target", payload["mixed_target"], "gray", 0.0, 1.0),
                ("Student prediction", payload["student_prediction"], "gray", 0.0, 1.0),
            ]
        )
        if is_v3:
            panels.extend(
                [
                    ("Soft teacher deviation", payload["soft_deviation"], "magma", 0.0, 1.0),
                    ("Fused support", payload["fused_support"], "viridis", 0.0, 1.0),
                    ("Support deficiency", payload["support_deficiency"], "magma", 0.0, 1.0),
                    ("Soft inertia", payload["soft_inertia"], "magma", 0.0, 1.0),
                ]
            )
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
    title = (
        f"{payload['dataset']}/{payload['stem']} | "
        f"epoch={int(payload['epoch']):03d}"
    )
    if is_v4:
        title += (
            f" | alpha={float(payload['global_teacher_ratio']):.4f}"
            " | transition_envelope="
            f"{float(payload['transition_envelope']):.4f}"
        )
    elif is_semantic_only:
        title = "AP-STCR A1: Semantic Only | " + title
        title += (
            f" | alpha={float(payload['global_teacher_ratio']):.4f}"
            " | transition_envelope="
            f"{float(payload['transition_envelope']):.4f}"
        )
    title += f" | bg={payload['bg_anchor_source']}"
    figure.suptitle(title)
    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(figure)
