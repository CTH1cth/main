import hashlib
import json

import torch
import torch.nn.functional as F

from common.bitc import validate_bitc_config
from common.ecst_clean import validate_ecst_clean_config
from common.ecst_minimal import validate_ecst_minimal_config
from common.ectp import validate_ectp_config
from common.eaogp import validate_eaogp_config
from common.r1hard_linear_pure_student import (
    is_r1hard_dagp_ndr_pure_student_config,
    is_r1hard_linear_pure_student_config,
)
from common.utils import config_to_dict


EXPLICIT_TEACHER_ROUTING_MODES = {
    "ecst",
    "minimal_ecst",
    "clean_ecst",
    "none",
    "bitc_v1",
    "ectp",
    "eaogp",
}
LEGACY_TEACHER_ROUTING_MODE = "legacy"

NO_ECST_FORBIDDEN_FLAGS = (
    "USE_RAST",
    "USE_ESA_ASYM",
    "ESA_POST_RESET_ENABLE",
    "USE_ESA_BER",
    "USE_TEPR_LITE",
    "USE_SOURCE_ARBITER",
    "USE_TADR_ROUTER",
    "USE_TCE",
    "USE_LCEG",
    "USE_HBNS_LITE",
    "USE_EPR_POS",
    "USE_DABE_OEM",
    "USE_DABE_PU_GROUP_BALANCED_STATIC",
    "USE_GKD_LITE",
    "USE_BITC",
    "USE_ECTP",
    "USE_EAOGP",
    "USE_ECST_MINIMAL",
    "USE_ECST_CLEAN",
)


def has_explicit_teacher_routing_mode(cfg):
    return hasattr(cfg, "TEACHER_ROUTING_MODE")


def get_teacher_routing_mode(cfg):
    """Return an explicit route mode while preserving old config dispatch."""
    if not has_explicit_teacher_routing_mode(cfg):
        return LEGACY_TEACHER_ROUTING_MODE
    mode = str(getattr(cfg, "TEACHER_ROUTING_MODE")).strip().lower()
    if mode not in EXPLICIT_TEACHER_ROUTING_MODES:
        raise RuntimeError(
            f"Unsupported TEACHER_ROUTING_MODE={mode!r}, "
            f"allowed={sorted(EXPLICIT_TEACHER_ROUTING_MODES)}"
        )
    return mode


def validate_teacher_routing_config(cfg):
    mode = get_teacher_routing_mode(cfg)
    use_ectp = bool(getattr(cfg, "USE_ECTP", False))
    use_eaogp = bool(getattr(cfg, "USE_EAOGP", False))
    if mode == LEGACY_TEACHER_ROUTING_MODE:
        if use_ectp or use_eaogp:
            raise RuntimeError(
                "USE_ECTP/USE_EAOGP requires an explicit matching "
                "TEACHER_ROUTING_MODE."
            )
        return mode

    use_ecst = bool(getattr(cfg, "USE_ECST", False))
    use_minimal = bool(getattr(cfg, "USE_ECST_MINIMAL", False))
    use_clean_ecst = bool(getattr(cfg, "USE_ECST_CLEAN", False))
    use_bitc = bool(getattr(cfg, "USE_BITC", False))
    if mode == "ectp" and not use_ectp:
        raise RuntimeError("TEACHER_ROUTING_MODE='ectp' requires USE_ECTP=True")
    if mode != "ectp" and use_ectp:
        raise RuntimeError("USE_ECTP=True requires TEACHER_ROUTING_MODE='ectp'")
    if mode == "eaogp" and not use_eaogp:
        raise RuntimeError("TEACHER_ROUTING_MODE='eaogp' requires USE_EAOGP=True")
    if mode != "eaogp" and use_eaogp:
        raise RuntimeError("USE_EAOGP=True requires TEACHER_ROUTING_MODE='eaogp'")
    if mode == "ecst" and not use_ecst:
        raise RuntimeError("TEACHER_ROUTING_MODE='ecst' requires USE_ECST=True")
    if mode == "none" and use_ecst:
        raise RuntimeError("TEACHER_ROUTING_MODE='none' requires USE_ECST=False")
    if mode == "minimal_ecst":
        if use_ecst or not use_minimal:
            raise RuntimeError(
                "TEACHER_ROUTING_MODE='minimal_ecst' requires "
                "USE_ECST=False and USE_ECST_MINIMAL=True"
            )
        validate_ecst_minimal_config(cfg)
    elif use_minimal:
        raise RuntimeError(
            "USE_ECST_MINIMAL=True requires TEACHER_ROUTING_MODE='minimal_ecst'"
        )
    if mode == "clean_ecst":
        if use_ecst or use_minimal or not use_clean_ecst:
            raise RuntimeError(
                "TEACHER_ROUTING_MODE='clean_ecst' requires USE_ECST=False, "
                "USE_ECST_MINIMAL=False and USE_ECST_CLEAN=True"
            )
        validate_ecst_clean_config(cfg)
    elif use_clean_ecst:
        raise RuntimeError(
            "USE_ECST_CLEAN=True requires TEACHER_ROUTING_MODE='clean_ecst'"
        )
    if mode == "bitc_v1" and not use_bitc:
        raise RuntimeError("TEACHER_ROUTING_MODE='bitc_v1' requires USE_BITC=True")
    if mode != "bitc_v1" and use_bitc:
        raise RuntimeError("USE_BITC=True requires TEACHER_ROUTING_MODE='bitc_v1'")
    if mode == "bitc_v1":
        if use_ecst:
            raise RuntimeError("BITC-v1 and ECST cannot be enabled together")
        enabled = [
            name
            for name in NO_ECST_FORBIDDEN_FLAGS
            if name != "USE_BITC" and bool(getattr(cfg, name, False))
        ]
        if enabled:
            raise RuntimeError(
                f"BITC-v1 cannot enable another teacher router: {enabled}"
            )
        validate_bitc_config(cfg)

    if mode == "ectp":
        conflicting_flags = []
        if use_ecst:
            conflicting_flags.append("USE_ECST")
        conflicting_flags.extend(
            name
            for name in NO_ECST_FORBIDDEN_FLAGS
            if name != "USE_ECTP" and bool(getattr(cfg, name, False))
        )
        for name in ("USE_PSSF", "USE_AP_STCR", "USE_CVSA"):
            if bool(getattr(cfg, name, False)):
                conflicting_flags.append(name)
        gkd_mode = str(getattr(cfg, "GKD_MODE", "off")).strip().lower()
        if gkd_mode != "off":
            conflicting_flags.append(f"GKD_MODE={gkd_mode}")
        if conflicting_flags:
            raise RuntimeError(
                "ECTP cannot enable another teacher router: "
                f"{conflicting_flags}"
            )
        validate_ectp_config(cfg)

    if mode == "eaogp":
        conflicting_flags = []
        if use_ecst:
            conflicting_flags.append("USE_ECST")
        conflicting_flags.extend(
            name
            for name in NO_ECST_FORBIDDEN_FLAGS
            if name != "USE_EAOGP" and bool(getattr(cfg, name, False))
        )
        for name in ("USE_PSSF", "USE_AP_STCR", "USE_CVSA"):
            if bool(getattr(cfg, name, False)):
                conflicting_flags.append(name)
        gkd_mode = str(getattr(cfg, "GKD_MODE", "off")).strip().lower()
        if gkd_mode != "off":
            conflicting_flags.append(f"GKD_MODE={gkd_mode}")
        if conflicting_flags:
            raise RuntimeError(
                "EAOGP cannot enable another teacher router: "
                f"{conflicting_flags}"
            )
        validate_eaogp_config(cfg)

    if mode == "none":
        enabled = [
            name
            for name in NO_ECST_FORBIDDEN_FLAGS
            if bool(getattr(cfg, name, False))
        ]
        gkd_mode = str(getattr(cfg, "GKD_MODE", "off")).strip().lower()
        if gkd_mode != "off":
            enabled.append(f"GKD_MODE={gkd_mode}")
        if enabled:
            raise RuntimeError(
                "No-ECST experiment cannot enable other teacher routers: "
                f"{enabled}"
            )

        use_pssf = bool(getattr(cfg, "USE_PSSF", False))
        supervision_mode = str(
            getattr(cfg, "SUPERVISION_MODE", "")
        ).strip().lower()
        pssf_modes = {"pssf_state", "ppse_v2_state"}
        if use_pssf or supervision_mode in pssf_modes:
            if not use_pssf or supervision_mode not in pssf_modes:
                raise RuntimeError(
                    "PSSF teacher-routing bypass requires USE_PSSF=True and "
                    "a supported PSSF/PPSE supervision mode."
                )
            teacher_fusion_mode = str(
                getattr(cfg, "TEACHER_FUSION_MODE", "")
            ).strip().lower()
            if teacher_fusion_mode != supervision_mode:
                raise RuntimeError(
                    "PSSF teacher-routing bypass requires matching "
                    "SUPERVISION_MODE and TEACHER_FUSION_MODE, got "
                    f"{supervision_mode!r}/{teacher_fusion_mode!r}."
                )
            return mode

        if bool(getattr(cfg, "USE_DABE_CLEAN", False)):
            dabev2hard_static_only = bool(
                getattr(cfg, "DABEV2HARD_STATIC_ONLY", False)
            )
            r1hard_pure_student = (
                is_r1hard_linear_pure_student_config(cfg)
            )
            r1hard_dagp_ndr = is_r1hard_dagp_ndr_pure_student_config(cfg)
            r1hard_linear_pure_student = (
                r1hard_pure_student and not r1hard_dagp_ndr
            )
            expected_clean = {
                "STATIC_WEIGHT_MODE": "ones",
                "DABE_CLEAN_STATIC_WEIGHT_MODE": "ones",
                "P_INIT_MODE": "dabe_clean_v1_desplsched",
                "TEACHER_FUSION_MODE": "dabe_clean_despl_sched",
                "TEACHER_TARGET_MODE": "binary",
            }
            mismatched_clean = {
                name: getattr(cfg, name, None)
                for name, value in expected_clean.items()
                if str(getattr(cfg, name, "")).lower() != value
            }
            if mismatched_clean:
                raise RuntimeError(
                    "DABE-Clean no-ECST protocol mismatch: "
                    f"{mismatched_clean}; expected={expected_clean}"
                )
            required_clean = (
                (
                    "USE_DABE_CLEAN",
                    "USE_DABE_CLEAN_DESPL_SCHEDULE",
                    "USE_DABE_PU_STATIC_LOSS",
                )
                if r1hard_linear_pure_student
                else (
                    "USE_DABE_CLEAN",
                    "USE_DABE_CLEAN_DESPL_SCHEDULE",
                    "USE_DAGP_SAFE_HEAD",
                    "USE_NDR_BRANCH",
                    "USE_NDR_COARSE_AUX",
                )
            )
            missing_clean = [
                name
                for name in required_clean
                if not bool(getattr(cfg, name, False))
            ]
            if missing_clean:
                raise RuntimeError(
                    "DABE-Clean no-ECST requires flags: "
                    f"{missing_clean}"
                )
            if r1hard_linear_pure_student:
                if str(getattr(cfg, "HEAD_TYPE", "")).lower() != "simple":
                    raise RuntimeError(
                        "Hard-R1 pure-Student requires HEAD_TYPE='simple'."
                    )
                forbidden_linear = [
                    name
                    for name in ("USE_DAGP_SAFE_HEAD", "USE_NDR_BRANCH")
                    if bool(getattr(cfg, name, False))
                ]
                if forbidden_linear:
                    raise RuntimeError(
                        "Hard-R1 pure-Student linear head forbids flags: "
                        f"{forbidden_linear}"
                    )
            if dabev2hard_static_only:
                if bool(getattr(cfg, "USE_TEACHER_BINARY_FULL_LOSS", False)) or bool(
                    getattr(cfg, "USE_TEACHER_SOFT_FULL_LOSS", False)
                ):
                    raise RuntimeError(
                        "DABE-v2-hard static-only forbids all Teacher loss flags."
                    )
            elif not bool(getattr(cfg, "USE_TEACHER_BINARY_FULL_LOSS", False)):
                raise RuntimeError(
                    "DABE-Clean no-ECST requires USE_TEACHER_BINARY_FULL_LOSS=True."
                )
            return mode

        expected = {
            "STATIC_WEIGHT_MODE": "ones",
            "DABE_PU_VERSION": "pu_v11",
            "P_INIT_MODE": "dabe_pu_v11_desplsched",
            "TEACHER_FUSION_MODE": "dabe_pu_despl_sched",
            "TEACHER_TARGET_MODE": "binary",
            "DABE_PU_STATIC_TARGET_MODE": "soft",
        }
        mismatched = {
            name: getattr(cfg, name, None)
            for name, value in expected.items()
            if str(getattr(cfg, name, "")).lower() != value
        }
        if mismatched:
            raise RuntimeError(
                "No-ECST single-variable protocol mismatch: "
                f"{mismatched}; expected={expected}"
            )
        required_true = (
            "USE_DABE_PU",
            "USE_DABE_PU_DESPL_SCHEDULE",
            "USE_DABE_PU_STATIC_LOSS",
            "USE_TEACHER_BINARY_FULL_LOSS",
            "USE_DAGP_SAFE_HEAD",
            "USE_NDR_BRANCH",
            "USE_NDR_COARSE_AUX",
        )
        missing = [
            name for name in required_true if not bool(getattr(cfg, name, False))
        ]
        if missing:
            raise RuntimeError(
                f"No-ECST single-variable protocol requires flags: {missing}"
            )
        if bool(getattr(cfg, "USE_TEACHER_SOFT_FULL_LOSS", False)):
            raise RuntimeError("No-ECST requires binary, not soft, teacher targets")
    return mode


def teacher_routing_uses_ecst(cfg):
    mode = get_teacher_routing_mode(cfg)
    if mode in {"ecst", "minimal_ecst", "clean_ecst"}:
        return True
    if mode in {"none", "bitc_v1", "ectp", "eaogp"}:
        return False
    return bool(getattr(cfg, "USE_ECST", False))


def ectp_teacher_bce_with_logits(logits, target, teacher_map):
    """Apply ECTP's projected target with an unweighted mean Teacher BCE."""
    if not torch.is_tensor(logits):
        raise TypeError("ECTP logits must be a tensor")
    if not torch.is_tensor(target):
        raise TypeError("ECTP target must be a tensor")
    if not torch.is_tensor(teacher_map):
        raise TypeError("ECTP teacher_map must be a tensor")
    if logits.ndim != 4 or int(logits.shape[1]) != 1:
        raise RuntimeError(
            f"ECTP logits must be [B,1,H,W], got {list(logits.shape)}"
        )
    if tuple(target.shape) != tuple(logits.shape) or tuple(
        teacher_map.shape
    ) != tuple(logits.shape):
        raise RuntimeError(
            "ECTP logits/target/map shape mismatch: "
            f"{list(logits.shape)}/{list(target.shape)}/"
            f"{list(teacher_map.shape)}"
        )
    if target.requires_grad:
        raise RuntimeError("ECTP projected Teacher target must be detached")
    if not bool(torch.isfinite(target).all().item()):
        raise RuntimeError("ECTP projected Teacher target contains NaN/Inf")
    target_min = float(target.min().item())
    target_max = float(target.max().item())
    if target_min < -1e-6 or target_max > 1.0 + 1e-6:
        raise RuntimeError(
            "ECTP projected Teacher target must be in [0,1], got "
            f"{target_min}/{target_max}"
        )
    expected_map = torch.ones_like(
        teacher_map,
        dtype=teacher_map.dtype,
        device=teacher_map.device,
    )
    if teacher_map.requires_grad or not torch.equal(teacher_map, expected_map):
        raise RuntimeError(
            "TEACHER_ROUTING_MODE='ectp' requires an exact detached all-one map"
        )
    return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")


def eaogp_teacher_bce_with_logits(logits, target, teacher_map):
    """Apply the EAOGP soft target with an exact identity pixel map."""

    if not torch.is_tensor(logits) or logits.ndim != 4 or int(logits.shape[1]) != 1:
        raise RuntimeError("EAOGP logits must be a [B,1,H,W] tensor.")
    if not torch.is_tensor(target) or tuple(target.shape) != tuple(logits.shape):
        raise RuntimeError("EAOGP logits/target shapes differ.")
    if not torch.is_tensor(teacher_map) or tuple(teacher_map.shape) != tuple(logits.shape):
        raise RuntimeError("EAOGP logits/teacher-map shapes differ.")
    if target.requires_grad or not bool(torch.isfinite(target).all().item()):
        raise RuntimeError("EAOGP effective Teacher target must be finite and detached.")
    target_min = float(target.min().item())
    target_max = float(target.max().item())
    if target_min < -1e-6 or target_max > 1.0 + 1e-6:
        raise RuntimeError(
            "EAOGP effective Teacher target must be in [0,1], got "
            f"{target_min}/{target_max}."
        )
    if teacher_map.requires_grad or not torch.equal(
        teacher_map, torch.ones_like(teacher_map)
    ):
        raise RuntimeError(
            "TEACHER_ROUTING_MODE='eaogp' requires an exact detached all-one map."
        )
    return F.binary_cross_entropy_with_logits(logits, target, reduction="mean")


def build_identity_teacher_route(teacher_prob):
    if not torch.is_tensor(teacher_prob):
        raise TypeError("teacher_prob must be a tensor")
    if teacher_prob.ndim != 4 or int(teacher_prob.shape[1]) != 1:
        raise RuntimeError(
            "teacher_prob must be [B,1,H,W], "
            f"got {list(teacher_prob.shape)}"
        )
    if not bool(torch.isfinite(teacher_prob).all().item()):
        raise RuntimeError("teacher_prob contains NaN/Inf")
    route_map = torch.ones_like(
        teacher_prob,
        dtype=torch.float32,
        device=teacher_prob.device,
    ).detach()
    if route_map.requires_grad or not torch.equal(
        route_map, torch.ones_like(route_map)
    ):
        raise RuntimeError("Identity teacher route must be detached and exactly one")
    return route_map, {
        "routing_mode": "none",
        "teacher_map_min": 1.0,
        "teacher_map_mean": 1.0,
        "teacher_map_max": 1.0,
        "memory_active": False,
    }


def teacher_routing_protocol_fingerprint(cfg):
    resolved = config_to_dict(cfg)
    for field in ("EXP_NAME", "USE_ECST", "TEACHER_ROUTING_MODE"):
        resolved.pop(field, None)
    payload = json.dumps(
        resolved,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def new_teacher_routing_accumulator():
    return {
        "batches": 0,
        "pixels": 0,
        "map_sum": 0.0,
        "map_min": None,
        "map_max": None,
        "memory_active_batches": 0,
    }


def accumulate_teacher_routing(accumulator, route_map, memory_active):
    if accumulator is None:
        return
    if route_map.ndim != 4 or int(route_map.shape[1]) != 1:
        raise RuntimeError(
            f"Teacher route map must be [B,1,H,W], got {list(route_map.shape)}"
        )
    if route_map.requires_grad:
        raise RuntimeError("Teacher route map must be detached")
    if not bool(torch.isfinite(route_map).all().item()):
        raise RuntimeError("Teacher route map contains NaN/Inf")
    value_min = float(route_map.min().item())
    value_max = float(route_map.max().item())
    accumulator["batches"] += 1
    accumulator["pixels"] += int(route_map.numel())
    accumulator["map_sum"] += float(route_map.double().sum().item())
    accumulator["map_min"] = (
        value_min
        if accumulator["map_min"] is None
        else min(float(accumulator["map_min"]), value_min)
    )
    accumulator["map_max"] = (
        value_max
        if accumulator["map_max"] is None
        else max(float(accumulator["map_max"]), value_max)
    )
    accumulator["memory_active_batches"] += int(bool(memory_active))


def finalize_teacher_routing(accumulator):
    if accumulator is None or int(accumulator["batches"]) <= 0:
        raise RuntimeError("Teacher routing accumulator is empty")
    pixels = int(accumulator["pixels"])
    if pixels <= 0:
        raise RuntimeError("Teacher routing accumulator has no pixels")
    batches = int(accumulator["batches"])
    return {
        "map_min": float(accumulator["map_min"]),
        "map_mean": float(accumulator["map_sum"]) / float(pixels),
        "map_max": float(accumulator["map_max"]),
        "memory_active": bool(accumulator["memory_active_batches"] > 0),
        "memory_active_ratio": float(accumulator["memory_active_batches"])
        / float(batches),
    }
