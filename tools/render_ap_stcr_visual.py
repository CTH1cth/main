import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
WORKDIR_ROOT = (PROJECT_ROOT / "workdir").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.ap_stcr import render_ap_stcr_diagnostic  # noqa: E402
from common.utils import find_gt_path, load_config, torch_load  # noqa: E402


def _resolve_workdir_path(value, name, must_exist):
    path = Path(value).expanduser().resolve()
    try:
        path.relative_to(WORKDIR_ROOT)
    except ValueError as error:
        raise RuntimeError(
            f"{name} must be inside {WORKDIR_ROOT}, got {path}."
        ) from error
    if must_exist and not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return path


def _diagnostic_paths(input_path):
    if input_path.is_file():
        return [input_path]
    paths = sorted(input_path.rglob("*.pt"))
    if not paths:
        raise RuntimeError(
            f"No AP-STCR diagnostic .pt files found under {input_path}."
        )
    return paths


def _load_gt_68(cfg, payload):
    dataset = str(payload["dataset"])
    stem = str(payload["stem"])
    gt_path = find_gt_path(cfg.DATA_ROOT, dataset, stem)
    with Image.open(gt_path) as image:
        gt = image.convert("L").resize((68, 68), resample=Image.NEAREST)
        array = np.asarray(gt, dtype=np.float32) / 255.0
    return torch.from_numpy((array > 0.5).astype(np.float32)).unsqueeze(0)


def _validate_payload(payload, path):
    if not isinstance(payload, dict):
        raise TypeError(f"AP-STCR diagnostic must be a dict: {path}")
    schema_version = payload.get("schema_version")
    if schema_version not in {
        "ap_stcr_diagnostic_v1",
        "ap_stcr_diagnostic_v2",
        "ap_stcr_diagnostic_v3",
        "ap_stcr_diagnostic_v4",
        "ap_stcr_diagnostic_v4_semantic_only",
    }:
        raise RuntimeError(f"AP-STCR diagnostic schema mismatch: {path}")
    if schema_version == "ap_stcr_diagnostic_v4_semantic_only":
        required = {
            "epoch",
            "dataset",
            "stem",
            "bg_anchor_source",
            "rgb",
            "fixed_pseudo",
            "fg_anchor_mask",
            "bg_anchor_mask",
            "semantic_margin",
            "teacher_soft",
            "teacher_binary",
            "soft_deviation",
            "semantic_contradiction",
            "transition_penalty",
            "local_acceptance",
            "effective_teacher_weight",
            "mixed_target",
            "student_prediction",
            "global_teacher_ratio",
            "transition_envelope",
        }
        forbidden = {
            "teacher_correction",
            "semantic_support",
            "temporal_support",
            "source_conflict",
            "temporal_instability",
            "combined_negative_evidence",
            "fused_support",
            "support_deficiency",
            "soft_inertia",
        }
        forbidden.update(
            name for name in payload if str(name).startswith("history_")
        )
        present_forbidden = sorted(forbidden.intersection(payload))
        if present_forbidden:
            raise RuntimeError(
                "AP-STCR semantic-only diagnostic contains forbidden "
                f"fields {present_forbidden}: {path}"
            )
    else:
        required = {
            "epoch",
            "dataset",
            "stem",
            "rgb",
            "fixed_pseudo",
            "fg_anchor_mask",
            "bg_anchor_mask",
            "semantic_margin",
            "teacher_binary",
            "teacher_correction",
            "semantic_support",
            "temporal_support",
            "local_acceptance",
            "effective_teacher_weight",
            "mixed_target",
            "student_prediction",
        }
    if schema_version == "ap_stcr_diagnostic_v2":
        required.add("source_conflict")
    elif schema_version == "ap_stcr_diagnostic_v3":
        required.update(
            {
                "teacher_soft",
                "soft_deviation",
                "source_conflict",
                "fused_support",
                "support_deficiency",
                "soft_inertia",
            }
        )
    elif schema_version == "ap_stcr_diagnostic_v4":
        required.update(
            {
                "teacher_soft",
                "soft_deviation",
                "semantic_contradiction",
                "temporal_instability",
                "combined_negative_evidence",
                "transition_penalty",
                "global_teacher_ratio",
                "transition_envelope",
            }
        )
    missing = sorted(required.difference(payload))
    if missing:
        raise RuntimeError(
            f"AP-STCR diagnostic is missing fields {missing}: {path}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Render AP-STCR offline diagnostics with GT."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--input",
        required=True,
        help="AP-STCR diagnostic .pt file or directory under workdir.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory under MY-baseline/workdir.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_path = _resolve_workdir_path(args.input, "--input", must_exist=True)
    out_root = _resolve_workdir_path(args.out, "--out", must_exist=False)
    out_root.mkdir(parents=True, exist_ok=True)

    paths = _diagnostic_paths(input_path)
    rendered = 0
    for path in paths:
        payload = torch_load(path, map_location="cpu")
        _validate_payload(payload, path)
        epoch = int(payload["epoch"])
        dataset = str(payload["dataset"])
        stem = str(payload["stem"])
        output_path = (
            out_root
            / f"epoch_{epoch:03d}"
            / f"{dataset}__{stem}_with_gt.png"
        )
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Output exists; pass --overwrite to replace it: {output_path}"
            )
        gt = _load_gt_68(cfg, payload)
        render_ap_stcr_diagnostic(payload, output_path, gt=gt)
        rendered += 1
        print(
            f"[AP-STCR OfflineVisual] {dataset}/{stem} "
            f"epoch={epoch:03d} -> {output_path}"
        )
    print(f"rendered = {rendered}")
    print(f"output_root = {out_root}")
    print("gt_used_for_training = False")


if __name__ == "__main__":
    main()
