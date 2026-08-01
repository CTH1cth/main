#!/usr/bin/env python3
"""Render the seven-panel ECTP diagnostic for one fixed training sample.

This intentionally avoids loading a full model/checkpoint.  The caller gives
the current Teacher probability (or binary mask) explicitly; the script pairs
it with the authoritative DABE-v2 and DABE-Clean cache entries and applies the
same production ECTP target builder used by training.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as torch_f  # noqa: E402
from PIL import Image  # noqa: E402


MAIN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MAIN_ROOT.parent
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.ectp import (  # noqa: E402
    build_ectp_projected_target,
    validate_ectp_config,
)
from common.utils import (  # noqa: E402
    load_config,
    manifest_to_map,
    read_jsonl,
    torch_load,
)


DEFAULT_CONFIG = (
    MAIN_ROOT
    / "configs"
    / "dinov1_s8_dabe_clean_v1_dp_dabev2hard_ectp_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "workdir" / "ectp_v1_batch_visualizations"
TEACHER_KEYS = (
    "teacher_probability",
    "teacher_prob",
    "teacher_soft",
    "probability",
    "prob",
    "teacher_binary",
    "mask",
    "prediction",
    "pred",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize ECTP for one fixed manifest sample. The explicit "
            "Teacher input must be a [0,1] image or tensor."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--stem", required=True)
    parser.add_argument(
        "--teacher-input",
        type=Path,
        required=True,
        help="Teacher probability/binary PNG or .pt/.pth tensor payload.",
    )
    parser.add_argument(
        "--teacher-key",
        default=None,
        help="Tensor key for a dict-valued .pt/.pth payload.",
    )
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument(
        "--alpha",
        type=float,
        required=True,
        help="Actual static_weight used by the training loop at this epoch.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        required=True,
        help="Actual teacher_weight used by the training loop at this epoch.",
    )
    parser.add_argument("--rgb", type=Path, default=None)
    parser.add_argument("--dabe-v2-root", type=Path, default=None)
    parser.add_argument("--clean-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_cfg_path(value: Any, override: Path | None) -> Path:
    path = override if override is not None else Path(str(value))
    path = path.expanduser()
    if not path.is_absolute():
        path = MAIN_ROOT / path
    return path.resolve()


def _load_payload_tensor(path: Path, key: str | None) -> torch.Tensor:
    suffix = path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        payload = torch_load(path, map_location="cpu")
        if isinstance(payload, dict):
            selected_key = key
            if selected_key is None:
                selected_key = next(
                    (
                        candidate
                        for candidate in TEACHER_KEYS
                        if torch.is_tensor(payload.get(candidate))
                    ),
                    None,
                )
            if selected_key is None or not torch.is_tensor(
                payload.get(selected_key)
            ):
                tensor_keys = sorted(
                    name for name, value in payload.items() if torch.is_tensor(value)
                )
                raise RuntimeError(
                    "Could not select Teacher tensor; pass --teacher-key. "
                    f"tensor_keys={tensor_keys} | {path}"
                )
            value = payload[selected_key]
        elif torch.is_tensor(payload):
            if key is not None:
                raise RuntimeError(
                    "--teacher-key was provided but Teacher payload is a tensor."
                )
            value = payload
        else:
            raise TypeError(f"Unsupported Teacher payload type: {type(payload)!r}")
        tensor = value.detach().cpu().float()
    else:
        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array)

    while tensor.ndim > 2 and int(tensor.shape[0]) == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise RuntimeError(
            "Teacher input must reduce to one 2-D probability plane, got "
            f"shape={tuple(tensor.shape)} | {path}"
        )
    if not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError(f"Teacher input contains NaN/Inf: {path}")
    value_min = float(tensor.min().item())
    value_max = float(tensor.max().item())
    if value_min < 0.0 or value_max > 1.0:
        raise RuntimeError(
            "Teacher input must be a probability/mask in [0,1], got "
            f"[{value_min},{value_max}] | {path}"
        )
    return tensor.detach()


def _load_cache_tensor(
    payload: dict[str, Any],
    key: str,
    expected_size: int,
    path: Path,
) -> torch.Tensor:
    value = payload.get(key)
    if not torch.is_tensor(value):
        raise RuntimeError(f"Missing {key!r}: {path}")
    value = value.detach().cpu().float()
    if tuple(value.shape) != (1, expected_size, expected_size):
        raise RuntimeError(
            f"{key} shape={tuple(value.shape)} != "
            f"{(1, expected_size, expected_size)} | {path}"
        )
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{key} contains NaN/Inf: {path}")
    if bool(((value < 0.0) | (value > 1.0)).any().item()):
        raise RuntimeError(f"{key} is outside [0,1]: {path}")
    return value


def _check_identity(
    payload: dict[str, Any], dataset: str, stem: str, path: Path
) -> None:
    if (
        str(payload.get("dataset")) != dataset
        or str(payload.get("stem")) != stem
    ):
        raise RuntimeError(
            "Cache identity mismatch | "
            f"expected={dataset}/{stem} | "
            f"actual={payload.get('dataset')}/{payload.get('stem')} | {path}"
        )


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-") or "sample"


def _stat_title(name: str, value: torch.Tensor) -> str:
    return (
        f"{name}\nmin={float(value.min()):.3f} "
        f"mean={float(value.mean()):.3f} max={float(value.max()):.3f}"
    )


def main() -> int:
    args = parse_args()
    if args.epoch < 0:
        raise ValueError(f"--epoch must be non-negative, got {args.epoch}.")
    alpha = float(args.alpha)
    beta = float(args.beta)
    if not (math.isfinite(alpha) and math.isfinite(beta)):
        raise ValueError("--alpha and --beta must be finite.")
    if alpha < 0.0 or beta < 0.0 or abs(alpha + beta - 1.0) > 1e-6:
        raise ValueError(
            f"Expected alpha,beta >= 0 and alpha+beta=1, got {alpha},{beta}."
        )

    config_path = args.config.expanduser().resolve()
    cfg = load_config(config_path)
    validate_ectp_config(cfg)
    loss_size = int(cfg.LOSS_SIZE)
    dabe_root = _resolve_cfg_path(cfg.DABE_CLEAN_DABE_V2_ROOT, args.dabe_v2_root)
    clean_root = _resolve_cfg_path(cfg.DABE_CLEAN_ROOT, args.clean_root)
    dabe_manifest = dabe_root / "manifest_train.jsonl"
    clean_manifest = clean_root / "manifest_train.jsonl"
    dabe_map = manifest_to_map(read_jsonl(dabe_manifest), dabe_manifest)
    clean_map = manifest_to_map(read_jsonl(clean_manifest), clean_manifest)
    key = (str(args.dataset), str(args.stem))
    if key not in dabe_map or key not in clean_map:
        raise KeyError(
            f"Sample {key} must exist in both DABE-v2 and DABE-Clean manifests."
        )

    dabe_row = dabe_map[key]
    clean_row = clean_map[key]
    dabe_path = Path(dabe_row["cache_path"])
    clean_path = Path(clean_row["cache_path"])
    dabe_payload = torch_load(dabe_path, map_location="cpu")
    clean_payload = torch_load(clean_path, map_location="cpu")
    if not isinstance(dabe_payload, dict) or not isinstance(clean_payload, dict):
        raise TypeError(f"Cache payload must be dict for {key}.")
    _check_identity(dabe_payload, key[0], key[1], dabe_path)
    _check_identity(clean_payload, key[0], key[1], clean_path)

    foreground = _load_cache_tensor(
        dabe_payload, "p_dabe_68", loss_size, dabe_path
    )
    foreground_clean = _load_cache_tensor(
        clean_payload, "foreground_evidence_68", loss_size, clean_path
    )
    background = _load_cache_tensor(
        clean_payload, "background_evidence_68", loss_size, clean_path
    )
    cache_error = float((foreground - foreground_clean).abs().max().item())
    if cache_error > 1e-5:
        raise RuntimeError(
            "DABE-v2/Clean foreground cache mismatch exceeds 1e-5: "
            f"{cache_error} | {key}"
        )
    static_target = (foreground > 0.5).float().detach()

    teacher_probability = _load_payload_tensor(
        args.teacher_input.expanduser().resolve(), args.teacher_key
    )
    if tuple(teacher_probability.shape) != (loss_size, loss_size):
        teacher_probability = torch_f.interpolate(
            teacher_probability[None, None],
            size=(loss_size, loss_size),
            mode="bilinear",
            align_corners=False,
        )[0, 0].detach()
    # Match the repository's existing binary EMA-Teacher source exactly; ECTP
    # must not redefine its thresholding behavior.
    teacher_binary = (teacher_probability[None] >= 0.5).float().detach()

    foreground4 = foreground.unsqueeze(0)
    background4 = background.unsqueeze(0)
    static4 = static_target.unsqueeze(0)
    teacher4 = teacher_binary.unsqueeze(0)
    projected_result = build_ectp_projected_target(
        foreground_evidence=foreground4,
        background_evidence=background4,
        static_target=static4,
        teacher_binary=teacher4,
        static_weight=alpha,
        teacher_weight=beta,
    )
    if not isinstance(projected_result, tuple) or len(projected_result) != 2:
        raise RuntimeError(
            "build_ectp_projected_target must return (target, stats)."
        )
    projected, ectp_stats = projected_result
    projected = projected[0].detach().cpu()
    support = ectp_stats["support"][0].detach().cpu()
    conflict = ectp_stats["conflict_bool"][0].detach().cpu()
    overlap = float(ectp_stats["overlap"])
    conflict_ratio = float(ectp_stats["conflict_ratio"])
    projected_shift_mean = float(ectp_stats["projected_target_shift_mean"])

    rgb_path = (
        args.rgb.expanduser().resolve()
        if args.rgb is not None
        else Path(dabe_row.get("image_path", ""))
    )
    if not rgb_path.is_file():
        raise FileNotFoundError(
            "RGB path is unavailable; pass --rgb explicitly: " f"{rgb_path}"
        )
    with Image.open(rgb_path) as image:
        rgb = np.asarray(image.convert("RGB"))

    panels = (
        (rgb, "RGB", None),
        (static_target[0].numpy(), "DABE-v2 Hard static", "gray"),
        (
            foreground[0].numpy(),
            _stat_title("Foreground evidence F", foreground),
            "viridis",
        ),
        (
            background[0].numpy(),
            _stat_title("Background evidence B", background),
            "viridis",
        ),
        (teacher_binary[0].numpy(), "Original Binary Teacher", "gray"),
        (
            support[0].numpy(),
            _stat_title("Support C + conflict overlay", support),
            "viridis",
        ),
        (
            projected[0].numpy(),
            _stat_title("Projected Teacher target", projected),
            "gray",
        ),
    )
    figure, axes = plt.subplots(1, 7, figsize=(25.2, 4.2), constrained_layout=True)
    for axis, (array, title, cmap) in zip(axes, panels):
        if cmap is None:
            axis.imshow(array)
        else:
            axis.imshow(
                array,
                cmap=cmap,
                vmin=0.0,
                vmax=1.0,
                interpolation="nearest",
            )
        axis.set_title(title, fontsize=9)
        axis.axis("off")
    conflict_overlay = np.ma.masked_where(
        ~conflict[0].numpy(), conflict[0].numpy().astype(np.float32)
    )
    axes[5].imshow(
        conflict_overlay,
        cmap="Reds",
        vmin=0.0,
        vmax=1.0,
        alpha=0.48,
        interpolation="nearest",
    )
    figure.suptitle(
        "ECTP | "
        f"{key[0]}/{key[1]} | epoch={args.epoch} | "
        f"alpha={alpha:.4f} beta={beta:.4f} overlap={overlap:.4f} | "
        f"conflict={conflict_ratio:.6f} shift={projected_shift_mean:.6f}",
        fontsize=11,
    )

    filename = (
        f"{_safe_name(key[0])}_{_safe_name(key[1])}_"
        f"epoch{args.epoch:02d}_alpha{alpha:.4f}_beta{beta:.4f}_"
        f"overlap{overlap:.4f}_conflict{conflict_ratio:.6f}_"
        f"projected_shift_mean{projected_shift_mean:.6f}.png"
    )
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / filename
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Visualization exists; pass --overwrite to replace: {output_path}"
        )
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)

    metadata = {
        "output": str(output_path),
        "dataset": key[0],
        "stem": key[1],
        "epoch": args.epoch,
        "alpha": alpha,
        "beta": beta,
        "overlap": overlap,
        "conflict_ratio": conflict_ratio,
        "projected_shift_mean": projected_shift_mean,
        "foreground_cache_consistency_error": cache_error,
        "ground_truth_read": False,
    }
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
