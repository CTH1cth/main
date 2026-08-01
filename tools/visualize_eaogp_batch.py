#!/usr/bin/env python3
"""Render the GT-free nine-panel EAOGP diagnostic for saved audit payloads."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image, ImageDraw


MAIN_ROOT = Path(__file__).resolve().parents[1]
CTH_ROOT = MAIN_ROOT.parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.utils import torch_load  # noqa: E402


PANEL_FIELDS = (
    ("RGB", "image_68", "rgb"),
    ("DABE-v2 Hard", "dabe_static_68", "mask"),
    ("DABE-v2 response F", "foreground_response_68", "map"),
    ("Same-source background B", "background_evidence_68", "map"),
    ("Anchor confidence C", "anchor_confidence_68", "map"),
    ("Binary EMA Teacher", "teacher_binary_68", "mask"),
    ("DINO-only consensus", "dino_graph_consensus_68", "map"),
    ("Dual-graph consensus", "dual_graph_consensus_68", "map"),
    ("Effective EAOGP target", "effective_teacher_target_68", "map"),
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "sample"


def _single_plane(name: str, value: Any) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"EAOGP visualization field {name!r} must be a tensor.")
    value = value.detach().cpu().float()
    while value.ndim > 2 and int(value.shape[0]) == 1:
        value = value.squeeze(0)
    if value.ndim != 2:
        raise RuntimeError(
            f"EAOGP visualization field {name!r} must reduce to [H,W], "
            f"got {list(value.shape)}."
        )
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"EAOGP visualization field {name!r} contains NaN/Inf.")
    minimum = float(value.min().item())
    maximum = float(value.max().item())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise RuntimeError(
            f"EAOGP visualization field {name!r} is outside [0,1]: "
            f"{minimum:.9g}/{maximum:.9g}."
        )
    return value.clamp(0.0, 1.0)


def _rgb_tensor(name: str, value: Any) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"EAOGP visualization field {name!r} must be a tensor.")
    value = value.detach().cpu().float()
    while value.ndim > 3 and int(value.shape[0]) == 1:
        value = value.squeeze(0)
    if value.ndim != 3 or int(value.shape[0]) != 3:
        raise RuntimeError(
            f"EAOGP visualization field {name!r} must reduce to [3,H,W], "
            f"got {list(value.shape)}."
        )
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"EAOGP visualization field {name!r} contains NaN/Inf.")
    return value.clamp(0.0, 1.0)


def _rgb_image(value: torch.Tensor, size: int) -> Image.Image:
    array = value.permute(1, 2, 0).numpy()
    image = Image.fromarray((array * 255.0).round().astype(np.uint8), mode="RGB")
    return image.resize((size, size), resample=Image.Resampling.BILINEAR)


def _map_image(value: torch.Tensor, size: int, *, binary: bool) -> Image.Image:
    array = value.numpy()
    image = Image.fromarray((array * 255.0).round().astype(np.uint8), mode="L")
    resample = Image.Resampling.NEAREST if binary else Image.Resampling.BILINEAR
    return image.resize((size, size), resample=resample).convert("RGB")


def _overlay_image(
    rgb: torch.Tensor,
    field: torch.Tensor,
    *,
    color: tuple[int, int, int],
    size: int,
) -> Image.Image:
    base = np.asarray(_rgb_image(rgb, size), dtype=np.float32)
    alpha = np.asarray(
        _map_image(field, size, binary=False).convert("L"), dtype=np.float32
    ) / 255.0
    alpha = (0.72 * alpha)[..., None]
    tint = np.empty_like(base)
    tint[...] = np.asarray(color, dtype=np.float32)
    blended = base * (1.0 - alpha) + tint * alpha
    return Image.fromarray(np.clip(blended, 0.0, 255.0).round().astype(np.uint8))


def _render_row(
    panels: list[tuple[str, Image.Image]],
    *,
    sample_name: str,
    panel_size: int,
) -> Image.Image:
    header = 43
    footer = 24
    canvas = Image.new(
        "RGB",
        (panel_size * len(panels), panel_size + header + footer),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        left = index * panel_size
        draw.multiline_text((left + 4, 4), title, fill=(0, 0, 0), spacing=1)
        canvas.paste(panel, (left, header))
    draw.text((4, header + panel_size + 5), sample_name, fill=(0, 0, 0))
    return canvas


def render_eaogp_visualization(
    payload: Mapping[str, Any],
    output_dir: Path,
    *,
    panel_size: int = 180,
) -> dict[str, str]:
    """Render one sample. Diagnostic dominance masks never enter training."""

    forbidden = sorted(
        key
        for key in payload
        if str(key).strip().lower()
        in {"gt", "gt_path", "mask", "mask_path", "ground_truth"}
    )
    if forbidden:
        raise RuntimeError(f"EAOGP visualization payload contains GT fields: {forbidden}")
    if int(panel_size) < 64:
        raise ValueError("panel_size must be at least 64.")

    dataset = str(payload.get("dataset", "dataset"))
    stem = str(payload.get("stem", "sample"))
    sample_name = f"{dataset}__{stem}"
    sample_dir = output_dir / _safe_name(sample_name)
    sample_dir.mkdir(parents=True, exist_ok=True)

    tensors: dict[str, torch.Tensor] = {}
    panels: list[tuple[str, Image.Image]] = []
    for title, field, field_type in PANEL_FIELDS:
        if field_type == "rgb":
            tensor = _rgb_tensor(field, payload.get(field))
            tensors[field] = tensor
            panel = _rgb_image(tensor, int(panel_size))
        else:
            tensor = _single_plane(field, payload.get(field))
            tensors[field] = tensor
            panel = _map_image(
                tensor,
                int(panel_size),
                binary=field_type == "mask",
            )
        panels.append((title, panel))

    overview = _render_row(
        panels,
        sample_name=sample_name,
        panel_size=int(panel_size),
    )
    overview_path = sample_dir / "eaogp_nine_panel.png"
    overview.save(overview_path)

    new_foreground = _single_plane(
        "new_foreground_68", payload.get("new_foreground_68")
    )
    new_background = _single_plane(
        "new_background_68", payload.get("new_background_68")
    )
    anchor = tensors["anchor_confidence_68"]
    uncertainty = _single_plane(
        "teacher_uncertainty_68", payload.get("teacher_uncertainty_68")
    )
    anchor_mass = anchor
    graph_mass = (1.0 - anchor) * uncertainty
    nonzero = (anchor_mass + graph_mass) > 0.0
    anchor_dominant = ((anchor_mass >= graph_mass) & nonzero).float()
    graph_dominant = ((graph_mass > anchor_mass) & nonzero).float()

    rgb = tensors["image_68"]
    overlay_specs = (
        ("New foreground", new_foreground, (255, 32, 32)),
        ("New background", new_background, (32, 96, 255)),
        ("DABE anchor-dominant", anchor_dominant, (32, 220, 80)),
        ("Online graph-dominant", graph_dominant, (220, 32, 220)),
    )
    overlay_panels = [
        (
            title,
            _overlay_image(rgb, value, color=color, size=int(panel_size)),
        )
        for title, value, color in overlay_specs
    ]
    overlays = _render_row(
        overlay_panels,
        sample_name=(
            sample_name
            + " | dominance is diagnostic argmax(C, (1-C)U), never used in training"
        ),
        panel_size=int(panel_size),
    )
    overlay_path = sample_dir / "eaogp_overlays.png"
    overlays.save(overlay_path)

    metadata = {
        "schema": "eaogp_visualization_v1",
        "dataset": dataset,
        "stem": stem,
        "training_gt_used": False,
        "nine_panel": str(overview_path),
        "overlays": str(overlay_path),
        "anchor_dominance_definition": "C >= (1-C)*U, positive combined mass",
        "graph_dominance_definition": "(1-C)*U > C, positive combined mass",
        "dominance_used_for_training": False,
    }
    metadata_path = sample_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata["metadata"] = str(metadata_path)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a saved GT-free EAOGP shadow-audit payload."
    )
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-size", type=int, default=180)
    args = parser.parse_args()

    payload_path = args.payload.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    for path in (payload_path, output_dir):
        if path != CTH_ROOT and CTH_ROOT not in path.parents:
            raise RuntimeError(f"Path must stay inside {CTH_ROOT}: {path}")
    payload = torch_load(payload_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"EAOGP visualization payload must be a dict: {payload_path}")
    result = render_eaogp_visualization(
        payload,
        output_dir,
        panel_size=int(args.panel_size),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
