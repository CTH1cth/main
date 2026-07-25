"""Fixed-layout BISA-v0 diagnostic rendering."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _show(ax, value, title, *, cmap="viridis", vmin=None, vmax=None):
    image = ax.imshow(np.asarray(value), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    return image


def render_bisa_visualization(payload: dict, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(3, 4, figsize=(16, 11), constrained_layout=True)
    axes = axes.reshape(-1)
    axes[0].imshow(np.asarray(payload["rgb"]))
    axes[0].set_title("RGB", fontsize=8)
    axes[0].axis("off")
    _show(axes[1], payload["gt"], "GT (diagnostic only)", cmap="gray", vmin=0, vmax=1)
    _show(axes[2], payload["dabe_target_soft"], "DABE target soft", vmin=0, vmax=1)
    _show(axes[3], payload["dabe_residual"], "DABE background residual", vmin=0, vmax=1)
    _show(axes[4], payload["teacher_prob"], "Teacher coarse probability", vmin=0, vmax=1)
    _show(axes[5], payload["teacher_error"], "Teacher error map", cmap="magma", vmin=0, vmax=3)
    limit = max(
        1e-6,
        float(np.nanquantile(np.abs(payload["weighted_self"]), 0.98)),
        float(np.nanquantile(np.abs(payload["weighted_local"]), 0.98)),
        float(np.nanquantile(np.abs(payload["random_local"]), 0.98)),
    )
    _show(axes[6], payload["weighted_self"], "weighted c_self centered", cmap="coolwarm", vmin=-limit, vmax=limit)
    _show(axes[7], payload["weighted_local"], "weighted c_local centered", cmap="coolwarm", vmin=-limit, vmax=limit)
    _show(axes[8], payload["random_local"], "random c_local centered", cmap="coolwarm", vmin=-limit, vmax=limit)
    _show(axes[9], payload["spillover"], "Spill ratio map", cmap="inferno", vmin=0)
    _show(axes[10], payload["group_map"], "Intervention groups", cmap="tab20")
    if payload.get("final_prob") is not None:
        _show(axes[11], payload["final_prob"], "Optional NDR final probability", vmin=0, vmax=1)
    else:
        axes[11].axis("off")
    figure.suptitle(str(payload.get("title", "BISA-v0")), fontsize=11)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
