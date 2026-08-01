#!/usr/bin/env python3
"""Read-only replay of pre-update Full-ECST temporal memory snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader


MAIN_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MAIN_ROOT.parent
CTH_ROOT = MAIN_ROOT.parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.dataset import CachedTrainDataset  # noqa: E402
from common.ecst import TemporalTeacherMemory  # noqa: E402
from common.utils import (  # noqa: E402
    config_to_dict,
    load_config,
    make_jsonable,
    set_seed,
    torch_load,
)
from model import build_seg_head  # noqa: E402
from train import (  # noqa: E402
    extract_logits,
    forward_seg_head,
    make_image_68,
    make_model_input,
    resize_logits_for_loss,
    set_model_epoch,
)


DEFAULT_CONFIG = MAIN_ROOT / "configs" / (
    "dinov1_s8_dabe_clean_v1_dp_dabev2hard_ecst_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "workdir" / (
    "dinov1_s8_dabe_clean_v1_dp_dabev2hard_ecst_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5"
) / "train" / "ckpt"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "workdir" / "dabev2hard_full_ecst_causal_audit"
)


def _inside_cth(path: Path) -> Path:
    path = path.expanduser()
    path = (Path.cwd() / path).resolve() if not path.is_absolute() else path.resolve()
    if path != CTH_ROOT and CTH_ROOT not in path.parents:
        raise RuntimeError(f"Path must stay inside {CTH_ROOT}: {path}")
    return path


def _parse_epochs(value: str) -> list[int]:
    result = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not result or result[0] < 1:
        raise argparse.ArgumentTypeError("epochs must be positive integers")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_fingerprint(cfg: Any) -> str:
    payload = json.dumps(
        make_jsonable(config_to_dict(cfg)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = torch.device(value)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return result


def _snapshot_path(output_root: Path, target_epoch: int) -> Path:
    return output_root / "memory_snapshots" / f"epoch_{int(target_epoch):03d}.pt"


def replay_memory_snapshots(
    *,
    cfg: Any,
    checkpoint_root: Path,
    output_root: Path,
    target_epochs: Sequence[int],
    max_samples: int = -1,
    batch_size: int = 16,
    device: torch.device,
    overwrite: bool = False,
) -> dict[str, Any]:
    targets = sorted({int(epoch) for epoch in target_epochs})
    if not targets or targets[0] < 1:
        raise ValueError("target_epochs must be positive.")
    update_end = int(getattr(cfg, "ECST_MEMORY_UPDATE_END_EPOCH", 20))
    if any(epoch > update_end + 1 for epoch in targets):
        raise RuntimeError(
            "History replay targets must be within the active pre-reset memory "
            f"window (<= {update_end + 1})."
        )
    if max_samples == 0 or max_samples < -1:
        raise ValueError("max_samples must be -1 or positive.")
    dataset = CachedTrainDataset(cfg, max_samples=int(max_samples))
    loader = DataLoader(
        dataset,
        batch_size=min(int(batch_size), len(dataset)),
        shuffle=False,
        num_workers=0,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    memory = TemporalTeacherMemory(
        num_samples=len(dataset),
        height=int(cfg.LOSS_SIZE),
        width=int(cfg.LOSS_SIZE),
        dtype=str(getattr(cfg, "ECST_MEMORY_DTYPE", "float16")),
    )
    teacher = build_seg_head(dataset.in_channels, cfg).to(device)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    snapshot_root = output_root / "memory_snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    source_hashes: dict[str, str] = {}
    saved: list[str] = []
    fingerprint = _config_fingerprint(cfg)

    def save_snapshot(target_epoch: int) -> None:
        path = _snapshot_path(output_root, target_epoch)
        if path.exists() and not overwrite:
            existing = torch_load(path, map_location="cpu")
            if (
                isinstance(existing, dict)
                and int(existing.get("epoch", -1)) == int(target_epoch)
                and int(existing.get("sample_count", -1)) == len(dataset)
                and str(existing.get("config_fingerprint", "")) == fingerprint
            ):
                saved.append(str(path))
                return
            raise FileExistsError(f"Conflicting memory snapshot exists: {path}")
        state = memory.state_dict()
        payload = {
            "schema": "full_ecst_replayed_memory_v1",
            "epoch": int(target_epoch),
            "memory_semantics": "pre_update_history_for_target_epoch",
            "last_source_epoch": int(target_epoch) - 1,
            "sample_count": len(dataset),
            "full_manifest": int(max_samples) < 0,
            "mean": state["mean"],
            "second": state["second"],
            "count": state["count"],
            "dtype": state["dtype"],
            "height": state["height"],
            "width": state["width"],
            "source_checkpoint_hashes": dict(source_hashes),
            "config_fingerprint": fingerprint,
            "training_gt_read": False,
            "test_gt_read": False,
        }
        torch.save(payload, path)
        saved.append(str(path))

    if 1 in targets:
        save_snapshot(1)
    max_source_epoch = max(targets) - 1
    rho = float(getattr(cfg, "ECST_TEMPORAL_RHO", 0.9))
    with torch.no_grad():
        for source_epoch in range(1, max_source_epoch + 1):
            checkpoint_path = checkpoint_root / f"epoch_{source_epoch:03d}.pth"
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    "Cannot replay exact epoch order because an intermediate "
                    f"checkpoint is missing: {checkpoint_path}"
                )
            checkpoint = torch_load(checkpoint_path, map_location="cpu")
            if int(checkpoint.get("epoch", -1)) != source_epoch:
                raise RuntimeError(
                    f"Checkpoint epoch mismatch at {checkpoint_path}."
                )
            if not isinstance(checkpoint.get("teacher"), dict):
                raise RuntimeError(f"Teacher state missing: {checkpoint_path}")
            teacher.load_state_dict(checkpoint["teacher"], strict=True)
            set_model_epoch(teacher, source_epoch)
            source_hashes[f"epoch_{source_epoch:03d}"] = _sha256(checkpoint_path)
            seen = torch.zeros(len(dataset), dtype=torch.bool)
            for batch in loader:
                if any(str(key).lower() in {"gt", "gt_path", "mask", "mask_path"} for key in batch):
                    raise RuntimeError("GT leaked into ECST memory replay batch.")
                indices = batch["sample_index"].detach().long().reshape(-1)
                if bool(seen.index_select(0, indices).any().item()):
                    raise RuntimeError(
                        f"A sample was updated twice in replay epoch {source_epoch}."
                    )
                model_input = make_model_input(cfg, batch, device)
                image_68 = make_image_68(cfg, batch, device)
                output = forward_seg_head(
                    teacher,
                    model_input,
                    cfg,
                    image_68=image_68,
                    return_aux=False,
                )
                logits = resize_logits_for_loss(extract_logits(output), cfg)
                probability = torch.sigmoid(logits).detach()
                memory.update(indices, probability, rho=rho)
                seen.index_fill_(0, indices, True)
            if not bool(seen.all().item()):
                raise RuntimeError(
                    f"Replay epoch {source_epoch} did not update every sample once."
                )
            target_epoch = source_epoch + 1
            if target_epoch in targets:
                save_snapshot(target_epoch)
    return {
        "schema": "full_ecst_memory_replay_summary_v1",
        "target_epochs": targets,
        "sample_count": len(dataset),
        "full_manifest": int(max_samples) < 0,
        "source_checkpoint_hashes": source_hashes,
        "snapshots": saved,
        "training_gt_read": False,
        "test_gt_read": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--epochs", type=_parse_epochs, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config_path = _inside_cth(args.config)
    cfg = load_config(config_path)
    if not bool(getattr(cfg, "USE_ECST", False)):
        raise RuntimeError("Memory replay requires the Full ECST config.")
    set_seed(int(cfg.SEED))
    summary = replay_memory_snapshots(
        cfg=cfg,
        checkpoint_root=_inside_cth(args.checkpoint_root),
        output_root=_inside_cth(args.output_root),
        target_epochs=args.epochs,
        max_samples=int(args.max_samples),
        batch_size=int(args.batch_size),
        device=_device(args.device),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

