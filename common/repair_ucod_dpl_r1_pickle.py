#!/usr/bin/env python3
"""Repair an existing UCOD-DPL R1 cache without recomputing R1.

The auditable raw cache is stored with ``torch.save``. UCOD-DPL's
``MetaListPickleIO`` instead calls the standard-library ``pickle.load``.
This script extracts ``r1_hard_68`` from every raw entry and atomically
rewrites only the derived ``data_*.pkl`` files in the expected format.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import read_jsonl, torch_load  # noqa: E402


EXPECTED_TOTAL = 4040
EXPECTED_SHAPE = (1, 68, 68)


def _atomic_pickle_save(value, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _atomic_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _validate_target(target, source: Path) -> torch.Tensor:
    if not torch.is_tensor(target) or tuple(target.shape) != EXPECTED_SHAPE:
        actual = None if not torch.is_tensor(target) else tuple(target.shape)
        raise RuntimeError(f"Expected {EXPECTED_SHAPE}, got {actual}: {source}")
    target = target.detach().cpu().float().contiguous()
    if not torch.isfinite(target).all() or not torch.all((target == 0) | (target == 1)):
        raise RuntimeError(f"R1 target is not finite binary float data: {source}")
    return target


def repair(raw_manifest: Path, ucod_cache: Path) -> dict:
    raw_manifest = raw_manifest.resolve()
    ucod_cache = ucod_cache.resolve()
    protocol_path = ucod_cache / "protocol.json"
    if not raw_manifest.is_file():
        raise FileNotFoundError(raw_manifest)
    if not ucod_cache.is_dir() or not protocol_path.is_file():
        raise FileNotFoundError(ucod_cache if not ucod_cache.is_dir() else protocol_path)

    rows = read_jsonl(raw_manifest)
    if len(rows) != EXPECTED_TOTAL:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL} raw rows, got {len(rows)}")
    keys = [(str(row["dataset"]), str(row["stem"])) for row in rows]
    if len(set(keys)) != EXPECTED_TOTAL:
        raise RuntimeError("Raw R1 manifest contains duplicate dataset/stem keys")

    index = {}
    foreground_sum = 0.0
    for position, row in enumerate(rows):
        source = Path(row["cache_path"]).resolve()
        payload = torch_load(source, map_location="cpu")
        if not isinstance(payload, dict):
            raise RuntimeError(f"Raw R1 payload must be a dict: {source}")
        if (payload.get("dataset"), payload.get("stem")) != keys[position]:
            raise RuntimeError(f"Raw R1 key mismatch: {source}")
        target = _validate_target(payload.get("r1_hard_68"), source)
        foreground_sum += float(target.mean())
        filename = f"data_{position}.pkl"
        _atomic_pickle_save(target, ucod_cache / filename)
        index[str(position)] = filename

    _atomic_json(ucod_cache / "index.json", index)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["ucod_serialization"] = "python_pickle"
    protocol["ucod_cache_repaired_from_raw"] = True
    protocol["ucod_repair_source_manifest"] = str(raw_manifest)
    protocol["mean_foreground_ratio"] = foreground_sum / EXPECTED_TOTAL
    _atomic_json(protocol_path, protocol)

    # Exercise the exact reader used by UCOD-DPL on representative entries.
    checked = []
    for position in (0, EXPECTED_TOTAL // 2, EXPECTED_TOTAL - 1):
        path = ucod_cache / index[str(position)]
        with path.open("rb") as handle:
            target = _validate_target(pickle.load(handle), path)
        checked.append({"index": position, "shape": list(target.shape)})

    return {
        "status": "PASS",
        "rewritten": EXPECTED_TOTAL,
        "serialization": "python_pickle",
        "raw_manifest": str(raw_manifest),
        "ucod_cache": str(ucod_cache),
        "representative_reads": checked,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_manifest", required=True, type=Path)
    parser.add_argument("--ucod_cache", required=True, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(repair(args.raw_manifest, args.ucod_cache), ensure_ascii=False, indent=2))
