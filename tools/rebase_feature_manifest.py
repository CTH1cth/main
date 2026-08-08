#!/usr/bin/env python3
"""Safely rebase a copied DINO feature manifest to the current server."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path


EXPECTED = {"CHAMELEON": 76, "TE-CAMO": 250, "TE-COD10K": 2026, "NC4K": 4121}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--feature_root", required=True, help="Root containing test/<dataset>/<stem>.pt")
    p.add_argument("--data_root", required=True, help="Root containing <dataset>/im and gt")
    p.add_argument("--backup_suffix", default=".before_rebase")
    return p.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def image_path(data_root: Path, dataset: str, stem: str) -> Path:
    image_dir = data_root / dataset / "im"
    matches = [image_dir / f"{stem}{suffix}" for suffix in IMAGE_SUFFIXES]
    matches = [p.resolve() for p in matches if p.is_file()]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one image for {dataset}/{stem}, found {matches}")
    return matches[0]


def main() -> None:
    a = parse_args()
    manifest = Path(a.manifest).expanduser().resolve()
    feature_root = Path(a.feature_root).expanduser().resolve()
    data_root = Path(a.data_root).expanduser().resolve()
    rows = read_rows(manifest)
    counts = Counter(str(r.get("dataset")) for r in rows)
    if len(rows) != sum(EXPECTED.values()) or dict(counts) != EXPECTED:
        raise RuntimeError(f"formal count mismatch: total={len(rows)}, counts={dict(counts)}")

    rebased = []
    missing = []
    for row in rows:
        dataset, stem = str(row["dataset"]), str(row["stem"])
        cache = (feature_root / "test" / dataset / f"{stem}.pt").resolve()
        if not cache.is_file():
            missing.append(str(cache))
            continue
        updated = dict(row)
        updated["cache_path"] = str(cache)
        updated["image_path"] = str(image_path(data_root, dataset, stem))
        rebased.append(updated)
    if missing:
        raise FileNotFoundError(f"{len(missing)} feature files missing; first={missing[:3]}")

    backup = manifest.with_name(manifest.name + a.backup_suffix)
    if not backup.exists():
        shutil.copy2(manifest, backup)
    temporary = manifest.with_name(f".{manifest.name}.{os.getpid()}.tmp")
    with temporary.open("w") as f:
        for row in rebased:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, manifest)
    print(json.dumps({
        "status": "rebased", "rows": len(rebased), "missing": 0,
        "manifest": str(manifest), "backup": str(backup),
        "feature_root": str(feature_root), "data_root": str(data_root),
        "counts": dict(counts),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
