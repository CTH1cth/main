#!/usr/bin/env python3
"""Export cached DINOv1-S/8 R1 maps as a SINet-V2 training dataset.

The export is deliberately supervision-only: it reads the immutable identity-view
R1 cache, upsamples each soft 37x37 residual to the source image resolution, then
applies a strict ``> threshold`` binarization. Ground-truth masks are never read
during pseudo-mask generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = (
    PROJECT_ROOT
    / "datasets/cache/dabe_cvbr_v1_train_singleview/dinov1-s8"
)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "datasets/COD"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "datasets/cache/sinetv2_r1_v1s_train"

EXPECTED_DATASET_COUNTS = {"TR-CAMO": 1000, "TR-COD10K": 3040}
TEST_DATASET_MAP = {
    "CHAMELEON": "CHAMELEON",
    "CAMO": "TE-CAMO",
    "COD10K": "TE-COD10K",
    "NC4K": "NC4K",
}
EXPECTED_TEST_COUNTS = {
    "CHAMELEON": 76,
    "CAMO": 250,
    "COD10K": 2026,
    "NC4K": 4121,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--field", default="b0_r1_bw2_37")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--expected-count", type=int, default=4040)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace already generated masks and links without deleting directories.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit an existing export without loading or rewriting R1 caches.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep already valid masks and continue exporting missing samples.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Manifest line {line_no} is not an object: {path}")
            rows.append(row)
    return rows


def image_path_for(data_root: Path, dataset: str, stem: str) -> Path:
    image_dir = data_root / dataset / "im"
    candidates = [image_dir / f"{stem}{suffix}" for suffix in (".jpg", ".png", ".jpeg")]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise FileNotFoundError(
            f"Expected exactly one source image for {dataset}/{stem}, found {found}"
        )
    return found[0].resolve()


def ensure_symlink(link_path: Path, target: Path, overwrite: bool) -> None:
    target = target.resolve()
    if link_path.is_symlink():
        current = link_path.resolve(strict=False)
        if current == target:
            return
        if not overwrite:
            raise FileExistsError(f"Symlink points elsewhere: {link_path} -> {current}")
        link_path.unlink()
    elif link_path.exists():
        if not overwrite:
            raise FileExistsError(f"Path already exists and is not a symlink: {link_path}")
        if link_path.is_dir():
            raise IsADirectoryError(f"Refusing to replace directory: {link_path}")
        link_path.unlink()
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target, target_is_directory=target.is_dir())


def validate_manifest(
    rows: List[Dict[str, Any]], cache_root: Path, expected_count: int
) -> None:
    if len(rows) != expected_count:
        raise RuntimeError(f"R1 manifest count {len(rows)} != expected {expected_count}")
    counts = Counter(str(row.get("dataset")) for row in rows)
    if dict(counts) != EXPECTED_DATASET_COUNTS:
        raise RuntimeError(
            f"Unexpected dataset counts: {dict(counts)} != {EXPECTED_DATASET_COUNTS}"
        )
    keys = [(str(row.get("dataset")), str(row.get("stem"))) for row in rows]
    if len(set(keys)) != len(keys):
        duplicates = [key for key, count in Counter(keys).items() if count > 1]
        raise RuntimeError(f"Duplicate manifest identities: {duplicates[:10]}")
    stems = [stem for _, stem in keys]
    if len(set(stems)) != len(stems):
        duplicates = [stem for stem, count in Counter(stems).items() if count > 1]
        raise RuntimeError(
            "SINet-V2 uses a flat training directory, but duplicate stems exist: "
            f"{duplicates[:10]}"
        )
    resolved_root = cache_root.resolve()
    for row in rows:
        cache_path = Path(str(row["cache_path"])).resolve()
        if resolved_root not in cache_path.parents:
            raise RuntimeError(f"Cache path escapes cache root: {cache_path}")
        if not cache_path.is_file():
            raise FileNotFoundError(cache_path)


def export_masks(args: argparse.Namespace) -> Dict[str, Any]:
    cache_root = args.cache_root.resolve()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    source_manifest = cache_root / "manifest_train.jsonl"
    rows = read_jsonl(source_manifest)
    validate_manifest(rows, cache_root, args.expected_count)

    image_output = output_root / "TrainDataset/Imgs"
    mask_output = output_root / "TrainDataset/PseudoMask"
    image_output.mkdir(parents=True, exist_ok=True)
    mask_output.mkdir(parents=True, exist_ok=True)

    export_rows: List[Dict[str, Any]] = []
    ratios: List[float] = []
    empty_count = 0
    full_count = 0

    for index, row in enumerate(rows, start=1):
        dataset = str(row["dataset"])
        stem = str(row["stem"])
        cache_path = Path(str(row["cache_path"])).resolve()
        image_path = image_path_for(data_root, dataset, stem)
        image_link = image_output / image_path.name
        mask_path = mask_output / f"{stem}.png"

        with Image.open(image_path) as source_image:
            width, height = source_image.size

        if mask_path.exists() and args.resume:
            with Image.open(mask_path) as existing_mask:
                if existing_mask.size != (width, height):
                    raise RuntimeError(
                        f"Existing mask size mismatch during resume: {mask_path}"
                    )
                existing_array = np.asarray(existing_mask.convert("L"), dtype=np.uint8)
            unique = set(np.unique(existing_array).tolist())
            if not unique.issubset({0, 255}):
                raise RuntimeError(
                    f"Existing mask is not binary during resume: {mask_path}"
                )
            ensure_symlink(image_link, image_path, overwrite=False)
            binary = existing_array == 255
            foreground_ratio = float(binary.mean())
            ratios.append(foreground_ratio)
            empty_count += int(not bool(binary.any()))
            full_count += int(bool(binary.all()))
            export_rows.append(
                {
                    "dataset": dataset,
                    "stem": stem,
                    "source_image": str(image_path),
                    "source_r1_cache": str(cache_path),
                    "image_link": str(image_link),
                    "pseudo_mask": str(mask_path),
                    "source_shape": [1, 37, 37],
                    "output_shape": [height, width],
                    "foreground_ratio": foreground_ratio,
                }
            )
            if index % 500 == 0:
                print(f"resumed {index}/{len(rows)}")
            continue

        if mask_path.exists() and not args.overwrite:
            raise FileExistsError(
                "Output mask already exists; use --resume, --overwrite, or "
                f"--audit-only: {mask_path}"
            )

        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(f"R1 payload is not a dict: {cache_path}")
        if payload.get("dataset") != dataset or payload.get("stem") != stem:
            raise RuntimeError(f"R1 payload identity mismatch: {cache_path}")
        if bool(payload.get("gt_used_for_generation", True)):
            raise RuntimeError(f"R1 payload reports GT use: {cache_path}")
        if payload.get("source_augs") != ["identity"]:
            raise RuntimeError(f"R1 payload is not identity single-view: {cache_path}")
        residual = payload.get(args.field)
        if not torch.is_tensor(residual):
            raise KeyError(f"Missing tensor field {args.field!r}: {cache_path}")
        residual = residual.detach().to(dtype=torch.float32, device="cpu")
        if tuple(residual.shape) != (1, 37, 37):
            raise RuntimeError(f"Unexpected R1 shape {tuple(residual.shape)}: {cache_path}")
        if not torch.isfinite(residual).all():
            raise RuntimeError(f"Non-finite R1 values: {cache_path}")
        if float(residual.min()) < 0.0 or float(residual.max()) > 1.0:
            raise RuntimeError(f"R1 values outside [0, 1]: {cache_path}")

        upsampled = F.interpolate(
            residual.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        binary = upsampled > float(args.threshold)
        foreground_ratio = float(binary.float().mean())
        ratios.append(foreground_ratio)
        empty_count += int(not bool(binary.any()))
        full_count += int(bool(binary.all()))

        mask_array = binary.numpy().astype(np.uint8) * 255
        temporary_mask = mask_path.with_name(
            f".{mask_path.name}.{os.getpid()}.tmp.png"
        )
        Image.fromarray(mask_array, mode="L").save(temporary_mask)
        os.replace(temporary_mask, mask_path)
        ensure_symlink(image_link, image_path, overwrite=args.overwrite)

        export_rows.append(
            {
                "dataset": dataset,
                "stem": stem,
                "source_image": str(image_path),
                "source_r1_cache": str(cache_path),
                "image_link": str(image_link),
                "pseudo_mask": str(mask_path),
                "source_shape": [1, 37, 37],
                "output_shape": [height, width],
                "foreground_ratio": foreground_ratio,
            }
        )
        if index % 500 == 0 or index == len(rows):
            print(f"exported {index}/{len(rows)}")

    test_root = output_root / "TestDataset"
    for output_name, source_name in TEST_DATASET_MAP.items():
        source_dataset = data_root / source_name
        image_source = source_dataset / "im"
        gt_source = source_dataset / "gt"
        image_count = sum(path.is_file() for path in image_source.iterdir())
        gt_count = sum(path.is_file() for path in gt_source.iterdir())
        expected = EXPECTED_TEST_COUNTS[output_name]
        if image_count != expected or gt_count != expected:
            raise RuntimeError(
                f"{output_name} count mismatch: images={image_count}, gt={gt_count}, "
                f"expected={expected}"
            )
        ensure_symlink(test_root / output_name / "Imgs", image_source, args.overwrite)
        ensure_symlink(test_root / output_name / "GT", gt_source, args.overwrite)

    export_manifest = output_root / "manifest_train.jsonl"
    with export_manifest.open("w", encoding="utf-8") as handle:
        for row in export_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    ratio_array = np.asarray(ratios, dtype=np.float64)
    protocol = {
        "protocol": "sinetv2_r1_v1s_static_hard",
        "num_samples": len(rows),
        "dataset_counts": dict(Counter(row["dataset"] for row in rows)),
        "r1_source": "DINOv1-S/8 identity single-view residual_pass1_37",
        "r1_cache_field": args.field,
        "source_grid": [37, 37],
        "conversion": (
            "bilinear upsample soft R1 to native image size with align_corners=False, "
            f"then strict > {args.threshold}"
        ),
        "threshold": float(args.threshold),
        "gt_used_for_export": False,
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "foreground_ratio": {
            "mean": float(ratio_array.mean()),
            "std": float(ratio_array.std()),
            "min": float(ratio_array.min()),
            "max": float(ratio_array.max()),
        },
        "empty_masks": empty_count,
        "full_masks": full_count,
        "train_image_root": str(image_output),
        "train_pseudo_mask_root": str(mask_output),
        "test_root": str(test_root),
        "official_sinetv2_source": "/home/dell01/CTH/0base/RISE-master/SINet-V2",
        "official_training_defaults": {
            "epoch_argument": 100,
            "actual_epoch_loop": "range(1, 100), i.e. epochs 1-99",
            "lr": 1e-4,
            "batchsize": 36,
            "trainsize": 352,
            "clip": 0.5,
            "decay_rate": 0.1,
            "decay_epoch": 50,
            "optimizer": "Adam",
            "validation_set": "CAMO",
        },
    }
    protocol_path = output_root / "protocol.json"
    with protocol_path.open("w", encoding="utf-8") as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return protocol


def file_stems(paths: Iterable[Path]) -> List[str]:
    return [path.stem for path in paths]


def audit_export(args: argparse.Namespace) -> Dict[str, Any]:
    output_root = args.output_root.resolve()
    train_root = output_root / "TrainDataset"
    image_paths = sorted((train_root / "Imgs").glob("*"))
    mask_paths = sorted((train_root / "PseudoMask").glob("*.png"))
    if len(image_paths) != args.expected_count or len(mask_paths) != args.expected_count:
        raise RuntimeError(
            f"Export count mismatch: images={len(image_paths)}, masks={len(mask_paths)}, "
            f"expected={args.expected_count}"
        )
    image_stems = file_stems(image_paths)
    mask_stems = file_stems(mask_paths)
    if image_stems != mask_stems:
        only_images = sorted(set(image_stems) - set(mask_stems))
        only_masks = sorted(set(mask_stems) - set(image_stems))
        raise RuntimeError(
            f"Image/mask identities differ: only_images={only_images[:10]}, "
            f"only_masks={only_masks[:10]}"
        )

    empty_count = 0
    full_count = 0
    ratios: List[float] = []
    for image_path, mask_path in zip(image_paths, mask_paths):
        if not image_path.is_symlink():
            raise RuntimeError(f"Training image is not an immutable symlink: {image_path}")
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != mask.size:
                raise RuntimeError(
                    f"Native size mismatch for {image_path.stem}: {image.size} vs {mask.size}"
                )
            mask_array = np.asarray(mask.convert("L"), dtype=np.uint8)
        unique = set(np.unique(mask_array).tolist())
        if not unique.issubset({0, 255}):
            raise RuntimeError(f"Mask is not binary: {mask_path}, unique={sorted(unique)[:10]}")
        foreground = mask_array == 255
        ratios.append(float(foreground.mean()))
        empty_count += int(not bool(foreground.any()))
        full_count += int(bool(foreground.all()))

    for output_name, expected in EXPECTED_TEST_COUNTS.items():
        dataset_root = output_root / "TestDataset" / output_name
        for leaf in ("Imgs", "GT"):
            link = dataset_root / leaf
            if not link.is_symlink() or not link.resolve().is_dir():
                raise RuntimeError(f"Invalid test protocol link: {link}")
            count = sum(path.is_file() for path in link.iterdir())
            if count != expected:
                raise RuntimeError(
                    f"{output_name}/{leaf} has {count} files, expected {expected}"
                )

    result = {
        "status": "ok",
        "num_samples": len(mask_paths),
        "empty_masks": empty_count,
        "full_masks": full_count,
        "foreground_ratio_mean": float(np.mean(ratios)),
        "output_root": str(output_root),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {args.threshold}")
    if args.torch_threads < 1:
        raise ValueError(f"torch-threads must be >= 1, got {args.torch_threads}")
    torch.set_num_threads(args.torch_threads)
    if not args.audit_only:
        protocol = export_masks(args)
        print(json.dumps(protocol, ensure_ascii=False, indent=2))
    audit_export(args)


if __name__ == "__main__":
    main()
