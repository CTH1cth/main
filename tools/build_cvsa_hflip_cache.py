#!/usr/bin/env python3
"""Build independently generated hflip DINO and DABE-PU-v1.1 caches for CVSA."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cache_dabe_pseudo import _params_from_cfg, _tensor_payload  # noqa: E402
from common.cache_features import (  # noqa: E402
    call_dino,
    key_to_feature_tensor,
    load_dino,
    resolve_key_projection,
)
from common.dabe_pseudo import DABE_PU_V11_DEFAULT_PARAMS, generate_dabe_pseudo  # noqa: E402
from common.utils import (  # noqa: E402
    build_image_items,
    ensure_dir,
    load_config,
    manifest_to_map,
    read_jsonl,
    torch_load,
    write_jsonl,
)
from models.supervision.cvsa import CVSA_CACHE_VERSION  # noqa: E402


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
DEFAULT_CONFIG = (
    "configs/dinov1_s8_dabepu_v11_cvsa_v1_hflip_router_"
    "dagp_uncgate_ndr_long45_lrfloor_2e5.py"
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hflip_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transpose = getattr(Image, "Transpose", None)
    method = (
        Image.Transpose.FLIP_LEFT_RIGHT
        if transpose is not None
        else Image.FLIP_LEFT_RIGHT
    )
    return image.transpose(method)


def _preprocess_hflip(image_path, size):
    image = _hflip_image(image_path)
    original_size = (image.height, image.width)
    resized = image.resize((int(size), int(size)), Image.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.unsqueeze(0), original_size


def _selected_items(cfg, split, max_samples):
    if split != "train":
        raise RuntimeError("CVSA hflip cache is train-only.")
    items = build_image_items(
        cfg.DATA_ROOT, cfg.TRAIN_DATASETS, require_gt=False
    )
    if int(max_samples) >= 0:
        items = items[: int(max_samples)]
    if not items:
        raise RuntimeError("CVSA cache item list is empty.")
    return items


def _prepare_output_root(root, overwrite):
    root = Path(root).expanduser()
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise FileExistsError(
            f"CVSA output directory is not empty; pass --overwrite to replace "
            f"matching files: {root}"
        )
    ensure_dir(root)
    return root


def build_feature_cache(cfg, output_root, split, max_samples, overwrite, logger=print):
    if cfg.BACKBONE_KEY != "dinov1-s8":
        raise RuntimeError("CVSA-v1 currently requires BACKBONE_KEY=dinov1-s8.")
    root = _prepare_output_root(output_root, overwrite)
    manifest_path = root / f"manifest_{split}.jsonl"
    items = _selected_items(cfg, split, max_samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = int(cfg.DINO["feature_input_size"])
    model = load_dino(cfg, device)
    key_holder = {"tensor": None}

    def hook_fn(_module, _inputs, output):
        key_holder["tensor"] = output.detach()

    key_module, key_path = resolve_key_projection(model)
    handle = key_module.register_forward_hook(hook_fn)
    rows = []
    logger(f"[CVSA Cache] stage=feature | device={device}")
    logger(f"[CVSA Cache] output_root={root.resolve()}")
    logger("[CVSA Cache] call_chain=source_rgb->horizontal_flip->DINO_key")
    logger("[CVSA Cache] training_gt_used=False")
    try:
        for item in tqdm(items, desc="CVSA hflip DINO"):
            dataset, stem = item["dataset"], item["stem"]
            out_dir = root / dataset
            ensure_dir(out_dir)
            out_path = out_dir / f"{stem}.pt"
            if out_path.exists() and not overwrite:
                raise FileExistsError(f"CVSA feature exists: {out_path}")
            inputs, original_size = _preprocess_hflip(
                item["image_path"], input_size
            )
            key_holder["tensor"] = None
            with torch.no_grad():
                call_dino(model, inputs.to(device))
                captured = key_holder["tensor"]
                if captured is None:
                    raise RuntimeError("DINO key hook did not capture a tensor.")
                feature = key_to_feature_tensor(captured)
            if list(feature.shape) != [384, 37, 37]:
                raise RuntimeError(
                    f"CVSA DINO feature shape mismatch: {list(feature.shape)}"
                )
            source_checksum = _sha256_file(item["image_path"])
            payload = {
                "dataset": dataset,
                "stem": stem,
                "image_path": str(Path(item["image_path"]).resolve()),
                "source_image_path": str(Path(item["image_path"]).resolve()),
                "source_image_checksum": source_checksum,
                "original_size": original_size,
                "view": "hflip",
                "backbone": cfg.BACKBONE_KEY,
                "backbone_key": cfg.BACKBONE_KEY,
                "model_key": cfg.DINO["model_name"],
                "key_projection_path": key_path,
                "cache_version": CVSA_CACHE_VERSION,
                "independently_generated": True,
                "generation_call_chain": "source_rgb->horizontal_flip->DINO_key",
                "training_gt_used": False,
                "tensor": feature.contiguous(),
            }
            torch.save(payload, out_path)
            checksum = _sha256_file(out_path)
            rows.append(
                {
                    "dataset": dataset,
                    "stem": stem,
                    "image_path": payload["image_path"],
                    "source_image_path": payload["source_image_path"],
                    "source_image_checksum": source_checksum,
                    "cache_path": str(out_path.resolve()),
                    "feature_path": str(out_path.resolve()),
                    "view": "hflip",
                    "backbone": cfg.BACKBONE_KEY,
                    "shape": list(feature.shape),
                    "feature_shape": list(feature.shape),
                    "cache_version": CVSA_CACHE_VERSION,
                    "independently_generated": True,
                    "generation_call_chain": payload["generation_call_chain"],
                    "checksum": checksum,
                    "feature_checksum": checksum,
                    "training_gt_used": False,
                }
            )
    finally:
        handle.remove()
    write_jsonl(manifest_path, rows)
    logger(f"[CVSA Cache] manifest={manifest_path.resolve()} | rows={len(rows)}")
    return manifest_path


def _primitive_result_fields(result):
    output = {}
    for key, value in result.items():
        if torch.is_tensor(value):
            continue
        if value is None or isinstance(value, (str, bool, int, float, list, dict)):
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                continue
            output[key] = value
    return output


def build_dabe_cache(
    cfg,
    feature_root,
    output_root,
    split,
    max_samples,
    overwrite,
    logger=print,
):
    root = _prepare_output_root(output_root, overwrite)
    manifest_path = root / f"manifest_{split}.jsonl"
    feature_manifest = Path(feature_root) / f"manifest_{split}.jsonl"
    if not feature_manifest.is_file():
        raise FileNotFoundError(
            f"CVSA hflip feature manifest is missing: {feature_manifest}"
        )
    feature_map = manifest_to_map(read_jsonl(feature_manifest), feature_manifest)
    items = _selected_items(cfg, split, max_samples)
    params = {
        **DABE_PU_V11_DEFAULT_PARAMS,
        **_params_from_cfg(cfg),
        "VERSION": "pu_v11",
    }
    rows = []
    temp_root = root / ".cvsa_hflip_rgb_tmp"
    ensure_dir(temp_root)
    logger(f"[CVSA Cache] stage=dabe_pu | output_root={root.resolve()}")
    logger(
        "[CVSA Cache] call_chain=source_rgb->horizontal_flip + "
        "independent_hflip_DINO->DABE-PU-v1.1"
    )
    logger("[CVSA Cache] training_gt_used=False")
    try:
        for item in tqdm(items, desc="CVSA hflip DABE-PU"):
            dataset, stem = item["dataset"], item["stem"]
            key = (dataset, stem)
            if key not in feature_map:
                raise RuntimeError(f"Missing CVSA hflip feature for {key}.")
            feature_row = feature_map[key]
            feature_path = feature_row.get(
                "feature_path", feature_row.get("cache_path")
            )
            feature_payload = torch_load(feature_path, map_location="cpu")
            if not isinstance(feature_payload, dict):
                raise RuntimeError(f"Invalid CVSA feature payload: {feature_path}")
            provenance = {
                "view": "hflip",
                "backbone": cfg.BACKBONE_KEY,
                "cache_version": CVSA_CACHE_VERSION,
                "independently_generated": True,
            }
            for field, expected in provenance.items():
                actual = feature_row.get(field, feature_payload.get(field))
                if actual != expected:
                    raise RuntimeError(
                        f"CVSA feature provenance mismatch for {key}: "
                        f"{field}={actual!r} != {expected!r}."
                    )
            feature = feature_payload.get("tensor")
            if not torch.is_tensor(feature) or list(feature.shape) != [384, 37, 37]:
                raise RuntimeError(f"Invalid CVSA hflip feature shape for {key}.")
            expected_feature_checksum = feature_row.get("checksum")
            if expected_feature_checksum != _sha256_file(feature_path):
                raise RuntimeError(f"CVSA hflip feature checksum mismatch for {key}.")

            temp_dir = temp_root / dataset
            ensure_dir(temp_dir)
            temp_image_path = temp_dir / f"{stem}.png"
            _hflip_image(item["image_path"]).save(temp_image_path)
            try:
                with torch.no_grad():
                    result = generate_dabe_pseudo(
                        feature.float(),
                        temp_image_path,
                        params=params,
                        augs="identity,hflip,vflip,rot180",
                    )
            finally:
                temp_image_path.unlink(missing_ok=True)
            target = result.get("target_soft_68")
            if not torch.is_tensor(target) or list(target.shape) != [1, 68, 68]:
                raise RuntimeError(f"CVSA hflip DABE target shape mismatch for {key}.")
            out_dir = root / dataset
            ensure_dir(out_dir)
            out_path = out_dir / f"{stem}.pt"
            if out_path.exists() and not overwrite:
                raise FileExistsError(f"CVSA DABE-PU cache exists: {out_path}")
            source_checksum = _sha256_file(item["image_path"])
            feature_checksum = _sha256_file(feature_path)
            payload = {
                "dataset": dataset,
                "stem": stem,
                "image_path": str(Path(item["image_path"]).resolve()),
                "source_image_path": str(Path(item["image_path"]).resolve()),
                "source_image_checksum": source_checksum,
                "backbone_key": cfg.BACKBONE_KEY,
                "view": "hflip",
                "dabe_version": "pu_v11",
                "source_feature_view": "hflip",
                "source_feature_path": str(Path(feature_path).resolve()),
                "source_feature_checksum": feature_checksum,
                "cache_version": CVSA_CACHE_VERSION,
                "independently_generated": True,
                "generation_call_chain": (
                    "source_rgb->horizontal_flip + "
                    "independent_hflip_DINO->DABE-PU-v1.1"
                ),
                "training_gt_used": False,
                **_tensor_payload(result),
                **_primitive_result_fields(result),
            }
            torch.save(payload, out_path)
            checksum = _sha256_file(out_path)
            rows.append(
                {
                    "dataset": dataset,
                    "stem": stem,
                    "image_path": payload["image_path"],
                    "source_image_path": payload["source_image_path"],
                    "source_image_checksum": source_checksum,
                    "cache_path": str(out_path.resolve()),
                    "fixed_path": str(out_path.resolve()),
                    "view": "hflip",
                    "backbone": cfg.BACKBONE_KEY,
                    "backbone_key": cfg.BACKBONE_KEY,
                    "dabe_version": "pu_v11",
                    "source_feature_view": "hflip",
                    "source_feature_path": payload["source_feature_path"],
                    "source_feature_checksum": feature_checksum,
                    "shape": list(target.shape),
                    "cache_version": CVSA_CACHE_VERSION,
                    "independently_generated": True,
                    "generation_call_chain": payload["generation_call_chain"],
                    "checksum": checksum,
                    "fixed_checksum": checksum,
                    "training_gt_used": False,
                }
            )
    finally:
        for path in sorted(temp_root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        if temp_root.exists():
            temp_root.rmdir()
    write_jsonl(manifest_path, rows)
    logger(f"[CVSA Cache] manifest={manifest_path.resolve()} | rows={len(rows)}")
    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Build independent CVSA hflip DINO/DABE-PU caches."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("feature", "dabe_pu", "all"), required=True)
    parser.add_argument("--backbone", default="dinov1-s8", choices=("dinov1-s8",))
    parser.add_argument("--split", default="train", choices=("train",))
    parser.add_argument("--feature_root", default=None)
    parser.add_argument("--fixed_root", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_samples == 0 or args.max_samples < -1:
        raise ValueError("--max_samples must be -1 or a positive integer.")
    cfg = load_config(args.config)
    if cfg.BACKBONE_KEY != args.backbone:
        raise RuntimeError(
            f"Config/backbone mismatch: {cfg.BACKBONE_KEY} != {args.backbone}."
        )
    cvsa = dict(cfg.CVSA)
    feature_root = Path(
        args.feature_root or cvsa["feature_cache_hflip_root"]
    )
    fixed_root = Path(args.fixed_root or cvsa["fixed_cache_hflip_root"])
    if args.stage == "feature" and args.output_root:
        feature_root = Path(args.output_root)
    elif args.stage == "dabe_pu" and args.output_root:
        fixed_root = Path(args.output_root)
    elif args.stage == "all" and args.output_root:
        raise ValueError(
            "--stage all uses the two roots from CVSA config; do not pass --output_root."
        )
    if args.stage in {"feature", "all"}:
        build_feature_cache(
            cfg,
            feature_root,
            args.split,
            args.max_samples,
            args.overwrite,
        )
    if args.stage in {"dabe_pu", "all"}:
        build_dabe_cache(
            cfg,
            feature_root,
            fixed_root,
            args.split,
            args.max_samples,
            args.overwrite,
        )


if __name__ == "__main__":
    main()
