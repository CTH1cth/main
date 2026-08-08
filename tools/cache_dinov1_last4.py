"""Cache DINOv1-S/8 attention-key maps f9--f12 for R1-HSD v1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.cache_features import (  # noqa: E402
    call_dino,
    key_to_feature_tensor,
    load_dino,
    preprocess_image,
    resolve_key_projection_at_layer,
    split_datasets,
)
from common.utils import (  # noqa: E402
    build_image_items,
    ensure_dir,
    load_config,
    read_jsonl,
    torch_load,
    write_jsonl,
)


LAST4_INDICES_0BASED = (8, 9, 10, 11)
LAST4_KEYS = ("f9", "f10", "f11", "f12")
LAST4_CACHE_VERSION = "dinov1_s8_last4_key_v1"


def last4_cache_root(cfg, output_root=None) -> Path:
    root = output_root or getattr(
        cfg,
        "LAST4_FEATURE_CACHE_ROOT",
        "../datasets/cache/dinov1_s8_last4_296",
    )
    return Path(root).expanduser().resolve()


def final_feature_manifest(cfg, split: str) -> Path:
    cache_key = str(getattr(cfg, "FEATURE_CACHE_KEY", cfg.BACKBONE_KEY))
    return (
        Path(cfg.CACHE_ROOT)
        / "features_cache"
        / cache_key
        / f"manifest_{split}.jsonl"
    ).resolve()


def _manifest_map(path: Path) -> dict[tuple[str, str], dict]:
    rows = read_jsonl(path)
    mapping = {}
    for row in rows:
        key = (str(row.get("dataset")), str(row.get("stem")))
        if key in mapping:
            raise RuntimeError(f"Duplicate cache key {key} in {path}.")
        mapping[key] = row
    return mapping


def _reference_f12(row, dataset: str, stem: str) -> torch.Tensor:
    payload = torch_load(row["cache_path"], map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Existing final-layer feature cache must be dict: {row['cache_path']}")
    if payload.get("dataset") != dataset or payload.get("stem") != stem:
        raise RuntimeError(
            f"Existing final-layer cache identity mismatch for {dataset}/{stem}: "
            f"{row['cache_path']}"
        )
    tensor = payload.get("tensor")
    if not torch.is_tensor(tensor) or list(tensor.shape) != [384, 37, 37]:
        shape = list(tensor.shape) if torch.is_tensor(tensor) else None
        raise RuntimeError(
            f"Existing f12 reference must be [384,37,37], got {shape}: "
            f"{row['cache_path']}"
        )
    return tensor.detach().cpu().float()


def capture_last4(model, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run one frozen forward and return CPU float32 f9--f12 key maps."""

    holders = {key: None for key in LAST4_KEYS}
    handles = []

    def make_hook(key):
        def hook(_module, _inputs, output):
            holders[key] = output.detach()

        return hook

    for index, key in zip(LAST4_INDICES_0BASED, LAST4_KEYS):
        module, _ = resolve_key_projection_at_layer(model, index)
        handles.append(module.register_forward_hook(make_hook(key)))
    try:
        with torch.no_grad():
            call_dino(model, inputs)
    finally:
        for handle in handles:
            handle.remove()

    missing = [key for key, value in holders.items() if value is None]
    if missing:
        raise RuntimeError(f"DINO last-four hooks failed to capture: {missing}")
    features = {
        key: key_to_feature_tensor(holders[key]).float()
        for key in LAST4_KEYS
    }
    for key, tensor in features.items():
        if list(tensor.shape) != [384, 37, 37]:
            raise RuntimeError(
                f"DINOv1-S/8 {key} must be [384,37,37], got {list(tensor.shape)}."
            )
        if tensor.requires_grad or not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError(f"DINOv1-S/8 {key} must be finite and detached.")
    return features


def _validated_resume_row(payload, out_path: Path, item, cfg):
    """Validate one completed payload and rebuild its manifest row."""

    expected_metadata = {
        "version": LAST4_CACHE_VERSION,
        "dataset": item["dataset"],
        "stem": item["stem"],
        "backbone_key": str(cfg.BACKBONE_KEY),
        "model_key": str(cfg.DINO["model_name"]),
        "input_size": 296,
        "patch_size": 8,
        "resize_interpolation": "bicubic",
        "feature_type": "attention_key_projection",
        "layer_indices_0based": list(LAST4_INDICES_0BASED),
        "feature_keys": list(LAST4_KEYS),
        "dtype": "float32",
    }
    if not isinstance(payload, dict):
        raise TypeError("payload is not a dict")
    mismatched = {
        name: (payload.get(name), expected)
        for name, expected in expected_metadata.items()
        if payload.get(name) != expected
    }
    if mismatched:
        raise RuntimeError(f"metadata mismatch: {mismatched}")
    features = payload.get("features")
    if not isinstance(features, dict):
        raise RuntimeError("payload is missing the features dict")
    shapes = {}
    for key in LAST4_KEYS:
        tensor = features.get(key)
        if not torch.is_tensor(tensor):
            raise RuntimeError(f"payload is missing features/{key}")
        if tensor.dtype != torch.float32 or list(tensor.shape) != [384, 37, 37]:
            raise RuntimeError(
                f"features/{key} shape/dtype mismatch: {list(tensor.shape)}/{tensor.dtype}"
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise RuntimeError(f"features/{key} contains NaN/Inf")
        shapes[key] = list(tensor.shape)
    tensor = payload.get("tensor")
    if not torch.is_tensor(tensor) or not torch.equal(tensor, features["f12"]):
        raise RuntimeError("payload tensor does not exactly match features/f12")
    f12_error = float(payload.get("f12_max_abs_error", float("inf")))
    if not (0.0 <= f12_error < 1e-6):
        raise RuntimeError(f"invalid recorded f12_max_abs_error={f12_error}")
    return {
        "version": LAST4_CACHE_VERSION,
        "dataset": item["dataset"],
        "stem": item["stem"],
        "image_path": item["image_path"],
        "cache_path": str(out_path.resolve()),
        "backbone_key": str(cfg.BACKBONE_KEY),
        "feature_type": "attention_key_projection",
        "layer_indices_0based": list(LAST4_INDICES_0BASED),
        "layers": [index + 1 for index in LAST4_INDICES_0BASED],
        "feature_keys": list(LAST4_KEYS),
        "dtype": "float32",
        "shape": shapes,
        "f12_max_abs_error": f12_error,
    }


def generate_last4_cache(
    cfg,
    split: str,
    overwrite: bool = False,
    resume: bool = False,
    max_samples: int = -1,
    output_root=None,
    logger=print,
):
    if overwrite and resume:
        raise ValueError("--overwrite and --resume cannot be used together.")
    if str(cfg.BACKBONE_KEY) != "dinov1-s8":
        raise RuntimeError(
            f"R1-HSD v1 last-four cache requires BACKBONE_KEY='dinov1-s8', got {cfg.BACKBONE_KEY!r}."
        )
    if int(cfg.DINO["feature_input_size"]) != 296 or int(cfg.DINO["patch_size"]) != 8:
        raise RuntimeError("R1-HSD v1 cache requires DINOv1-S/8 input=296 and patch=8.")
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")

    root = last4_cache_root(cfg, output_root=output_root)
    split_root = root / split
    manifest_path = root / f"manifest_{split}.jsonl"
    ensure_dir(split_root)
    if manifest_path.exists() and not overwrite and not resume:
        raise FileExistsError(
            f"Manifest exists; pass --overwrite to regenerate: {manifest_path}"
        )

    reference_manifest = final_feature_manifest(cfg, split)
    if not reference_manifest.is_file():
        raise FileNotFoundError(
            f"Existing final-layer feature manifest not found: {reference_manifest}"
        )
    reference_map = _manifest_map(reference_manifest)

    items = build_image_items(
        cfg.DATA_ROOT, split_datasets(cfg, split), require_gt=False
    )
    if int(max_samples) >= 0:
        items = items[: int(max_samples)]
    missing_reference = [
        (item["dataset"], item["stem"])
        for item in items
        if (item["dataset"], item["stem"]) not in reference_map
    ]
    if missing_reference:
        raise RuntimeError(
            "Existing final-layer cache is missing samples: "
            f"{missing_reference[:10]}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dino(cfg, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()

    logger(f"version = {LAST4_CACHE_VERSION}")
    logger(f"backbone_key = {cfg.BACKBONE_KEY}")
    logger(f"model_path = {cfg.DINO['model_path']}")
    logger("feature_type = attention_key_projection")
    logger(f"layer_indices_0based = {list(LAST4_INDICES_0BASED)}")
    logger(f"feature_keys = {list(LAST4_KEYS)}")
    logger("feature_input_size = 296")
    logger("feature_resize_interpolation = bicubic")
    logger("feature_dtype = float32")
    logger("feature_postprocess = none")
    logger(f"f12_reference_manifest = {reference_manifest}")
    logger(f"output_root = {root}")
    logger(f"resume = {bool(resume)}")

    rows = []
    max_f12_error = 0.0
    resumed_valid = 0
    resumed_rebuilt = 0
    for item in tqdm(items, desc=f"cache DINOv1 last4 {split}"):
        dataset = item["dataset"]
        stem = item["stem"]
        out_dir = split_root / dataset
        ensure_dir(out_dir)
        out_path = out_dir / f"{stem}.pt"
        if out_path.exists() and resume:
            try:
                existing_payload = torch_load(out_path, map_location="cpu")
                existing_row = _validated_resume_row(
                    existing_payload, out_path, item, cfg
                )
            except Exception as exc:
                logger(
                    f"resume_rebuild = {out_path} | "
                    f"reason={type(exc).__name__}: {exc}"
                )
                resumed_rebuilt += 1
            else:
                rows.append(existing_row)
                max_f12_error = max(
                    max_f12_error,
                    float(existing_row["f12_max_abs_error"]),
                )
                resumed_valid += 1
                continue
        elif out_path.exists() and not overwrite:
            raise FileExistsError(
                f"Cache exists; pass --resume or --overwrite: {out_path}"
            )

        inputs, original_size = preprocess_image(
            item["image_path"], 296, interpolation="bicubic"
        )
        features = capture_last4(model, inputs.to(device))
        reference = _reference_f12(reference_map[(dataset, stem)], dataset, stem)
        f12_error = float((features["f12"] - reference).abs().max().item())
        if f12_error >= 1e-6:
            raise RuntimeError(
                f"f12 compatibility failed for {dataset}/{stem}: "
                f"max_abs_error={f12_error:.9g} must be < 1e-6."
            )
        max_f12_error = max(max_f12_error, f12_error)

        shapes = {key: list(features[key].shape) for key in LAST4_KEYS}
        payload = {
            "version": LAST4_CACHE_VERSION,
            "dataset": dataset,
            "stem": stem,
            "image_path": item["image_path"],
            "original_size": original_size,
            "backbone_key": str(cfg.BACKBONE_KEY),
            "model_key": str(cfg.DINO["model_name"]),
            "input_size": 296,
            "patch_size": 8,
            "resize_interpolation": "bicubic",
            "feature_type": "attention_key_projection",
            "layer_indices_0based": list(LAST4_INDICES_0BASED),
            "layer_indices_1based": [index + 1 for index in LAST4_INDICES_0BASED],
            "layers": [index + 1 for index in LAST4_INDICES_0BASED],
            "feature_keys": list(LAST4_KEYS),
            "dtype": "float32",
            "features": features,
            "tensor": features["f12"],
            "shape": shapes,
            "f12_reference_cache_path": reference_map[(dataset, stem)]["cache_path"],
            "f12_max_abs_error": f12_error,
        }
        temporary_path = out_path.with_suffix(out_path.suffix + ".tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(out_path)
        rows.append(
            {
                "version": LAST4_CACHE_VERSION,
                "dataset": dataset,
                "stem": stem,
                "image_path": item["image_path"],
                "cache_path": str(out_path.resolve()),
                "backbone_key": str(cfg.BACKBONE_KEY),
                "feature_type": "attention_key_projection",
                "layer_indices_0based": list(LAST4_INDICES_0BASED),
                "layers": [index + 1 for index in LAST4_INDICES_0BASED],
                "feature_keys": list(LAST4_KEYS),
                "dtype": "float32",
                "shape": shapes,
                "f12_max_abs_error": f12_error,
            }
        )

    write_jsonl(manifest_path, rows)
    logger(f"wrote_manifest = {manifest_path}")
    logger(f"num_items = {len(rows)}")
    logger(f"resume_valid_skipped = {resumed_valid}")
    logger(f"resume_invalid_rebuilt = {resumed_rebuilt}")
    logger(f"f12_max_abs_error = {max_f12_error:.9g}")
    return manifest_path, max_f12_error


def main():
    parser = argparse.ArgumentParser(
        description="Cache frozen DINOv1-S/8 f9--f12 attention-key maps."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    generate_last4_cache(
        cfg,
        split=args.split,
        overwrite=args.overwrite,
        resume=args.resume,
        max_samples=args.max_samples,
        output_root=args.output_root,
        logger=print,
    )


if __name__ == "__main__":
    main()
