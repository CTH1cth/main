import argparse
import csv
import hashlib
import json
import math
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.dataset import CachedTrainDataset  # noqa: E402
from common.icar_probe_utils import (  # noqa: E402
    build_split_lookup,
    make_probe,
    prototype_features,
    self_support_indices,
    spearman,
    standardize_apply,
    standardize_fit,
    variant_columns,
)
from common.utils import load_config, set_seed, torch_load  # noqa: E402
from model import build_seg_head  # noqa: E402
from tools.probe_icar_separability import (  # noqa: E402
    RAW_DIM,
    build_raw_feature_rows,
    clone_state,
    evaluate_run,
    extract_checkpoint_samples,
    load_frozen_models,
    max_state_diff,
    output_logits,
    predict_logits,
    validate_config,
)
from train import (  # noqa: E402
    compute_dino_core_margin_68,
    extract_logits,
    make_image_68,
    make_model_input,
    resize_logits_for_loss,
    set_model_epoch,
)


DATASETS = ("TR-CAMO", "TR-COD10K")
LEARNED_SCORES = ("V4", "V6")
BASELINE_SCORES = ("esa_margin", "final_prob")
SUPPORTED_SCORES = LEARNED_SCORES + BASELINE_SCORES
PROBE_TYPE = "mlp"
BASELINE_SEED = -1
PROBE_FORMAT = "icar_sp_tinymlp_v1"
REQUIRED_REPORTS = (
    "summary.md",
    "summary.json",
    "temporal_results.csv",
    "per_image_results.csv",
    "seed_stability.csv",
    "baseline_comparison.csv",
    "probe_weights_manifest.json",
)


def parse_csv(value, cast=str):
    if value is None or not str(value).strip():
        return []
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "ICAR-ETVA: read-only temporal validity audit for V4/V6 ambiguity scores. "
            "Official student/teacher models are never updated."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--workdir", default="")
    parser.add_argument("--icar-probe-root", required=True)
    parser.add_argument("--source-epochs", default="15,20")
    parser.add_argument("--future-map", default="15:20,25,30,35;20:25,30,35")
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--max-images-per-source", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--probe-seeds", default="3407,3408,3409")
    parser.add_argument("--score-types", default="V4,V6,esa_margin,final_prob")
    parser.add_argument("--v6-support-seed", type=int, default=3407)
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def resolve_device(value):
    value = str(value).lower()
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    return device


def read_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required ICAR-SP artifact not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def report_safe(value):
    if isinstance(value, dict):
        return {key: report_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [report_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report_safe(payload), handle, indent=2, ensure_ascii=False, sort_keys=True)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([report_safe(row) for row in rows])


def read_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required ICAR-SP reference CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_future_map(value, source_epochs):
    parsed = {}
    for group in str(value).split(";"):
        group = group.strip()
        if not group:
            continue
        if ":" not in group:
            raise ValueError(f"Invalid future-map group without ':': {group!r}")
        source_text, future_text = group.split(":", 1)
        source = int(source_text.strip())
        if source in parsed:
            raise ValueError(f"Duplicate future-map source epoch: {source}")
        futures = parse_csv(future_text, int)
        if not futures:
            raise ValueError(f"No future epochs for source {source}")
        if len(set(futures)) != len(futures):
            raise ValueError(f"Duplicate future epoch for source {source}: {futures}")
        if any(epoch <= source for epoch in futures):
            raise ValueError(f"Future epochs must be greater than source {source}: {futures}")
        parsed[source] = futures
    if set(parsed) != set(source_epochs):
        raise ValueError(
            f"future-map sources {sorted(parsed)} do not match --source-epochs {sorted(source_epochs)}"
        )
    return parsed


def checkpoint_path(workdir, epoch):
    return Path(workdir) / "train" / "ckpt" / f"epoch_{int(epoch):03d}.pth"


def resolve_workdir(explicit_workdir, probe_summary):
    if explicit_workdir:
        return Path(explicit_workdir).expanduser().resolve()
    ckpt15 = Path(probe_summary.get("ckpt15", "")).expanduser()
    if not ckpt15.exists():
        raise RuntimeError(
            "--workdir was omitted and ICAR-SP summary.ckpt15 cannot be resolved. "
            "Pass the real Long35 workdir explicitly."
        )
    return ckpt15.resolve().parents[2]


def validate_checkpoint(path, cfg, epoch, require_teacher):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required checkpoint not found: {path}")
    checkpoint = torch_load(path, map_location="cpu")
    if int(checkpoint.get("epoch", -1)) != int(epoch):
        raise RuntimeError(f"Checkpoint epoch mismatch: {path} stores {checkpoint.get('epoch')}")
    if checkpoint.get("backbone_key") != cfg.BACKBONE_KEY:
        raise RuntimeError(
            f"Checkpoint backbone mismatch at epoch {epoch}: "
            f"{checkpoint.get('backbone_key')} != {cfg.BACKBONE_KEY}"
        )
    if "student" not in checkpoint or (require_teacher and "teacher" not in checkpoint):
        required = "student+teacher" if require_teacher else "student"
        raise RuntimeError(f"Checkpoint requires {required}: {path}")
    saved_cfg = checkpoint.get("config", {})
    if isinstance(saved_cfg, dict):
        for key in ("EXP_NAME", "HEAD_TYPE", "P_INIT_MODE", "DABE_PU_VERSION"):
            expected = getattr(cfg, key, None)
            if expected is not None and str(saved_cfg.get(key, "")) != str(expected):
                raise RuntimeError(
                    f"Checkpoint config mismatch at epoch {epoch}: "
                    f"{key}={saved_cfg.get(key)!r}, expected {expected!r}"
                )
    return checkpoint


def infer_in_channels(state):
    for key in ("base_head.weight", "proj.weight", "coarse_path.base_head.weight"):
        if key in state:
            return int(state[key].shape[1])
    raise KeyError("Cannot infer decoder input channels from checkpoint.")


def load_frozen_student(cfg, path, epoch, device):
    checkpoint = validate_checkpoint(path, cfg, epoch, require_teacher=False)
    student = build_seg_head(infer_in_channels(checkpoint["student"]), cfg).to(device)
    student.load_state_dict(checkpoint["student"], strict=True)
    set_model_epoch(student, int(epoch))
    student.eval()
    for parameter in student.parameters():
        parameter.requires_grad_(False)
    return student, checkpoint


def validate_probe_artifacts(probe_root, cfg, source_epochs, probe_seeds):
    probe_root = Path(probe_root).expanduser().resolve()
    summary_path = probe_root / "summary.json"
    split_path = probe_root / "split_manifest.json"
    norm_path = probe_root / "normalization_stats.json"
    summary = read_json(summary_path)
    split_manifest = read_json(split_path)
    normalization = read_json(norm_path)
    if Path(summary.get("config", "")).resolve() != Path(cfg.__file__).resolve():
        raise RuntimeError(
            f"ICAR-SP config mismatch: {summary.get('config')} != {Path(cfg.__file__).resolve()}"
        )
    if sorted(int(value) for value in summary.get("probe_seeds", [])) != [3407, 3408, 3409]:
        raise RuntimeError("ICAR-SP summary does not contain the required probe seeds 3407/3408/3409.")
    if sorted(int(value) for value in summary.get("support_split_seeds", [])) != [3407, 3408, 3409]:
        raise RuntimeError("ICAR-SP summary does not contain the original support split seeds.")
    keys = set()
    split_counts = defaultdict(int)
    dataset_counts = defaultdict(int)
    for row in split_manifest:
        key = (str(row.get("dataset")), str(row.get("stem")))
        if key in keys:
            raise RuntimeError(f"Duplicate ICAR-SP split row: {key}")
        if key[0] not in DATASETS or row.get("split") not in {"train", "val", "test"}:
            raise RuntimeError(f"Invalid ICAR-SP split row: {row}")
        keys.add(key)
        split_counts[str(row["split"])] += 1
        dataset_counts[key[0]] += 1
    if len(keys) != int(summary.get("num_selected_images", -1)):
        raise RuntimeError("ICAR-SP split row count does not match summary.num_selected_images.")
    for epoch in source_epochs:
        for variant in LEARNED_SCORES:
            expected_dim = 427 if variant == "V4" else 433
            for seed in probe_seeds:
                key = f"epoch{epoch}_{variant}_mlp_seed{seed}"
                item = normalization.get(key)
                if item is None:
                    raise RuntimeError(f"Missing ICAR-SP normalization key: {key}")
                if len(item.get("mean", [])) != expected_dim or len(item.get("std", [])) != expected_dim:
                    raise RuntimeError(f"Normalization dimension mismatch for {key}")
    return {
        "root": probe_root,
        "summary": summary,
        "split_manifest": split_manifest,
        "normalization": normalization,
        "summary_path": summary_path,
        "split_path": split_path,
        "norm_path": norm_path,
        "split_counts": dict(split_counts),
        "dataset_counts": dict(dataset_counts),
    }


def select_dataset_pairs(dataset, split_manifest, split_name=None):
    item_lookup = {
        (str(item["dataset"]), str(item["stem"])): (index, item)
        for index, item in enumerate(dataset.items)
        if str(item.get("dataset")) in DATASETS
    }
    selected = []
    for row in split_manifest:
        if split_name is not None and row["split"] != split_name:
            continue
        key = (str(row["dataset"]), str(row["stem"]))
        if key not in item_lookup:
            raise RuntimeError(f"ICAR-SP split sample missing from CachedTrainDataset: {key}")
        selected.append(item_lookup[key])
    selected.sort(key=lambda pair: (str(pair[1]["dataset"]), str(pair[1]["stem"])))
    return selected


def deterministic_subsample(rows, count, seed, dataset_name):
    rows = list(rows)
    if count < 0 or count >= len(rows):
        return rows
    rng = random.Random(int(seed) + int(hashlib.sha1(dataset_name.encode()).hexdigest()[:8], 16))
    rng.shuffle(rows)
    rows = rows[:count]
    rows.sort(key=lambda pair: str(pair[1]["stem"]))
    return rows


def select_audit_pairs(dataset, split_manifest, max_images, max_images_per_source, seed):
    test_pairs = select_dataset_pairs(dataset, split_manifest, split_name="test")
    by_source = {name: [] for name in DATASETS}
    for pair in test_pairs:
        by_source[str(pair[1]["dataset"])].append(pair)
    if int(max_images_per_source) > 0:
        requested = {name: int(max_images_per_source) for name in DATASETS}
    else:
        total = max(1, int(max_images))
        requested = {
            "TR-CAMO": min(300, max(1, int(round(total * 0.30)))),
            "TR-COD10K": min(700, max(1, total - int(round(total * 0.30)))),
        }
    selected = []
    counts = {}
    for name in DATASETS:
        count = min(requested[name], len(by_source[name]))
        chosen = deterministic_subsample(by_source[name], count, seed, name)
        selected.extend(chosen)
        counts[name] = len(chosen)
    selected.sort(key=lambda pair: (str(pair[1]["dataset"]), str(pair[1]["stem"])))
    return selected, counts


def train_summary_lookup(summary):
    lookup = {}
    for row in summary.get("train_summaries", []):
        key = (
            int(row.get("checkpoint_epoch", -1)),
            str(row.get("variant", "")).upper(),
            str(row.get("probe_type", "")).lower(),
            int(row.get("seed", -1)),
        )
        lookup[key] = row
    return lookup


def train_probe_to_saved_epoch(x_variant, y, meta, mean, std, seed, best_epoch, batch_size, device):
    set_seed(int(seed))
    train_mask = meta["split"] == "train"
    x_train = standardize_apply(x_variant[train_mask], mean, std)
    y_train = y[train_mask].astype(np.float32)
    model = make_probe(PROBE_TYPE, x_variant.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=False,
    )
    for _ in range(1, int(best_epoch) + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(xb).squeeze(1), yb)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite TinyMLP recovery loss.")
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def float_or_nan(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def validate_recovered_probe(epoch, variant, seed, model, x_variant, y, meta, mean, std, threshold, reference_rows, device):
    x_all = standardize_apply(x_variant, mean, std)
    logits = predict_logits(model, x_all, device)
    recovered, _ = evaluate_run(epoch, variant, PROBE_TYPE, seed, y, logits, meta, threshold)
    reference_lookup = {
        (
            int(row["checkpoint_epoch"]),
            str(row["variant"]),
            str(row["probe_type"]),
            int(row["seed"]),
            str(row["subset"]),
            str(row["source"]),
        ): row
        for row in reference_rows
    }
    diffs = []
    checked = 0
    for row in recovered:
        key = (
            int(epoch),
            variant,
            PROBE_TYPE,
            int(seed),
            str(row["subset"]),
            str(row["source"]),
        )
        reference = reference_lookup.get(key)
        if reference is None:
            continue
        for metric in ("micro_auroc", "macro_auroc"):
            lhs = float_or_nan(row.get(metric))
            rhs = float_or_nan(reference.get(metric))
            if math.isfinite(lhs) and math.isfinite(rhs):
                diffs.append(abs(lhs - rhs))
                checked += 1
    max_diff = max(diffs) if diffs else float("inf")
    if checked == 0 or max_diff > 1e-5:
        raise RuntimeError(
            f"Recovered probe does not reproduce ICAR-SP metrics: epoch={epoch}, "
            f"variant={variant}, seed={seed}, checked={checked}, max_diff={max_diff}"
        )
    return {"checked_metrics": checked, "max_auroc_abs_diff": max_diff}


def probe_artifact_path(weights_dir, epoch, variant, seed):
    return Path(weights_dir) / f"epoch{int(epoch):03d}_{variant}_mlp_seed{int(seed)}.pth"


def recover_missing_probes(cfg, dataset, artifact_info, checkpoint_paths, device, args):
    weights_dir = artifact_info["root"] / "probe_weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    all_seeds = [3407, 3408, 3409]
    required = [
        (epoch, variant, seed)
        for epoch in (15, 20)
        for variant in LEARNED_SCORES
        for seed in all_seeds
    ]
    missing = [key for key in required if not probe_artifact_path(weights_dir, *key).exists()]
    manifest_path = weights_dir / "manifest.json"
    manifest_rows = []
    if manifest_path.exists():
        manifest_rows = read_json(manifest_path).get("artifacts", [])
    if not missing:
        return weights_dir, manifest_rows, False

    print(
        f"[ICAR-ETVA] {len(missing)} probe weights are missing; "
        "starting one-time exact TinyMLP recovery.",
        flush=True,
    )
    split_lookup = build_split_lookup(artifact_info["split_manifest"])
    full_pairs = select_dataset_pairs(dataset, artifact_info["split_manifest"], split_name=None)
    reference_rows = read_csv(artifact_info["root"] / "metrics_by_source.csv")
    summaries = train_summary_lookup(artifact_info["summary"])
    recovery_args = SimpleNamespace(
        extract_batch_size=int(args.extract_batch_size),
        num_workers=int(args.num_workers),
    )
    existing = {
        (int(row["checkpoint_epoch"]), str(row["variant"]), int(row["probe_seed"])): row
        for row in manifest_rows
    }
    for epoch in (15, 20):
        epoch_missing = [key for key in missing if key[0] == epoch]
        if not epoch_missing:
            continue
        print(f"[ICAR-ETVA] recovering source feature rows at epoch {epoch}", flush=True)
        student, teacher, _ = load_frozen_models(cfg, checkpoint_paths[epoch], epoch, device)
        data = extract_checkpoint_samples(
            cfg,
            dataset,
            full_pairs,
            split_lookup,
            artifact_info["summary"]["support_split_seeds"],
            epoch,
            student,
            teacher,
            device,
            recovery_args,
        )
        if data["x"].shape[1] != RAW_DIM:
            raise RuntimeError(f"Recovered ICAR raw feature width mismatch: {data['x'].shape}")
        if max(data["state_diff"].values()) != 0.0:
            raise RuntimeError(f"Official model changed during probe recovery: {data['state_diff']}")
        y = data["meta"]["label"].astype(np.float32)
        for _, variant, seed in epoch_missing:
            key = f"epoch{epoch}_{variant}_mlp_seed{seed}"
            norm = artifact_info["normalization"][key]
            mean = np.asarray(norm["mean"], dtype=np.float32)
            std = np.asarray(norm["std"], dtype=np.float32)
            cols = variant_columns(variant)
            x_variant = data["x"][:, cols]
            calc_mean, calc_std = standardize_fit(x_variant[data["meta"]["split"] == "train"])
            norm_diff = max(
                float(np.max(np.abs(calc_mean - mean))),
                float(np.max(np.abs(calc_std - std))),
            )
            if norm_diff > 1e-6:
                raise RuntimeError(f"Saved normalization cannot be reproduced for {key}: {norm_diff}")
            train_info = summaries.get((epoch, variant, PROBE_TYPE, seed))
            if train_info is None:
                raise RuntimeError(f"Missing ICAR-SP train summary for {key}")
            print(
                f"[ICAR-ETVA] recover epoch={epoch} variant={variant} seed={seed} "
                f"best_epoch={train_info['best_epoch']}",
                flush=True,
            )
            model = train_probe_to_saved_epoch(
                x_variant,
                y,
                data["meta"],
                mean,
                std,
                seed,
                int(train_info["best_epoch"]),
                int(args.query_batch_size),
                device,
            )
            validation = validate_recovered_probe(
                epoch,
                variant,
                seed,
                model,
                x_variant,
                y,
                data["meta"],
                mean,
                std,
                float(train_info["threshold"]),
                reference_rows,
                device,
            )
            artifact_path = probe_artifact_path(weights_dir, epoch, variant, seed)
            torch.save(
                {
                    "format": PROBE_FORMAT,
                    "checkpoint_epoch": epoch,
                    "variant": variant,
                    "probe_type": PROBE_TYPE,
                    "probe_seed": seed,
                    "input_dim": len(cols),
                    "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                    "normalization_key": key,
                    "best_epoch": int(train_info["best_epoch"]),
                    "threshold": float(train_info["threshold"]),
                    "source_checkpoint": str(Path(checkpoint_paths[epoch]).resolve()),
                    "split_manifest_sha256": file_sha256(artifact_info["split_path"]),
                    "normalization_stats_sha256": file_sha256(artifact_info["norm_path"]),
                    "reproduction": validation,
                },
                artifact_path,
            )
            row = {
                "checkpoint_epoch": epoch,
                "variant": variant,
                "probe_seed": seed,
                "path": str(artifact_path.resolve()),
                "normalization_key": key,
                "best_epoch": int(train_info["best_epoch"]),
                **validation,
            }
            existing[(epoch, variant, seed)] = row
            del model
        del data, student, teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
    manifest_rows = [existing[key] for key in sorted(existing)]
    write_json(
        manifest_path,
        {
            "format": PROBE_FORMAT,
            "split_manifest_sha256": file_sha256(artifact_info["split_path"]),
            "normalization_stats_sha256": file_sha256(artifact_info["norm_path"]),
            "artifacts": manifest_rows,
        },
    )
    return weights_dir, manifest_rows, True


def load_probe_models(weights_dir, normalization, source_epochs, score_types, probe_seeds, device):
    models = {}
    norms = {}
    for epoch in source_epochs:
        for variant in score_types:
            if variant not in LEARNED_SCORES:
                continue
            for seed in probe_seeds:
                path = probe_artifact_path(weights_dir, epoch, variant, seed)
                if not path.exists():
                    raise FileNotFoundError(f"Recovered probe artifact missing: {path}")
                artifact = torch_load(path, map_location="cpu")
                expected = {
                    "format": PROBE_FORMAT,
                    "checkpoint_epoch": epoch,
                    "variant": variant,
                    "probe_type": PROBE_TYPE,
                    "probe_seed": seed,
                }
                for key, value in expected.items():
                    if artifact.get(key) != value:
                        raise RuntimeError(f"Probe artifact mismatch in {path}: {key}={artifact.get(key)!r}")
                model = make_probe(PROBE_TYPE, int(artifact["input_dim"])).to(device)
                model.load_state_dict(artifact["state_dict"], strict=True)
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                norm_key = artifact["normalization_key"]
                norm = normalization[norm_key]
                models[(epoch, variant, seed)] = model
                norms[(epoch, variant, seed)] = (
                    np.asarray(norm["mean"], dtype=np.float32),
                    np.asarray(norm["std"], dtype=np.float32),
                )
    return models, norms


def stable_rank_select(scores, indices, k):
    scores = np.asarray(scores, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if scores.shape[0] != indices.shape[0] or not np.isfinite(scores).all():
        raise RuntimeError("Candidate score/index mismatch or non-finite score.")
    order = np.lexsort((indices, scores))
    bottom = indices[order[:k]]
    top = indices[order[-k:]]
    if len(set(top.tolist()) & set(bottom.tolist())):
        raise RuntimeError("ICAR-ETVA top and bottom candidates overlap.")
    return np.sort(top), np.sort(bottom)


def score_probe(model, rows, mean, std, device):
    x = standardize_apply(rows, mean, std)
    return predict_logits(model, x, device).astype(np.float64)


def make_candidate(dataset_name, stem, source_epoch, score_type, probe_seed, audit_count, source_logits, scores, audit_idx):
    k = min(24, int(audit_count) // 4)
    if k < 4:
        return {
            "dataset": dataset_name,
            "stem": stem,
            "source_epoch": int(source_epoch),
            "score_type": score_type,
            "probe_seed": int(probe_seed),
            "audit_count": int(audit_count),
            "valid": False,
            "invalid_reason": "k_lt4",
        }
    top_idx, bottom_idx = stable_rank_select(scores, audit_idx, k)
    flat_logits = source_logits.reshape(-1).astype(np.float64)
    source_top_logits = flat_logits[top_idx]
    source_bottom_logits = flat_logits[bottom_idx]
    source_top_prob = 1.0 / (1.0 + np.exp(-np.clip(source_top_logits, -80.0, 80.0)))
    source_bottom_prob = 1.0 / (1.0 + np.exp(-np.clip(source_bottom_logits, -80.0, 80.0)))
    return {
        "dataset": dataset_name,
        "stem": stem,
        "source_epoch": int(source_epoch),
        "score_type": score_type,
        "probe_seed": int(probe_seed),
        "audit_count": int(audit_count),
        "valid": True,
        "invalid_reason": "",
        "k": int(k),
        "top_idx": top_idx,
        "bottom_idx": bottom_idx,
        "source_top_logits": source_top_logits,
        "source_bottom_logits": source_bottom_logits,
        "source_top_prob": source_top_prob,
        "source_bottom_prob": source_bottom_prob,
        "top_coord_sha1": hashlib.sha1(top_idx.tobytes()).hexdigest(),
        "bottom_coord_sha1": hashlib.sha1(bottom_idx.tobytes()).hexdigest(),
    }


def build_source_candidates(
    cfg,
    dataset,
    selected_pairs,
    checkpoint_paths,
    source_epochs,
    score_types,
    probe_seeds,
    probe_models,
    probe_norms,
    v6_support_seed,
    device,
    args,
):
    loader = DataLoader(
        Subset(dataset, [index for index, _ in selected_pairs]),
        batch_size=int(args.inference_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    all_candidates = {}
    score_cache = {}
    state_diffs = {}
    source_metadata = {}
    for epoch in source_epochs:
        print(f"[ICAR-ETVA] extracting source candidates at epoch {epoch}", flush=True)
        student, teacher, checkpoint = load_frozen_models(cfg, checkpoint_paths[epoch], epoch, device)
        before = {"student": clone_state(student), "teacher": clone_state(teacher)}
        source_metadata[epoch] = {
            "checkpoint": str(Path(checkpoint_paths[epoch]).resolve()),
            "teacher_source": "checkpoint_teacher",
            "student_teacher_max_abs_diff": max_state_diff(checkpoint["student"], checkpoint["teacher"]),
            "epoch20_reset_applied": False,
        }
        with torch.no_grad():
            for batch in loader:
                if any(str(key).lower().startswith("gt") for key in batch):
                    raise RuntimeError(f"GT-like field leaked into ICAR-ETVA batch: {sorted(batch)}")
                model_input = make_model_input(cfg, batch, device)
                image_68 = make_image_68(cfg, batch, device)
                output = student(model_input, image_68=image_68, return_aux=True, return_probe_aux=True)
                teacher_output = teacher(model_input, image_68=image_68, return_aux=False)
                if "probe_ndr_detail_feat" not in output:
                    raise RuntimeError("Source model did not return probe_ndr_detail_feat.")
                final_logits = resize_logits_for_loss(extract_logits(output), cfg).detach()
                teacher_logits = resize_logits_for_loss(extract_logits(teacher_output), cfg).detach()
                base_logits = output_logits(output, "base_logits", cfg).detach()
                coarse_logits = output_logits(
                    output, "coarse_logits_68", cfg, fallback=output.get("coarse_logits")
                ).detach()
                final_prob = final_logits.sigmoid()
                teacher_prob = teacher_logits.sigmoid()
                coarse_prob = coarse_logits.sigmoid()
                coarse_uncert = torch.clamp(1.0 - 2.0 * (coarse_prob - 0.5).abs(), 0.0, 1.0)
                final_uncert = torch.clamp(1.0 - 2.0 * (final_prob - 0.5).abs(), 0.0, 1.0)
                fg_core_68 = batch["pu_fg_core"].to(device, non_blocking=True).float()
                bg_core_68 = batch["pu_bg_core"].to(device, non_blocking=True).float()
                esa_margin, _ = compute_dino_core_margin_68(
                    cfg,
                    batch,
                    fg_core_68,
                    bg_core_68,
                    (int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)),
                    device,
                    prefix="ESA",
                )
                dino_68 = F.normalize(
                    F.interpolate(
                        batch["feature"].to(device, non_blocking=True).float(),
                        size=(int(cfg.LOSS_SIZE), int(cfg.LOSS_SIZE)),
                        mode="bilinear",
                        align_corners=False,
                    ),
                    dim=1,
                )
                detail_68 = F.normalize(output["probe_ndr_detail_feat"].detach().float(), dim=1)
                sobel_68 = output["sobel_68"].detach().float()
                v0_maps = torch.cat(
                    [
                        base_logits,
                        coarse_logits,
                        final_logits,
                        coarse_prob,
                        final_prob,
                        coarse_uncert,
                        final_uncert,
                    ],
                    dim=1,
                ).detach()
                for image_index, (dataset_name, stem) in enumerate(zip(batch["dataset"], batch["stem"])):
                    dataset_name = str(dataset_name)
                    stem = str(stem)
                    fg_core = batch["pu_fg_core"][image_index, 0].numpy() > 0.5
                    bg_core = (batch["pu_bg_core"][image_index, 0].numpy() > 0.5) & (~fg_core)
                    extent = batch["pu_extent"][image_index, 0].numpy() > 0.5
                    unknown = batch["pu_unknown"][image_index, 0].numpy() > 0.5
                    teacher_prob_np = teacher_prob[image_index, 0].cpu().numpy()
                    audit_mask = extent & (teacher_prob_np < 0.5) & (~fg_core) & (~bg_core) & (~unknown)
                    audit_idx = np.flatnonzero(audit_mask.reshape(-1)).astype(np.int64)
                    final_logits_np = final_logits[image_index, 0].cpu().numpy()
                    final_prob_np = final_prob[image_index, 0].cpu().numpy()
                    esa_np = esa_margin[image_index, 0].detach().cpu().numpy()
                    if audit_idx.size:
                        v0_np = v0_maps[image_index].cpu().numpy()
                        dino_np = dino_68[image_index].cpu().numpy()
                        detail_np = detail_68[image_index].cpu().numpy()
                        image_np = image_68[image_index].detach().cpu().numpy()
                        sobel_np = sobel_68[image_index].cpu().numpy()
                        zero_support = np.zeros((audit_idx.size, 6), dtype=np.float32)
                        self_support = zero_support
                        v6_reason = ""
                        if "V6" in score_types:
                            self_fg, self_bg, _, _, self_reason = self_support_indices(
                                final_prob_np,
                                audit_mask,
                                dataset_name,
                                stem,
                                int(v6_support_seed),
                            )
                            if self_fg is None or self_bg is None:
                                v6_reason = self_reason or "invalid_self_support"
                            else:
                                self_support = np.concatenate(
                                    [
                                        prototype_features(dino_np, self_fg, self_bg, audit_idx),
                                        prototype_features(detail_np, self_fg, self_bg, audit_idx),
                                    ],
                                    axis=1,
                                )
                        raw_rows = build_raw_feature_rows(
                            v0_np,
                            esa_np[None, ...],
                            dino_np,
                            detail_np,
                            image_np,
                            sobel_np,
                            zero_support,
                            self_support,
                            audit_idx,
                        )
                    else:
                        raw_rows = np.zeros((0, RAW_DIM), dtype=np.float32)
                        v6_reason = "audit_region_empty"
                    methods = []
                    if "V4" in score_types:
                        for seed in probe_seeds:
                            mean, std = probe_norms[(epoch, "V4", seed)]
                            scores = score_probe(
                                probe_models[(epoch, "V4", seed)],
                                raw_rows[:, variant_columns("V4")],
                                mean,
                                std,
                                device,
                            ) if audit_idx.size else np.zeros((0,), dtype=np.float64)
                            methods.append(("V4", seed, scores, ""))
                    if "V6" in score_types:
                        for seed in probe_seeds:
                            mean, std = probe_norms[(epoch, "V6", seed)]
                            scores = score_probe(
                                probe_models[(epoch, "V6", seed)],
                                raw_rows[:, variant_columns("V6")],
                                mean,
                                std,
                                device,
                            ) if audit_idx.size and not v6_reason else np.zeros((0,), dtype=np.float64)
                            methods.append(("V6", seed, scores, v6_reason))
                    if "esa_margin" in score_types:
                        methods.append(("esa_margin", BASELINE_SEED, esa_np.reshape(-1)[audit_idx], ""))
                    if "final_prob" in score_types:
                        methods.append(("final_prob", BASELINE_SEED, final_prob_np.reshape(-1)[audit_idx], ""))
                    for score_type, probe_seed, scores, invalid_reason in methods:
                        key = (epoch, dataset_name, stem, score_type, int(probe_seed))
                        if invalid_reason:
                            candidate = {
                                "dataset": dataset_name,
                                "stem": stem,
                                "source_epoch": epoch,
                                "score_type": score_type,
                                "probe_seed": int(probe_seed),
                                "audit_count": int(audit_idx.size),
                                "valid": False,
                                "invalid_reason": invalid_reason,
                            }
                        else:
                            candidate = make_candidate(
                                dataset_name,
                                stem,
                                epoch,
                                score_type,
                                int(probe_seed),
                                int(audit_idx.size),
                                final_logits_np,
                                scores,
                                audit_idx,
                            )
                        all_candidates[key] = candidate
                        if score_type in LEARNED_SCORES and not invalid_reason and audit_idx.size:
                            score_cache[key] = {
                                "audit_idx": audit_idx.copy(),
                                "scores": np.asarray(scores, dtype=np.float64).copy(),
                                "top_idx": candidate.get("top_idx"),
                                "bottom_idx": candidate.get("bottom_idx"),
                                "valid": bool(candidate.get("valid")),
                            }
        after = {"student": clone_state(student), "teacher": clone_state(teacher)}
        state_diffs[epoch] = {
            role: max_state_diff(before[role], after[role]) for role in ("student", "teacher")
        }
        if max(state_diffs[epoch].values()) != 0.0:
            raise RuntimeError(f"Official model state changed during source audit: {state_diffs[epoch]}")
        del student, teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return all_candidates, score_cache, state_diffs, source_metadata


def temporal_row(candidate, future_epoch, future_logits):
    base = {
        "source_epoch": int(candidate["source_epoch"]),
        "future_epoch": int(future_epoch),
        "source_dataset": candidate["dataset"],
        "stem": candidate["stem"],
        "score_type": candidate["score_type"],
        "probe_seed": int(candidate["probe_seed"]),
        "audit_count": int(candidate["audit_count"]),
        "valid": int(bool(candidate.get("valid"))),
        "invalid_reason": candidate.get("invalid_reason", ""),
    }
    if not candidate.get("valid"):
        return {
            **base,
            "k": 0,
            "selected_pairs": 0,
            "pair_order_correct_count": 0,
            "pair_order_flip_count": 0,
            "top_future_fg_count": 0,
            "bottom_future_fg_count": 0,
        }
    flat = np.asarray(future_logits, dtype=np.float64).reshape(-1)
    top_idx = candidate["top_idx"]
    bottom_idx = candidate["bottom_idx"]
    future_top_logits = flat[top_idx]
    future_bottom_logits = flat[bottom_idx]
    future_top_prob = 1.0 / (1.0 + np.exp(-np.clip(future_top_logits, -80.0, 80.0)))
    future_bottom_prob = 1.0 / (1.0 + np.exp(-np.clip(future_bottom_logits, -80.0, 80.0)))
    source_top_logits = candidate["source_top_logits"]
    source_bottom_logits = candidate["source_bottom_logits"]
    source_top_prob = candidate["source_top_prob"]
    source_bottom_prob = candidate["source_bottom_prob"]
    source_pair = source_top_logits[:, None] - source_bottom_logits[None, :]
    future_pair = future_top_logits[:, None] - future_bottom_logits[None, :]
    correct = int((future_pair > 0.0).sum())
    flip = int(((source_pair * future_pair) < 0.0).sum())
    k = int(candidate["k"])
    source_prob_gap = float(source_top_prob.mean() - source_bottom_prob.mean())
    future_prob_gap = float(future_top_prob.mean() - future_bottom_prob.mean())
    source_logit_gap = float(source_top_logits.mean() - source_bottom_logits.mean())
    future_logit_gap = float(future_top_logits.mean() - future_bottom_logits.mean())
    return {
        **base,
        "k": k,
        "selected_pairs": k * k,
        "top_coord_sha1": candidate["top_coord_sha1"],
        "bottom_coord_sha1": candidate["bottom_coord_sha1"],
        "source_top_prob": float(source_top_prob.mean()),
        "source_bottom_prob": float(source_bottom_prob.mean()),
        "source_prob_gap": source_prob_gap,
        "future_top_prob": float(future_top_prob.mean()),
        "future_bottom_prob": float(future_bottom_prob.mean()),
        "future_prob_gap": future_prob_gap,
        "source_logit_gap": source_logit_gap,
        "future_logit_gap": future_logit_gap,
        "delta_prob_gap": future_prob_gap - source_prob_gap,
        "delta_logit_gap": future_logit_gap - source_logit_gap,
        "top_future_fg_count": int((future_top_prob >= 0.5).sum()),
        "bottom_future_fg_count": int((future_bottom_prob >= 0.5).sum()),
        "top_future_fg_ratio": float((future_top_prob >= 0.5).mean()),
        "bottom_future_fg_ratio": float((future_bottom_prob >= 0.5).mean()),
        "future_fg_ratio_gap": float(
            (future_top_prob >= 0.5).mean() - (future_bottom_prob >= 0.5).mean()
        ),
        "pair_order_correct_count": correct,
        "pair_order_flip_count": flip,
        "pair_order_correct_ratio": correct / max(k * k, 1),
        "pair_order_flip_ratio": flip / max(k * k, 1),
    }


def track_future_predictions(cfg, dataset, selected_pairs, checkpoint_paths, future_map, candidates, device, args):
    loader = DataLoader(
        Subset(dataset, [index for index, _ in selected_pairs]),
        batch_size=int(args.inference_batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
    )
    needed_futures = sorted({epoch for epochs in future_map.values() for epoch in epochs})
    rows = []
    state_diffs = {}
    future_metadata = {}
    candidates_by_image = defaultdict(list)
    for key, candidate in candidates.items():
        source_epoch, dataset_name, stem, _, _ = key
        for future_epoch in future_map[source_epoch]:
            candidates_by_image[(future_epoch, dataset_name, stem)].append(candidate)
    for epoch in needed_futures:
        print(f"[ICAR-ETVA] reading fixed coordinates from future epoch {epoch}", flush=True)
        student, checkpoint = load_frozen_student(cfg, checkpoint_paths[epoch], epoch, device)
        before = clone_state(student)
        future_metadata[epoch] = {"checkpoint": str(Path(checkpoint_paths[epoch]).resolve())}
        with torch.no_grad():
            for batch in loader:
                if any(str(key).lower().startswith("gt") for key in batch):
                    raise RuntimeError(f"GT-like field leaked into future batch: {sorted(batch)}")
                model_input = make_model_input(cfg, batch, device)
                image_68 = make_image_68(cfg, batch, device)
                output = student(model_input, image_68=image_68, return_aux=False)
                logits = resize_logits_for_loss(extract_logits(output), cfg).detach().cpu().numpy()
                for image_index, (dataset_name, stem) in enumerate(zip(batch["dataset"], batch["stem"])):
                    for candidate in candidates_by_image.get((epoch, str(dataset_name), str(stem)), []):
                        rows.append(temporal_row(candidate, epoch, logits[image_index, 0]))
        after = clone_state(student)
        state_diffs[epoch] = max_state_diff(before, after)
        if state_diffs[epoch] != 0.0:
            raise RuntimeError(f"Future student state changed at epoch {epoch}: {state_diffs[epoch]}")
        del student, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows, state_diffs, future_metadata


def weighted_mean(rows, value_key, weight_key):
    valid = [
        row
        for row in rows
        if value_key in row and math.isfinite(float(row[value_key])) and float(row.get(weight_key, 0)) > 0
    ]
    total = sum(float(row[weight_key]) for row in valid)
    if total <= 0:
        return float("nan")
    return sum(float(row[value_key]) * float(row[weight_key]) for row in valid) / total


def aggregate_group(rows, source_epoch, future_epoch, dataset_name, score_type, probe_seed):
    selected = [
        row
        for row in rows
        if int(row["source_epoch"]) == int(source_epoch)
        and int(row["future_epoch"]) == int(future_epoch)
        and row["score_type"] == score_type
        and int(row["probe_seed"]) == int(probe_seed)
        and (dataset_name == "Mixed" or row["source_dataset"] == dataset_name)
    ]
    if not selected:
        raise RuntimeError(
            f"No temporal rows for {source_epoch}->{future_epoch}, {dataset_name}, {score_type}, {probe_seed}"
        )
    valid = [row for row in selected if int(row["valid"]) == 1]
    top_count = sum(int(row.get("k", 0)) for row in valid)
    bottom_count = top_count
    pair_count = sum(int(row.get("selected_pairs", 0)) for row in valid)
    top_fg = sum(int(row.get("top_future_fg_count", 0)) for row in valid)
    bottom_fg = sum(int(row.get("bottom_future_fg_count", 0)) for row in valid)
    source_top = weighted_mean(valid, "source_top_prob", "k")
    source_bottom = weighted_mean(valid, "source_bottom_prob", "k")
    future_top = weighted_mean(valid, "future_top_prob", "k")
    future_bottom = weighted_mean(valid, "future_bottom_prob", "k")
    return {
        "source_epoch": int(source_epoch),
        "future_epoch": int(future_epoch),
        "source_dataset": dataset_name,
        "score_type": score_type,
        "probe_seed": int(probe_seed),
        "num_images": len(selected),
        "valid_images": len(valid),
        "valid_image_ratio": len(valid) / max(len(selected), 1),
        "selected_top": top_count,
        "selected_bottom": bottom_count,
        "selected_pairs": pair_count,
        "source_top_prob": source_top,
        "source_bottom_prob": source_bottom,
        "source_prob_gap": source_top - source_bottom if math.isfinite(source_top + source_bottom) else float("nan"),
        "future_top_prob": future_top,
        "future_bottom_prob": future_bottom,
        "future_prob_gap": future_top - future_bottom if math.isfinite(future_top + future_bottom) else float("nan"),
        "source_logit_gap": weighted_mean(valid, "source_logit_gap", "k"),
        "future_logit_gap": weighted_mean(valid, "future_logit_gap", "k"),
        "delta_prob_gap": weighted_mean(valid, "delta_prob_gap", "k"),
        "delta_logit_gap": weighted_mean(valid, "delta_logit_gap", "k"),
        "future_top_fg_ratio": top_fg / max(top_count, 1),
        "future_bottom_fg_ratio": bottom_fg / max(bottom_count, 1),
        "future_fg_ratio_gap": top_fg / max(top_count, 1) - bottom_fg / max(bottom_count, 1),
        "pair_order_correct_ratio": sum(int(row.get("pair_order_correct_count", 0)) for row in valid)
        / max(pair_count, 1),
        "pair_order_flip_ratio": sum(int(row.get("pair_order_flip_count", 0)) for row in valid)
        / max(pair_count, 1),
    }


def build_temporal_summary(rows, source_epochs, future_map, score_types, probe_seeds):
    out = []
    for source_epoch in source_epochs:
        for future_epoch in future_map[source_epoch]:
            for dataset_name in (*DATASETS, "Mixed"):
                for score_type in score_types:
                    seeds = probe_seeds if score_type in LEARNED_SCORES else [BASELINE_SEED]
                    for seed in seeds:
                        out.append(
                            aggregate_group(
                                rows,
                                source_epoch,
                                future_epoch,
                                dataset_name,
                                score_type,
                                seed,
                            )
                        )
    return out


def jaccard(a, b):
    a = set(np.asarray(a, dtype=np.int64).tolist())
    b = set(np.asarray(b, dtype=np.int64).tolist())
    union = len(a | b)
    return len(a & b) / union if union else float("nan")


def finite_mean(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(values)) if values else float("nan")


def build_seed_stability(score_cache, source_epochs, probe_seeds):
    per_image = []
    if len(probe_seeds) >= 2:
        seed_pairs = [
            (probe_seeds[i], probe_seeds[j])
            for i in range(len(probe_seeds))
            for j in range(i + 1, len(probe_seeds))
        ]
    else:
        seed_pairs = []
    images = sorted({(key[0], key[1], key[2], key[3]) for key in score_cache})
    for source_epoch, dataset_name, stem, variant in images:
        if variant not in LEARNED_SCORES:
            continue
        values = []
        for seed_a, seed_b in seed_pairs:
            a = score_cache.get((source_epoch, dataset_name, stem, variant, seed_a))
            b = score_cache.get((source_epoch, dataset_name, stem, variant, seed_b))
            if not a or not b or not a["valid"] or not b["valid"]:
                continue
            if not np.array_equal(a["audit_idx"], b["audit_idx"]):
                raise RuntimeError("Probe seeds do not share identical audit coordinates.")
            values.append(
                (
                    jaccard(a["top_idx"], b["top_idx"]),
                    jaccard(a["bottom_idx"], b["bottom_idx"]),
                    spearman(a["scores"], b["scores"]),
                )
            )
        per_image.append(
            {
                "row_type": "per_image",
                "source_epoch": int(source_epoch),
                "source_dataset": dataset_name,
                "stem": stem,
                "score_type": variant,
                "valid_seed_pairs": len(values),
                "top_jaccard": finite_mean([value[0] for value in values]),
                "bottom_jaccard": finite_mean([value[1] for value in values]),
                "seed_score_spearman": finite_mean([value[2] for value in values]),
            }
        )
    aggregate = []
    for source_epoch in source_epochs:
        for variant in LEARNED_SCORES:
            for dataset_name in (*DATASETS, "Mixed"):
                selected = [
                    row
                    for row in per_image
                    if int(row["source_epoch"]) == int(source_epoch)
                    and row["score_type"] == variant
                    and (dataset_name == "Mixed" or row["source_dataset"] == dataset_name)
                    and int(row["valid_seed_pairs"]) > 0
                ]
                aggregate.append(
                    {
                        "row_type": "aggregate",
                        "source_epoch": int(source_epoch),
                        "source_dataset": dataset_name,
                        "stem": "",
                        "score_type": variant,
                        "valid_images": len(selected),
                        "valid_seed_pairs": sum(int(row["valid_seed_pairs"]) for row in selected),
                        "top_jaccard": finite_mean([row["top_jaccard"] for row in selected]),
                        "bottom_jaccard": finite_mean([row["bottom_jaccard"] for row in selected]),
                        "seed_score_spearman": finite_mean(
                            [row["seed_score_spearman"] for row in selected]
                        ),
                    }
                )
    return per_image + aggregate, aggregate


def attach_stability(temporal_rows, stability_aggregate):
    lookup = {
        (int(row["source_epoch"]), row["source_dataset"], row["score_type"]): row
        for row in stability_aggregate
    }
    for row in temporal_rows:
        if row["score_type"] not in LEARNED_SCORES:
            row.update(
                {"top_jaccard": float("nan"), "bottom_jaccard": float("nan"), "seed_score_spearman": float("nan")}
            )
            continue
        stat = lookup[(int(row["source_epoch"]), row["source_dataset"], row["score_type"])]
        for key in ("top_jaccard", "bottom_jaccard", "seed_score_spearman"):
            row[key] = stat[key]


def build_baseline_comparison(temporal_rows):
    lookup = {
        (
            int(row["source_epoch"]),
            int(row["future_epoch"]),
            row["source_dataset"],
            row["score_type"],
            int(row["probe_seed"]),
        ): row
        for row in temporal_rows
    }
    comparisons = []
    for row in temporal_rows:
        if row["score_type"] not in LEARNED_SCORES:
            continue
        common = (int(row["source_epoch"]), int(row["future_epoch"]), row["source_dataset"])
        esa = lookup.get((*common, "esa_margin", BASELINE_SEED))
        final = lookup.get((*common, "final_prob", BASELINE_SEED))
        if esa is None or final is None:
            raise RuntimeError("Both ESA and final_prob baselines are required for learned-score comparison.")
        comparison = {
            **{key: row[key] for key in ("source_epoch", "future_epoch", "source_dataset", "score_type", "probe_seed")},
            "correct_gain_vs_esa": row["pair_order_correct_ratio"] - esa["pair_order_correct_ratio"],
            "correct_gain_vs_final_prob": row["pair_order_correct_ratio"] - final["pair_order_correct_ratio"],
            "gap_gain_vs_esa": row["future_prob_gap"] - esa["future_prob_gap"],
            "gap_gain_vs_final_prob": row["future_prob_gap"] - final["future_prob_gap"],
            "learned_correct_ratio": row["pair_order_correct_ratio"],
            "esa_correct_ratio": esa["pair_order_correct_ratio"],
            "final_prob_correct_ratio": final["pair_order_correct_ratio"],
            "learned_future_prob_gap": row["future_prob_gap"],
            "esa_future_prob_gap": esa["future_prob_gap"],
            "final_prob_future_prob_gap": final["future_prob_gap"],
            "selected_pairs": row["selected_pairs"],
        }
        comparisons.append(comparison)
        row.update(
            {key: comparison[key] for key in ("correct_gain_vs_esa", "correct_gain_vs_final_prob", "gap_gain_vs_esa", "gap_gain_vs_final_prob")}
        )
    return comparisons


def weighted_summary(rows, key, weight="selected_pairs"):
    valid = [row for row in rows if math.isfinite(float(row[key])) and float(row.get(weight, 0)) > 0]
    total = sum(float(row[weight]) for row in valid)
    return (
        sum(float(row[key]) * float(row[weight]) for row in valid) / total
        if total > 0
        else float("nan")
    )


def classify_variant(variant, dataset_name, temporal_rows, comparisons, probe_seeds):
    rows = [
        row
        for row in temporal_rows
        if row["score_type"] == variant and row["source_dataset"] == dataset_name
    ]
    comp = [
        row
        for row in comparisons
        if row["score_type"] == variant and row["source_dataset"] == dataset_name
    ]
    relation_groups = defaultdict(list)
    for row in rows:
        relation_groups[(int(row["source_epoch"]), int(row["future_epoch"]))].append(row)
    relation_gap = []
    relation_delta = []
    seed_stds = []
    for group in relation_groups.values():
        relation_gap.append(float(np.mean([float(row["future_prob_gap"]) for row in group])))
        relation_delta.append(float(np.mean([float(row["delta_prob_gap"]) for row in group])))
        if len(group) == len(probe_seeds):
            seed_stds.append(float(np.std([float(row["pair_order_correct_ratio"]) for row in group])))
    top_jaccard = finite_mean([row["top_jaccard"] for row in rows])
    bottom_jaccard = finite_mean([row["bottom_jaccard"] for row in rows])
    result = {
        "pair_order_correct_ratio": weighted_summary(rows, "pair_order_correct_ratio"),
        "future_prob_gap": weighted_summary(rows, "future_prob_gap", "selected_top"),
        "future_fg_ratio_gap": weighted_summary(rows, "future_fg_ratio_gap", "selected_top"),
        "min_relation_future_gap": min(relation_gap) if relation_gap else float("nan"),
        "min_relation_delta_prob_gap": min(relation_delta) if relation_delta else float("nan"),
        "correct_gain_vs_esa": weighted_summary(comp, "correct_gain_vs_esa"),
        "correct_gain_vs_final_prob": weighted_summary(comp, "correct_gain_vs_final_prob"),
        "top_jaccard": top_jaccard,
        "bottom_jaccard": bottom_jaccard,
        "max_seed_correct_std": max(seed_stds) if seed_stds else float("nan"),
    }
    result["passed"] = all(
        (
            result["pair_order_correct_ratio"] >= 0.65,
            result["future_prob_gap"] >= 0.05,
            result["future_fg_ratio_gap"] >= 0.20,
            result["min_relation_future_gap"] > 0.0,
            result["min_relation_delta_prob_gap"] >= -0.01,
            result["correct_gain_vs_esa"] >= 0.05,
            result["correct_gain_vs_final_prob"] >= 0.05,
            result["top_jaccard"] >= 0.50,
            result["bottom_jaccard"] >= 0.50,
            result["max_seed_correct_std"] <= 0.03,
        )
    )
    return result


def automatic_verdict(temporal_rows, comparisons, probe_seeds, full_protocol):
    checks = {
        variant: {
            dataset_name: classify_variant(variant, dataset_name, temporal_rows, comparisons, probe_seeds)
            for dataset_name in DATASETS
        }
        for variant in LEARNED_SCORES
    }
    if not full_protocol:
        return {
            "class": "SANITY_ONLY",
            "message": "Sanity run completed; E1-E4 requires the full three-seed/full-future protocol.",
            "checks": checks,
        }
    variant_pass = {
        variant: all(checks[variant][dataset]["passed"] for dataset in DATASETS)
        for variant in LEARNED_SCORES
    }
    if variant_pass["V4"] and not variant_pass["V6"]:
        code = "E2"
        message = (
            "E2: Local semantic-detail resolver is valid, but SELF support adds no reliable information. "
            "Use a simplified V4-style resolver; do not include SELF-support prototypes."
        )
    elif any(variant_pass.values()):
        code = "E1"
        message = (
            "E1: Learned ambiguity score has usable temporal validity. "
            "A controlled full resolver experiment is justified."
        )
    else:
        camo_pass = any(checks[variant]["TR-CAMO"]["passed"] for variant in LEARNED_SCORES)
        cod_bad = all(
            checks[variant]["TR-COD10K"]["pair_order_correct_ratio"] < 0.60
            or checks[variant]["TR-COD10K"]["future_prob_gap"] <= 0.0
            or checks[variant]["TR-COD10K"]["correct_gain_vs_esa"] <= 0.0
            for variant in LEARNED_SCORES
        )
        if camo_pass and cod_bad:
            code = "E3"
            message = (
                "E3: Resolver is source-biased and will likely reproduce the "
                "CHAMELEON/CAMO-COD10K trade-off. Stop the ICAR route."
            )
        else:
            code = "E4"
            message = (
                "E4: Core separability does not transfer to ambiguous extent. "
                "Stop ICAR with the current feature source."
            )
    return {"class": code, "message": message, "variant_pass": variant_pass, "checks": checks}


def write_reports(out_dir, metadata, temporal_rows, per_image_rows, seed_rows, comparisons, verdict, probe_manifest):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "temporal_results.csv", temporal_rows)
    write_csv(out_dir / "per_image_results.csv", per_image_rows)
    write_csv(out_dir / "seed_stability.csv", seed_rows)
    write_csv(out_dir / "baseline_comparison.csv", comparisons)
    write_json(out_dir / "probe_weights_manifest.json", probe_manifest)
    summary = {
        "metadata": metadata,
        "verdict": verdict,
        "temporal_results": temporal_rows,
        "semantic_warning": (
            "Future model prediction is only a temporal-consistency proxy, not semantic ground truth. "
            "Passing ETVA supports ranking stability, but does not prove that top-score pixels are true object pixels."
        ),
    }
    write_json(out_dir / "summary.json", summary)
    lines = [
        "# ICAR-ETVA Temporal Validity Audit",
        "",
        f"Verdict: **{verdict['class']}**",
        "",
        verdict["message"],
        "",
        "## Audit State",
        "",
        f"- Config: `{metadata['config']}`",
        f"- Workdir: `{metadata['workdir']}`",
        f"- ICAR probe root: `{metadata['icar_probe_root']}`",
        f"- Images: `{metadata['num_images']}` "
        f"(TR-CAMO={metadata['image_counts']['TR-CAMO']}, "
        f"TR-COD10K={metadata['image_counts']['TR-COD10K']})",
        f"- Probe seeds: `{metadata['probe_seeds']}`",
        f"- V6 support seed: `{metadata['v6_support_seed']}`",
        f"- Probe weights rebuilt: `{metadata['probe_weights_rebuilt']}`",
        "- Teacher source: `checkpoint_teacher` for source epochs",
        "- Future source: `checkpoint_student` at fixed source coordinates",
        "- GT read: `False`",
        "- Official model optimizer/backward/EMA update: `False`",
        "",
        "## Temporal Results",
        "",
        "| source→future | dataset | score | seed | valid | correct | flip | future gap | FG gap |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in temporal_rows:
        lines.append(
            f"| {row['source_epoch']}→{row['future_epoch']} | {row['source_dataset']} | "
            f"{row['score_type']} | {row['probe_seed']} | {row['valid_image_ratio']:.4f} | "
            f"{row['pair_order_correct_ratio']:.4f} | {row['pair_order_flip_ratio']:.4f} | "
            f"{row['future_prob_gap']:.4f} | {row['future_fg_ratio_gap']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Limit",
            "",
            "Future model prediction is only a temporal-consistency proxy, not semantic ground truth.",
            "",
            "Passing ETVA supports ranking stability, but does not prove that top-score pixels are true object pixels.",
            "",
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    zip_path = out_dir / "icar_etva.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in REQUIRED_REPORTS:
            archive.write(out_dir / name, arcname=name)
    return summary, zip_path


def main():
    args = parse_args()
    cfg = load_config(args.config)
    validate_config(cfg)
    source_epochs = parse_csv(args.source_epochs, int)
    if source_epochs != sorted(set(source_epochs)) or any(epoch not in (15, 20) for epoch in source_epochs):
        raise RuntimeError("ICAR-ETVA source epochs must be an ordered subset of 15,20.")
    future_map = parse_future_map(args.future_map, source_epochs)
    probe_seeds = parse_csv(args.probe_seeds, int)
    if not probe_seeds or any(seed not in (3407, 3408, 3409) for seed in probe_seeds):
        raise RuntimeError("Probe seeds must be a non-empty subset of 3407,3408,3409.")
    score_types = []
    for value in parse_csv(args.score_types):
        canonical = value.upper() if value.upper() in LEARNED_SCORES else value.lower()
        if canonical not in SUPPORTED_SCORES:
            raise RuntimeError(f"Unsupported score type: {value}")
        if canonical not in score_types:
            score_types.append(canonical)
    if not all(score in score_types for score in BASELINE_SCORES):
        raise RuntimeError("ICAR-ETVA requires both esa_margin and final_prob baselines.")
    if not all(score in score_types for score in LEARNED_SCORES):
        raise RuntimeError("ICAR-ETVA requires both V4 and V6 learned scores.")
    device = resolve_device(args.device)
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_info = validate_probe_artifacts(args.icar_probe_root, cfg, (15, 20), (3407, 3408, 3409))
    workdir = resolve_workdir(args.workdir, artifact_info["summary"])
    all_epochs = sorted(set(source_epochs) | {epoch for values in future_map.values() for epoch in values})
    checkpoint_paths = {epoch: checkpoint_path(workdir, epoch) for epoch in all_epochs}
    for epoch in all_epochs:
        validate_checkpoint(checkpoint_paths[epoch], cfg, epoch, require_teacher=epoch in source_epochs)
    for epoch, summary_key in ((15, "ckpt15"), (20, "ckpt20")):
        if epoch in source_epochs:
            summary_path = Path(artifact_info["summary"][summary_key]).resolve()
            if summary_path != Path(checkpoint_paths[epoch]).resolve():
                raise RuntimeError(
                    f"ICAR-SP source checkpoint mismatch at epoch {epoch}: "
                    f"{summary_path} != {Path(checkpoint_paths[epoch]).resolve()}"
                )

    dataset = CachedTrainDataset(cfg, max_samples=-1)
    selected_pairs, image_counts = select_audit_pairs(
        dataset,
        artifact_info["split_manifest"],
        args.max_images,
        args.max_images_per_source,
        args.seed,
    )
    if not selected_pairs:
        raise RuntimeError("No ICAR-SP test-split images selected for audit.")
    print(f"device = {device}", flush=True)
    print(f"workdir = {workdir}", flush=True)
    print(f"audit_images = {len(selected_pairs)} | counts = {image_counts}", flush=True)

    weights_dir, probe_manifest_rows, rebuilt = recover_missing_probes(
        cfg, dataset, artifact_info, {15: checkpoint_path(workdir, 15), 20: checkpoint_path(workdir, 20)}, device, args
    )
    probe_models, probe_norms = load_probe_models(
        weights_dir,
        artifact_info["normalization"],
        source_epochs,
        score_types,
        probe_seeds,
        device,
    )
    candidates, score_cache, source_state_diffs, source_metadata = build_source_candidates(
        cfg,
        dataset,
        selected_pairs,
        checkpoint_paths,
        source_epochs,
        score_types,
        probe_seeds,
        probe_models,
        probe_norms,
        args.v6_support_seed,
        device,
        args,
    )
    per_image_rows, future_state_diffs, future_metadata = track_future_predictions(
        cfg,
        dataset,
        selected_pairs,
        checkpoint_paths,
        future_map,
        candidates,
        device,
        args,
    )
    expected_rows = len(selected_pairs) * sum(len(future_map[epoch]) for epoch in source_epochs) * (
        len([score for score in score_types if score in BASELINE_SCORES])
        + len([score for score in score_types if score in LEARNED_SCORES]) * len(probe_seeds)
    )
    if len(per_image_rows) != expected_rows:
        raise RuntimeError(f"Temporal row count mismatch: {len(per_image_rows)} != {expected_rows}")
    temporal_rows = build_temporal_summary(
        per_image_rows, source_epochs, future_map, score_types, probe_seeds
    )
    seed_rows, seed_aggregate = build_seed_stability(score_cache, source_epochs, probe_seeds)
    attach_stability(temporal_rows, seed_aggregate)
    comparisons = build_baseline_comparison(temporal_rows)
    full_protocol = (
        source_epochs == [15, 20]
        and future_map == {15: [20, 25, 30, 35], 20: [25, 30, 35]}
        and probe_seeds == [3407, 3408, 3409]
        and image_counts == {"TR-CAMO": 200, "TR-COD10K": 608}
    )
    verdict = automatic_verdict(temporal_rows, comparisons, probe_seeds, full_protocol)
    probe_manifest = {
        "weights_dir": str(weights_dir.resolve()),
        "rebuilt_this_run": bool(rebuilt),
        "artifacts": probe_manifest_rows,
    }
    metadata = {
        "config": str(Path(args.config).resolve()),
        "workdir": str(workdir),
        "icar_probe_root": str(artifact_info["root"]),
        "source_epochs": source_epochs,
        "future_map": future_map,
        "checkpoint_paths": {str(key): str(value.resolve()) for key, value in checkpoint_paths.items()},
        "source_checkpoint_metadata": source_metadata,
        "future_checkpoint_metadata": future_metadata,
        "num_images": len(selected_pairs),
        "image_counts": image_counts,
        "probe_seeds": probe_seeds,
        "score_types": score_types,
        "v6_support_seed": int(args.v6_support_seed),
        "probe_weights_rebuilt": bool(rebuilt),
        "probe_recovery_note": (
            "Any optimizer/backward used by this script was restricted to TinyMLP probe recovery. "
            "Official student/teacher models remained frozen."
        ),
        "source_model_state_max_diff": source_state_diffs,
        "future_student_state_max_diff": future_state_diffs,
        "gt_read": False,
        "official_model_optimizer_or_backward": False,
        "ema_update": False,
        "full_protocol": full_protocol,
    }
    _, zip_path = write_reports(
        out_dir,
        metadata,
        temporal_rows,
        per_image_rows,
        seed_rows,
        comparisons,
        verdict,
        probe_manifest,
    )
    print(f"verdict = {verdict['class']} | {verdict['message']}", flush=True)
    print(f"report_zip = {zip_path}", flush=True)


if __name__ == "__main__":
    main()
