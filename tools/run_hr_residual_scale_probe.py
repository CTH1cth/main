import argparse
import csv
import json
import math
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.dataset import CachedEvalDataset  # noqa: E402
from common.metrics import CODMetrics  # noqa: E402
from common.utils import ensure_cache_available, ensure_dir, load_config, torch_load  # noqa: E402
from eval import (  # noqa: E402
    dataloader_worker_kwargs,
    infer_in_channels,
    make_image_136,
    make_image_68,
    make_model_input,
)
from model import build_seg_head  # noqa: E402


METRIC_FIELDS = [
    "scale",
    "dataset",
    "S_m",
    "F_beta_w",
    "F_beta_m",
    "E_phi_m",
    "MAE",
    "fallback_ratio",
    "valid_img_ratio",
    "band_ratio_mean",
    "hr_active_pixel_ratio",
    "mean_abs_logit_delta",
    "mean_abs_prob_delta",
    "binary_flip_ratio",
    "flip_0_to_1_ratio",
    "flip_1_to_0_ratio",
    "flip_ratio_inside_band",
    "flip_ratio_outside_band",
]

FLIP_FIELDS = [
    "scale",
    "dataset",
    "num_images",
    "fallback_ratio",
    "valid_img_ratio",
    "band_ratio_mean",
    "hr_active_pixel_ratio",
    "mean_abs_logit_delta",
    "mean_abs_prob_delta",
    "binary_flip_ratio",
    "flip_0_to_1_ratio",
    "flip_1_to_0_ratio",
    "flip_ratio_inside_band",
    "flip_ratio_outside_band",
    "scale1_reproduction_max_abs_error",
]


def parse_csv_list(value, cast=str):
    values = [item.strip() for item in str(value).split(",") if item.strip()]
    if not values:
        raise ValueError("Comma-separated argument must not be empty.")
    return [cast(item) for item in values]


def parse_args():
    parser = argparse.ArgumentParser(description="Eval-only HR-BFR residual scale probe.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--scales", required=True, help="Comma-separated non-negative scales.")
    parser.add_argument("--datasets", required=True, help="Comma-separated test dataset names.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Keep 1 to reproduce eval.py numerics exactly; larger values are diagnostic-only.",
    )
    parser.add_argument("--metric-workers", type=int, default=0)
    return parser.parse_args()


def validate_args(args, scales):
    config_path = Path(args.config).expanduser()
    ckpt_path = Path(args.ckpt).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if int(args.max_samples) == 0 or int(args.max_samples) < -1:
        raise ValueError("--max-samples must be -1 or a positive integer.")
    if int(args.batch_size) <= 0:
        raise ValueError("--batch-size must be positive.")
    if int(args.metric_workers) < 0:
        raise ValueError("--metric-workers must be non-negative.")
    if len(set(scales)) != len(scales):
        raise ValueError(f"Duplicate scales are not allowed: {scales}")
    for scale in scales:
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError(f"Scale must be finite and non-negative, got {scale}")


def write_csv(path, rows, fields):
    ensure_dir(Path(path).parent)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_student(cfg, ckpt_path, device):
    checkpoint = torch_load(ckpt_path, map_location="cpu")
    if checkpoint.get("backbone_key") != cfg.BACKBONE_KEY:
        raise RuntimeError(
            f"Checkpoint backbone mismatch: {checkpoint.get('backbone_key')} != {cfg.BACKBONE_KEY}"
        )
    if "student" not in checkpoint:
        raise KeyError(f"Checkpoint has no student state: {ckpt_path}")
    state = checkpoint["student"]
    student = build_seg_head(infer_in_channels(state), cfg).to(device)
    student.load_state_dict(state, strict=True)
    if not hasattr(student, "set_hr_eval_res_scale"):
        raise RuntimeError("Configured model does not expose HR-BFR eval residual scale control.")
    source_epoch = int(checkpoint.get("epoch", getattr(cfg, "MAX_EPOCH", 35)))
    if hasattr(student, "set_epoch"):
        student.set_epoch(source_epoch)
    student.eval()
    return student, checkpoint, source_epoch


def safe_ratio(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def variable_gt_collate(samples):
    # Test GT keeps its native resolution and therefore cannot be stacked. All
    # model inputs are fixed-size tensors and still use the standard collator.
    batch = {"gt": [sample["gt"] for sample in samples]}
    for key in samples[0]:
        if key == "gt":
            continue
        batch[key] = default_collate([sample[key] for sample in samples])
    return batch


def update_metric_from_pairs(metric, pairs):
    for gt, pred in pairs:
        metric.step(gt, pred)


class ProbeStats:
    def __init__(self):
        self.images = 0
        self.pixels = 0
        self.inside_pixels = 0
        self.outside_pixels = 0
        self.valid_images = 0.0
        self.fallback_images = 0.0
        self.band_ratio_sum = 0.0
        self.active_ratio_sum = 0.0
        self.abs_logit_delta_sum = 0.0
        self.abs_prob_delta_sum = 0.0
        self.flip_pixels = 0
        self.flip_0_to_1_pixels = 0
        self.flip_1_to_0_pixels = 0
        self.flip_inside_pixels = 0
        self.flip_outside_pixels = 0
        self.scale1_reproduction_max_abs_error = 0.0

    def update(self, output, scale):
        required = {
            "hr_logits",
            "hr_anchor_logits",
            "hr_residual_logits",
            "hr_band_gate_136",
            "hr_band_gate_raw_136",
            "hr_valid_img_mask",
            "hr_beta_eff",
        }
        missing = sorted(required - set(output))
        if missing:
            raise KeyError(f"HR-BFR output missing probe fields: {missing}")

        scaled_logits = output["hr_logits"].detach()
        anchor_logits = output["hr_anchor_logits"].detach()
        residual_logits = output["hr_residual_logits"].detach()
        band = output["hr_band_gate_136"].detach() > 0.5
        raw_band = output["hr_band_gate_raw_136"].detach() > 0.5
        valid = output["hr_valid_img_mask"].detach().reshape(-1)
        if scaled_logits.shape != anchor_logits.shape or band.shape != scaled_logits.shape:
            raise RuntimeError(
                "Probe shape mismatch: "
                f"scaled={list(scaled_logits.shape)} anchor={list(anchor_logits.shape)} "
                f"band={list(band.shape)}"
            )

        batch_size = int(scaled_logits.shape[0])
        pixels = int(scaled_logits.numel())
        delta = (scaled_logits - anchor_logits).abs()
        anchor_prob = torch.sigmoid(anchor_logits)
        scaled_prob = torch.sigmoid(scaled_logits)
        prob_delta = (scaled_prob - anchor_prob).abs()
        anchor_bin = anchor_prob >= 0.5
        scaled_bin = scaled_prob >= 0.5
        flip = anchor_bin != scaled_bin
        flip_0_to_1 = (~anchor_bin) & scaled_bin
        flip_1_to_0 = anchor_bin & (~scaled_bin)
        outside = ~band

        if float(scale) == 0.0 and not torch.equal(scaled_logits, anchor_logits):
            raise RuntimeError("scale=0.0 must produce hr_logits exactly equal to anchor logits.")
        invalid = valid <= 0.5
        if bool(invalid.any()) and not torch.equal(scaled_logits[invalid], anchor_logits[invalid]):
            raise RuntimeError("Invalid HR-BFR image did not fall back exactly to anchor logits.")
        if float(scale) == 1.0:
            beta = output["hr_beta_eff"].detach().to(dtype=anchor_logits.dtype)
            expected = anchor_logits + beta * band.to(dtype=anchor_logits.dtype) * residual_logits
            error = float((scaled_logits - expected).abs().max().item())
            self.scale1_reproduction_max_abs_error = max(
                self.scale1_reproduction_max_abs_error,
                error,
            )
            if error > 1e-7:
                raise RuntimeError(f"scale=1.0 failed to reproduce HR-BFR formula: max error={error:.12g}")

        self.images += batch_size
        self.pixels += pixels
        self.inside_pixels += int(band.sum().item())
        self.outside_pixels += int(outside.sum().item())
        self.valid_images += float(valid.sum().item())
        self.fallback_images += float(batch_size - valid.sum().item())
        self.band_ratio_sum += float(raw_band.flatten(1).float().mean(dim=1).sum().item())
        self.active_ratio_sum += float(band.flatten(1).float().mean(dim=1).sum().item())
        self.abs_logit_delta_sum += float(delta.sum().item())
        self.abs_prob_delta_sum += float(prob_delta.sum().item())
        self.flip_pixels += int(flip.sum().item())
        self.flip_0_to_1_pixels += int(flip_0_to_1.sum().item())
        self.flip_1_to_0_pixels += int(flip_1_to_0.sum().item())
        self.flip_inside_pixels += int((flip & band).sum().item())
        self.flip_outside_pixels += int((flip & outside).sum().item())

    def result(self):
        row = {
            "num_images": int(self.images),
            "fallback_ratio": safe_ratio(self.fallback_images, self.images),
            "valid_img_ratio": safe_ratio(self.valid_images, self.images),
            "band_ratio_mean": safe_ratio(self.band_ratio_sum, self.images),
            "hr_active_pixel_ratio": safe_ratio(self.active_ratio_sum, self.images),
            "mean_abs_logit_delta": safe_ratio(self.abs_logit_delta_sum, self.pixels),
            "mean_abs_prob_delta": safe_ratio(self.abs_prob_delta_sum, self.pixels),
            "binary_flip_ratio": safe_ratio(self.flip_pixels, self.pixels),
            "flip_0_to_1_ratio": safe_ratio(self.flip_0_to_1_pixels, self.pixels),
            "flip_1_to_0_ratio": safe_ratio(self.flip_1_to_0_pixels, self.pixels),
            "flip_ratio_inside_band": safe_ratio(self.flip_inside_pixels, self.inside_pixels),
            "flip_ratio_outside_band": safe_ratio(self.flip_outside_pixels, self.outside_pixels),
            "scale1_reproduction_max_abs_error": float(self.scale1_reproduction_max_abs_error),
        }
        if row["flip_ratio_outside_band"] > 1e-8:
            raise RuntimeError(
                "HR-BFR changed binary predictions outside the effective band: "
                f"ratio={row['flip_ratio_outside_band']:.12g}"
            )
        return row


def metric_values(result):
    return {
        "S_m": float(result["SMeasure"]),
        "F_beta_w": float(result["WFM"]),
        "F_beta_m": float(result["F_MEAN"]),
        "E_phi_m": float(result["E_MEAN"]),
        "MAE": float(result["MAE"]),
    }


def compose_scaled_hr_logits(output, scale):
    anchor = output["hr_anchor_logits"]
    if float(scale) == 0.0:
        return anchor
    beta = output["hr_beta_eff"].to(dtype=anchor.dtype)
    band = output["hr_band_gate_136"].to(dtype=anchor.dtype)
    residual = output["hr_residual_logits"].to(dtype=anchor.dtype)
    return anchor + anchor.new_tensor(float(scale)) * beta * band * residual


@torch.inference_mode()
def evaluate_scales_dataset(cfg, student, loader, device, scales):
    # Everything before the final residual fusion is scale-independent. Run the
    # checkpoint once per image, then apply the exact eval formula for each scale.
    student.set_hr_eval_res_scale(1.0)
    student.eval()
    metrics = {float(scale): CODMetrics() for scale in scales}
    stats = {float(scale): ProbeStats() for scale in scales}
    threshold = float(cfg.THRESHOLD)
    if abs(threshold - 0.5) > 1e-12:
        raise RuntimeError(f"Probe requires the unchanged threshold 0.5, config has {threshold}")

    workers = int(getattr(loader, "probe_metric_workers", 0))
    if workers <= 0:
        workers = min(len(scales), max(1, int(os.cpu_count() or 1)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for batch in loader:
            gt_list = [gt.float() for gt in batch["gt"]]
            model_input = make_model_input(cfg, batch, device)
            image_68 = make_image_68(cfg, batch, device)
            image_136 = make_image_136(cfg, batch, device)
            output = student(
                model_input,
                image_68=image_68,
                image_136=image_136,
                return_aux=False,
            )
            if not isinstance(output, dict):
                raise RuntimeError("HR residual probe requires dict model output.")
            metric_futures = []
            for scale in scales:
                scale = float(scale)
                scaled_output = dict(output)
                scaled_output["hr_logits"] = compose_scaled_hr_logits(output, scale)
                scaled_output["hr_eval_res_scale"] = scaled_output["hr_logits"].new_tensor(scale)
                stats[scale].update(scaled_output, scale)
                pairs = []
                for index, gt in enumerate(gt_list):
                    logits = F.interpolate(
                        scaled_output["hr_logits"][index : index + 1],
                        size=gt.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    pred = (torch.sigmoid(logits) > threshold).float().cpu()
                    pairs.append((gt.unsqueeze(0), pred))
                metric_futures.append(executor.submit(update_metric_from_pairs, metrics[scale], pairs))
            for future in metric_futures:
                future.result()

    return {
        scale: (metric_values(metrics[scale].get_result()), stats[scale].result())
        for scale in metrics
    }


def signed_metric_delta(row, base):
    values = []
    for key in ("S_m", "F_beta_w", "F_beta_m", "E_phi_m"):
        values.append(float(row[key]) - float(base[key]))
    values.append(float(base["MAE"]) - float(row["MAE"]))
    return values


def diagnose(rows):
    lookup = {(float(row["scale"]), row["dataset"]): row for row in rows}
    datasets = sorted({row["dataset"] for row in rows})
    high_scales = sorted({float(row["scale"]) for row in rows if float(row["scale"]) > 1.0})
    if not high_scales or not all((1.0, dataset) in lookup for dataset in datasets):
        return (
            "C",
            "The learned residual does not contain sufficiently useful boundary information.\n"
            "Move to HR-BFR-v2 signed boundary displacement.",
            "Probe lacks scale=1.0 or a scale above 1.0; the conservative C classification is used.",
        )

    scale_scores = {}
    for scale in high_scales:
        deltas = []
        for dataset in datasets:
            if (scale, dataset) in lookup:
                deltas.extend(signed_metric_delta(lookup[(scale, dataset)], lookup[(1.0, dataset)]))
        if deltas:
            scale_scores[scale] = sum(deltas) / len(deltas)
    required = {"CHAMELEON", "TE-CAMO", "TE-COD10K", "NC4K"}
    all_required = required.issubset(datasets)

    def scale_detail(scale):
        per_dataset = {}
        favorable = 0
        harmful = 0
        total = 0
        improve_counts = {}
        decline_counts = {}
        for dataset in datasets:
            if (scale, dataset) not in lookup:
                continue
            deltas = signed_metric_delta(lookup[(scale, dataset)], lookup[(1.0, dataset)])
            per_dataset[dataset] = sum(deltas) / len(deltas)
            improve_counts[dataset] = sum(delta > 1e-8 for delta in deltas)
            decline_counts[dataset] = sum(delta < -1e-8 for delta in deltas)
            favorable += improve_counts[dataset]
            harmful += decline_counts[dataset]
            total += len(deltas)
        cod = lookup.get((scale, "TE-COD10K"))
        cod_redline_ok = bool(
            cod
            and cod["S_m"] >= 0.727
            and cod["F_beta_w"] >= 0.573
            and cod["E_phi_m"] >= 0.824
            and cod["MAE"] <= 0.058
        )
        return {
            "scale": scale,
            "per_dataset": per_dataset,
            "favorable": favorable,
            "harmful": harmful,
            "total": total,
            "improve_counts": improve_counts,
            "decline_counts": decline_counts,
            "cod_redline_ok": cod_redline_ok,
        }

    details = [scale_detail(scale) for scale in sorted(high_scales, reverse=True)]
    chosen = max(details, key=lambda item: scale_scores.get(item["scale"], -float("inf")))
    a_match = next(
        (
            item
            for item in details
            if all_required
            and all(item["improve_counts"].get(dataset, 0) >= 3 for dataset in required)
            and item["cod_redline_ok"]
        ),
        None,
    )
    b_match = next(
        (
            item
            for item in details
            if all_required
            and item["improve_counts"].get("CHAMELEON", 0) >= 3
            and item["improve_counts"].get("TE-CAMO", 0) >= 3
            and (
                item["decline_counts"].get("TE-COD10K", 0) >= 3
                or item["decline_counts"].get("NC4K", 0) >= 3
                or not item["cod_redline_ok"]
            )
        ),
        None,
    )
    d_match = next(
        (
            item
            for item in details
            if item["total"]
            and item["harmful"] >= math.ceil(0.6 * item["total"])
            and all(score < -2e-4 for score in item["per_dataset"].values())
        ),
        None,
    )

    if a_match is not None:
        chosen = a_match
        label = "A"
        conclusion = (
            "HR residual direction appears useful; current scale is too weak.\n"
            "Recommend training HR-BFR-v1.1 with larger beta."
        )
    elif d_match is not None:
        chosen = d_match
        label = "D"
        conclusion = (
            "The learned HR residual direction is harmful.\n"
            "Stop HR-BFR-v1 scaling experiments."
        )
    elif b_match is not None:
        chosen = b_match
        label = "B"
        conclusion = (
            "Increasing residual strength creates dataset trade-off.\n"
            "Do not enlarge beta; move to directional boundary refinement."
        )
    else:
        label = "C"
        conclusion = (
            "The learned residual does not contain sufficiently useful boundary information.\n"
            "Move to HR-BFR-v2 signed boundary displacement."
        )
    detail = (
        f"Selected high scale={chosen['scale']:g}; "
        f"favorable metrics={chosen['favorable']}/{chosen['total']}; "
        f"harmful metrics={chosen['harmful']}/{chosen['total']}; "
        f"COD10K redline={chosen['cod_redline_ok']}."
    )
    return label, conclusion, detail


def format_delta(value):
    return f"{value:+.6f}"


def build_summary(args, rows, flip_rows, diagnosis, source_epoch):
    lookup = {(float(row["scale"]), row["dataset"]): row for row in rows}
    scales = sorted({float(row["scale"]) for row in rows})
    datasets = []
    for row in rows:
        if row["dataset"] not in datasets:
            datasets.append(row["dataset"])
    lines = [
        "# HR Residual Scale Probe",
        "",
        f"- Config: `{args.config}`",
        f"- Checkpoint: `{args.ckpt}`",
        f"- Checkpoint epoch: `{source_epoch}`",
        "- Threshold: `0.5`",
        "- Diagnostics are computed at native HR resolution (136x136).",
        "- Metrics use the unchanged eval path: HR logits are resized to GT size and binarized at 0.5.",
        "",
    ]
    for scale in scales:
        lines.extend(
            [
                f"## Scale {scale:g}",
                "",
                "| Dataset | S_m | Fw | Fm | E | MAE | Δavg vs 1.0 | Δavg vs 0.0 | flip | outside flip |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for dataset in datasets:
            row = lookup[(scale, dataset)]
            delta_1 = (
                sum(signed_metric_delta(row, lookup[(1.0, dataset)])) / 5.0
                if (1.0, dataset) in lookup
                else 0.0
            )
            delta_0 = (
                sum(signed_metric_delta(row, lookup[(0.0, dataset)])) / 5.0
                if (0.0, dataset) in lookup
                else 0.0
            )
            lines.append(
                f"| {dataset} | {row['S_m']:.6f} | {row['F_beta_w']:.6f} | "
                f"{row['F_beta_m']:.6f} | {row['E_phi_m']:.6f} | {row['MAE']:.6f} | "
                f"{format_delta(delta_1)} | {format_delta(delta_0)} | "
                f"{row['binary_flip_ratio']:.8f} | {row['flip_ratio_outside_band']:.8f} |"
            )
        lines.append("")

    label, conclusion, detail = diagnosis
    lines.extend(
        [
            "## Automatic Diagnosis",
            "",
            f"Classification: **{label}**",
            "",
            conclusion,
            "",
            detail,
            "",
            "## Sanity Checks",
            "",
        ]
    )
    if 0.0 in scales:
        zero_rows = [row for row in rows if float(row["scale"]) == 0.0]
        lines.append(
            f"- scale=0 max logit delta: `{max(row['mean_abs_logit_delta'] for row in zero_rows):.12g}`"
        )
        lines.append(
            f"- scale=0 max binary flip ratio: `{max(row['binary_flip_ratio'] for row in zero_rows):.12g}`"
        )
    lines.append(
        f"- max outside-band flip ratio: `{max(row['flip_ratio_outside_band'] for row in rows):.12g}`"
    )
    scale1_errors = [
        row["scale1_reproduction_max_abs_error"]
        for row in flip_rows
        if float(row["scale"]) == 1.0
    ]
    if scale1_errors:
        lines.append(f"- scale=1 reproduction max error: `{max(scale1_errors):.12g}`")
    if 1.0 in scales and 2.0 in scales:
        ratios = []
        for dataset in datasets:
            d1 = lookup[(1.0, dataset)]["mean_abs_logit_delta"]
            d2 = lookup[(2.0, dataset)]["mean_abs_logit_delta"]
            if d1 > 0.0:
                ratios.append(d2 / d1)
        if ratios:
            lines.append(f"- scale=2 / scale=1 logit-delta ratio mean: `{sum(ratios) / len(ratios):.8f}`")
    return "\n".join(lines) + "\n"


def create_zip(out_dir):
    zip_path = Path(out_dir) / "hr_residual_scale_probe.zip"
    members = [
        Path(out_dir) / "summary.md",
        Path(out_dir) / "summary.json",
        Path(out_dir) / "metrics_by_scale.csv",
        Path(out_dir) / "flip_stats_by_scale.csv",
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in members:
            zf.write(path, arcname=path.name)
    return zip_path


def main():
    args = parse_args()
    scales = parse_csv_list(args.scales, float)
    datasets = parse_csv_list(args.datasets, str)
    validate_args(args, scales)
    cfg = load_config(args.config)
    if not bool(getattr(cfg, "USE_HR_BFR", False)):
        raise RuntimeError("Probe requires USE_HR_BFR=True.")
    if str(getattr(cfg, "HEAD_TYPE", "")).lower() != "dagp_safe_csd_v1r":
        raise RuntimeError("Probe requires HEAD_TYPE='dagp_safe_csd_v1r'.")
    if float(getattr(cfg, "HR_BFR_EVAL_RES_SCALE", 1.0)) != 1.0:
        raise RuntimeError("Base config HR_BFR_EVAL_RES_SCALE must remain 1.0.")

    out_dir = Path(args.out).expanduser()
    ensure_dir(out_dir)
    ensure_cache_available(cfg, "feature", split="test")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student, checkpoint, source_epoch = load_student(cfg, args.ckpt, device)
    initial_state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}

    loaders = {}
    sample_counts = {}
    for dataset_name in datasets:
        dataset = CachedEvalDataset(
            cfg,
            split="test",
            datasets=[dataset_name],
            max_samples=int(args.max_samples),
        )
        sample_counts[dataset_name] = len(dataset)
        loaders[dataset_name] = DataLoader(
            dataset,
            batch_size=int(args.batch_size),
            shuffle=False,
            drop_last=False,
            num_workers=int(cfg.NUM_WORKERS),
            pin_memory=torch.cuda.is_available(),
            collate_fn=variable_gt_collate,
            **dataloader_worker_kwargs(cfg),
        )
        loaders[dataset_name].probe_metric_workers = int(args.metric_workers)

    print(f"device = {device}")
    print(f"checkpoint = {args.ckpt}")
    print(f"source_epoch = {source_epoch}")
    print(f"scales = {scales}")
    print(f"datasets = {datasets}")
    print(f"sample_counts = {sample_counts}")
    print(f"batch_size = {int(args.batch_size)} | metric_workers = {int(args.metric_workers) or 'auto'}")
    print("training = False | optimizer = None | checkpoint mutation = False")

    metric_rows = []
    flip_rows = []
    for dataset_name in datasets:
        results_by_scale = evaluate_scales_dataset(
            cfg,
            student,
            loaders[dataset_name],
            device,
            scales,
        )
        for scale in scales:
            metric_result, stat_result = results_by_scale[float(scale)]
            row = {"scale": float(scale), "dataset": dataset_name, **metric_result, **stat_result}
            metric_rows.append({field: row[field] for field in METRIC_FIELDS})
            flip_rows.append({field: row[field] for field in FLIP_FIELDS})
            print(
                f"[Probe] scale={scale:g} dataset={dataset_name} "
                f"S={row['S_m']:.6f} Fw={row['F_beta_w']:.6f} "
                f"E={row['E_phi_m']:.6f} MAE={row['MAE']:.6f} "
                f"logit_delta={row['mean_abs_logit_delta']:.8f} "
                f"flip={row['binary_flip_ratio']:.8f} "
                f"outside_flip={row['flip_ratio_outside_band']:.12g}"
            )

    for key, before in initial_state.items():
        after = student.state_dict()[key].detach().cpu()
        if not torch.equal(before, after):
            raise RuntimeError(f"Eval-only probe unexpectedly changed model state: {key}")

    metric_rows.sort(key=lambda row: (float(row["scale"]), datasets.index(row["dataset"])))
    flip_rows.sort(key=lambda row: (float(row["scale"]), datasets.index(row["dataset"])))
    diagnosis = diagnose(metric_rows)
    write_csv(out_dir / "metrics_by_scale.csv", metric_rows, METRIC_FIELDS)
    write_csv(out_dir / "flip_stats_by_scale.csv", flip_rows, FLIP_FIELDS)
    summary_md = build_summary(args, metric_rows, flip_rows, diagnosis, source_epoch)
    (out_dir / "summary.md").write_text(summary_md, encoding="utf-8")
    summary_json = {
        "config": str(args.config),
        "checkpoint": str(args.ckpt),
        "checkpoint_epoch": int(source_epoch),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "threshold": 0.5,
        "scales": scales,
        "datasets": datasets,
        "sample_counts": sample_counts,
        "diagnosis": {
            "classification": diagnosis[0],
            "conclusion": diagnosis[1],
            "detail": diagnosis[2],
        },
        "metrics_by_scale": metric_rows,
        "flip_stats_by_scale": flip_rows,
        "training_performed": False,
        "optimizer_created": False,
        "model_state_unchanged": True,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    zip_path = create_zip(out_dir)
    print(f"diagnosis = {diagnosis[0]} | {diagnosis[1].replace(chr(10), ' ')}")
    print(f"output_zip = {zip_path}")


if __name__ == "__main__":
    main()
