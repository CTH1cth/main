import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.dataset import CachedTrainDataset  # noqa: E402
from common.utils import ensure_dir, load_config, torch_load  # noqa: E402
from model import build_seg_head  # noqa: E402
from train import (  # noqa: E402
    build_esa_ber_loss,
    build_rast_teacher_weight_map,
    make_image_68,
    make_model_input,
    resize_logits_for_loss,
    set_model_epoch,
)


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows, key):
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def main():
    parser = argparse.ArgumentParser(description="Audit ESA-v2-BER candidates without training or GT.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out_dir = Path(args.out)
    ensure_dir(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = CachedTrainDataset(cfg, max_samples=args.max_samples)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size or cfg.BATCH_SIZE),
        shuffle=False,
        num_workers=int(cfg.NUM_WORKERS),
        pin_memory=torch.cuda.is_available(),
    )
    checkpoint = torch_load(args.ckpt, map_location="cpu")
    if "student" not in checkpoint or "teacher" not in checkpoint:
        raise RuntimeError("ESA-v2-BER audit requires checkpoint student and teacher states.")
    student_state = checkpoint["student"]
    teacher_state = checkpoint["teacher"]
    if "base_head.weight" not in student_state:
        raise RuntimeError("ESA-v2-BER audit checkpoint is not a DAGP-Safe head checkpoint.")
    in_channels = int(student_state["base_head.weight"].shape[1])
    student = build_seg_head(in_channels, cfg).to(device)
    teacher = build_seg_head(in_channels, cfg).to(device)
    student.load_state_dict(student_state, strict=True)
    teacher.load_state_dict(teacher_state, strict=True)
    audit_epoch = int(getattr(cfg, "ESA_BER_START_EPOCH", 21))
    set_model_epoch(student, audit_epoch)
    set_model_epoch(teacher, audit_epoch)
    student.eval()
    teacher.eval()

    per_image_rows = []
    batch_summaries = []
    with torch.no_grad():
        for batch in loader:
            if "gt" in batch:
                raise RuntimeError("ESA-v2-BER audit must not read training GT.")
            model_input = make_model_input(cfg, batch, device)
            image_68 = make_image_68(cfg, batch, device)
            student_out = student(model_input, image_68=image_68, return_aux=True)
            final_logits = resize_logits_for_loss(student_out["logits"], cfg)
            teacher_out = teacher(model_input, image_68=image_68, return_aux=False)
            teacher_logits = resize_logits_for_loss(
                teacher_out["logits"] if isinstance(teacher_out, dict) else teacher_out,
                cfg,
            )
            teacher_prob = teacher_logits.sigmoid().detach()
            teacher_binary = (teacher_prob >= 0.5).float().detach()
            teacher_map, rast_stats = build_rast_teacher_weight_map(
                cfg, batch, teacher_binary, audit_epoch, device
            )
            _, stats, _ = build_esa_ber_loss(
                cfg,
                audit_epoch,
                student_out,
                final_logits,
                batch,
                teacher_prob,
                teacher_binary,
                teacher_map,
                rast_stats,
                device,
            )
            per_image_rows.extend(stats["per_image"])
            batch_summaries.append(stats)

    if not per_image_rows:
        raise RuntimeError("ESA-v2-BER audit produced no sample rows.")
    per_image_fields = [
        "dataset",
        "stem",
        "valid",
        "selected_pairs",
        "pos_raw_ratio",
        "neg_extent_raw_ratio",
        "neg_hard_bg_raw_ratio",
        "neg_raw_ratio",
        "pos_margin_mean",
        "pos_conn_mean",
        "neg_margin_mean",
        "neg_conn_mean",
        "logit_gap",
        "rank_violation_ratio",
    ]
    write_csv(out_dir / "per_image_candidate_stats.csv", per_image_rows, per_image_fields)

    grouped = defaultdict(list)
    for row in per_image_rows:
        grouped[row["dataset"]].append(row)
    source_rows = []
    for source, rows in sorted(grouped.items()):
        source_rows.append(
            {
                "source": source,
                "num_images": len(rows),
                "valid_image_ratio": sum(bool(row["valid"]) for row in rows) / len(rows),
                "pos_raw_ratio": mean(rows, "pos_raw_ratio"),
                "neg_extent_raw_ratio": mean(rows, "neg_extent_raw_ratio"),
                "neg_hard_bg_raw_ratio": mean(rows, "neg_hard_bg_raw_ratio"),
                "neg_raw_ratio": mean(rows, "neg_raw_ratio"),
                "selected_pairs_mean": mean(rows, "selected_pairs"),
                "pos_margin_mean": mean(rows, "pos_margin_mean"),
                "pos_conn_mean": mean(rows, "pos_conn_mean"),
                "neg_margin_mean": mean(rows, "neg_margin_mean"),
                "neg_conn_mean": mean(rows, "neg_conn_mean"),
                "logit_gap": mean(rows, "logit_gap"),
                "rank_violation_ratio": mean(rows, "rank_violation_ratio"),
            }
        )
    source_fields = list(source_rows[0].keys())
    write_csv(out_dir / "per_source_stats.csv", source_rows, source_fields)

    summary = {
        "config": args.config,
        "checkpoint": args.ckpt,
        "audit_epoch": audit_epoch,
        "training_gt_used": False,
        "num_images": len(per_image_rows),
        "valid_image_ratio": sum(bool(row["valid"]) for row in per_image_rows) / len(per_image_rows),
        "pos_raw_ratio": mean(per_image_rows, "pos_raw_ratio"),
        "neg_extent_raw_ratio": mean(per_image_rows, "neg_extent_raw_ratio"),
        "neg_hard_bg_raw_ratio": mean(per_image_rows, "neg_hard_bg_raw_ratio"),
        "neg_raw_ratio": mean(per_image_rows, "neg_raw_ratio"),
        "selected_pairs_mean": mean(per_image_rows, "selected_pairs"),
        "logit_gap": mean(per_image_rows, "logit_gap"),
        "rank_violation_ratio": mean(per_image_rows, "rank_violation_ratio"),
        "topk_sem_weight_sum_error_max": max(
            float(stats["topk_sem_weight_sum_error"]) for stats in batch_summaries
        ),
        "sources": source_rows,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    lines = [
        "# ESA-v2-BER Candidate Audit",
        "",
        f"- Config: `{args.config}`",
        f"- Checkpoint: `{args.ckpt}`",
        f"- Audit epoch: {audit_epoch}",
        "- Training GT used: false",
        f"- Images: {summary['num_images']}",
        f"- Valid image ratio: {summary['valid_image_ratio']:.6f}",
        f"- Positive raw ratio: {summary['pos_raw_ratio']:.8f}",
        f"- Negative extent raw ratio: {summary['neg_extent_raw_ratio']:.8f}",
        f"- Negative hard-bg raw ratio: {summary['neg_hard_bg_raw_ratio']:.8f}",
        f"- Selected pairs mean: {summary['selected_pairs_mean']:.4f}",
        f"- Current logit gap: {summary['logit_gap']:.6f}",
        f"- Rank violation ratio: {summary['rank_violation_ratio']:.6f}",
        f"- Top-k weight sum error max: {summary['topk_sem_weight_sum_error_max']:.8g}",
        "",
        "## Per Source",
        "",
    ]
    for row in source_rows:
        lines.append(
            f"- {row['source']}: valid={row['valid_image_ratio']:.6f}, "
            f"pairs={row['selected_pairs_mean']:.4f}, gap={row['logit_gap']:.6f}, "
            f"violation={row['rank_violation_ratio']:.6f}"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

