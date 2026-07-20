#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_DYNAMIC_EPOCHS = (1, 6, 7, 10, 15, 20, 21, 25, 35, 40, 42, 45)
REQUIRED_DATASETS = ("CHAMELEON", "CAMO", "COD10K", "NC4K")

TABLE_ROW_RE = re.compile(
    r"^\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*"
    r"\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|"
)


def parse_key_values(line):
    values = {}
    for part in line.split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def normalize_dataset(name):
    aliases = {
        "TE-CAMO": "CAMO",
        "TE-COD10K": "COD10K",
    }
    return aliases.get(str(name).strip(), str(name).strip())


def next_metric_row(lines, start, path):
    for index in range(start, min(start + 10, len(lines))):
        match = TABLE_ROW_RE.match(lines[index].strip())
        if match:
            values = [float(value) for value in match.groups()]
            return {
                "S_m": values[0],
                "F_beta_w": values[1],
                "F_beta_m": values[2],
                "E_phi_m": values[3],
                "MAE": values[4],
            }, index
    raise RuntimeError(f"Metric table row missing near line {start + 1}: {path}")


def parse_training_log(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Training log not found: {path}")
    lines = path.read_text(errors="replace").splitlines()
    rows = {}
    validation = {}
    for index, line in enumerate(lines):
        match = re.search(
            r"\[Train\] Epoch\s+(\d+)/\d+\s+\|\s+avg_train_loss=([0-9.eE+-]+)",
            line,
        )
        if match:
            epoch = int(match.group(1))
            rows.setdefault(epoch, {})["train_loss"] = float(match.group(2))
            continue
        match = re.search(r"\[PredArea\] epoch=(\d+)", line)
        if match:
            epoch = int(match.group(1))
            values = parse_key_values(line)
            target = rows.setdefault(epoch, {})
            for source, destination in (
                ("student_prob_mean", "student_prob_mean"),
                ("teacher_prob_mean", "teacher_prob_mean"),
                ("student_pred_area_mean", "student_pred_area"),
                ("teacher_pred_area_mean", "teacher_pred_area"),
            ):
                if source in values:
                    target[destination] = float(values[source])
            continue
        match = re.search(r"\[StaticWeight\] epoch=(\d+)", line)
        if match:
            epoch = int(match.group(1))
            values = parse_key_values(line)
            pair = values.get("global_static/teacher")
            if pair:
                static_weight, teacher_weight = pair.split("/", 1)
                rows.setdefault(epoch, {})["global_static_weight"] = float(
                    static_weight
                )
                rows.setdefault(epoch, {})["global_teacher_weight"] = float(
                    teacher_weight
                )
            continue
        match = re.search(r"\[TeacherRouting\] epoch=(\d+)", line)
        if match:
            epoch = int(match.group(1))
            values = parse_key_values(line)
            triple = values.get("map_min/mean/max")
            if triple:
                _, mean_value, _ = triple.split("/", 2)
                rows.setdefault(epoch, {})["teacher_route_map_mean"] = float(
                    mean_value
                )
            continue
        match = re.search(r"\[ECST\] epoch=(\d+)", line)
        if match:
            epoch = int(match.group(1))
            values = parse_key_values(line)
            triple = values.get("map_min/mean/max")
            if triple:
                _, mean_value, _ = triple.split("/", 2)
                rows.setdefault(epoch, {})["teacher_route_map_mean"] = float(
                    mean_value
                )
            region = values.get("map_core_conflict/extent_bg/unknown")
            if region:
                core, extent, unknown = region.split("/", 2)
                target = rows.setdefault(epoch, {})
                target["core_conflict_weight_mean"] = float(core)
                target["extent_bg_weight_mean"] = float(extent)
                target["unknown_weight_mean"] = float(unknown)
            continue
        match = re.search(
            r"\[Validation\] Epoch\s+(\d+)/\d+\s+\|\s+Dataset:\s+([^|]+)",
            line,
        )
        if match:
            epoch = int(match.group(1))
            dataset = normalize_dataset(match.group(2))
            metrics, _ = next_metric_row(lines, index + 1, path)
            validation[(epoch, dataset)] = metrics

    required_fields = (
        "train_loss",
        "student_prob_mean",
        "teacher_prob_mean",
        "student_pred_area",
        "teacher_pred_area",
        "global_static_weight",
        "global_teacher_weight",
        "teacher_route_map_mean",
    )
    for epoch in REQUIRED_DYNAMIC_EPOCHS:
        missing = [name for name in required_fields if name not in rows.get(epoch, {})]
        if missing:
            raise RuntimeError(
                f"Training log missing epoch {epoch} fields {missing}: {path}"
            )
        if (epoch, "CAMO") not in validation:
            raise RuntimeError(
                f"Training log missing CAMO validation for epoch {epoch}: {path}"
            )
    return rows, validation


def parse_eval_log(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation log not found: {path}")
    lines = path.read_text(errors="replace").splitlines()
    results = {}
    for index, line in enumerate(lines):
        match = re.search(r"\[Eval\] Dataset:\s+(.+)$", line)
        if not match:
            continue
        dataset = normalize_dataset(match.group(1))
        metrics, metric_index = next_metric_row(lines, index + 1, path)
        detail = None
        for detail_index in range(
            metric_index + 1, min(metric_index + 8, len(lines))
        ):
            if "F_MAX=" in lines[detail_index]:
                detail = parse_key_values(lines[detail_index])
                break
        if detail is None:
            raise RuntimeError(
                f"F_MAX/E_MAX/mIoU row missing for {dataset}: {path}"
            )
        metrics.update(
            {
                "F_MAX": float(detail["F_MAX"]),
                "E_MAX": float(detail["E_MAX"]),
                "mIoU": float(detail["mIoU@0.5"]),
            }
        )
        results[dataset] = metrics
    missing = sorted(set(REQUIRED_DATASETS) - set(results))
    if missing:
        raise RuntimeError(f"Evaluation log missing datasets {missing}: {path}")
    return results


def parse_eval_run(value):
    parts = str(value).split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--eval-run must be MODEL:EPOCH:SELECTION_RULE:LOG_PATH"
        )
    model, epoch, selection_rule, path = parts
    try:
        epoch = int(epoch)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"Invalid checkpoint epoch in --eval-run: {value}"
        ) from error
    if not model or not selection_rule or not path:
        raise argparse.ArgumentTypeError(f"Invalid --eval-run: {value}")
    if selection_rule not in {"matched_epoch", "best_camo"}:
        raise argparse.ArgumentTypeError(
            "SELECTION_RULE must be matched_epoch or best_camo"
        )
    return model, epoch, selection_rule, Path(path)


def validate_eval_runs(runs):
    matched = {(model, epoch) for model, epoch, rule, _ in runs if rule == "matched_epoch"}
    required = {
        ("sw_ones_ecst", 42),
        ("sw_ones_ecst", 45),
        ("sw_ones_noecst", 42),
        ("sw_ones_noecst", 45),
    }
    missing = sorted(required - matched)
    if missing:
        raise RuntimeError(f"Missing required matched-epoch eval runs: {missing}")
    if not any(
        model == "sw_ones_noecst" and rule == "best_camo"
        for model, _, rule, _ in runs
    ):
        raise RuntimeError("Missing sw_ones_noecst best_camo eval run")


def ensure_output_inside_project(path):
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise RuntimeError(
            f"Output must stay inside {PROJECT_ROOT}, got {resolved}"
        ) from error
    return resolved


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Build strict sw_ones ECST vs No-ECST comparison CSVs."
    )
    parser.add_argument("--ecst-train-log", required=True)
    parser.add_argument("--noecst-train-log", required=True)
    parser.add_argument(
        "--eval-run",
        action="append",
        type=parse_eval_run,
        required=True,
        help="Repeat MODEL:EPOCH:SELECTION_RULE:LOG_PATH.",
    )
    parser.add_argument(
        "--out",
        default="../workdir/noecst_comparison",
    )
    args = parser.parse_args()

    validate_eval_runs(args.eval_run)
    output = ensure_output_inside_project(args.out)
    ecst_rows, ecst_validation = parse_training_log(args.ecst_train_log)
    noecst_rows, noecst_validation = parse_training_log(args.noecst_train_log)

    comparison_rows = []
    for model, epoch, selection_rule, path in args.eval_run:
        metrics_by_dataset = parse_eval_log(path)
        for dataset in REQUIRED_DATASETS:
            row = {
                "model": model,
                "checkpoint_epoch": epoch,
                "selection_rule": selection_rule,
                "dataset": dataset,
            }
            row.update(metrics_by_dataset[dataset])
            comparison_rows.append(row)
    write_csv(
        output / "sw_ones_vs_noecst.csv",
        (
            "model",
            "checkpoint_epoch",
            "selection_rule",
            "dataset",
            "S_m",
            "F_beta_w",
            "F_beta_m",
            "E_phi_m",
            "MAE",
            "F_MAX",
            "E_MAX",
            "mIoU",
        ),
        comparison_rows,
    )

    dynamics_rows = []
    for model, rows, validation in (
        ("sw_ones_ecst", ecst_rows, ecst_validation),
        ("sw_ones_noecst", noecst_rows, noecst_validation),
    ):
        for epoch in REQUIRED_DYNAMIC_EPOCHS:
            source = rows[epoch]
            metrics = validation[(epoch, "CAMO")]
            dynamics_rows.append(
                {
                    "epoch": epoch,
                    "model": model,
                    "train_loss": source["train_loss"],
                    "validation_CAMO_Sm": metrics["S_m"],
                    "validation_CAMO_Fw": metrics["F_beta_w"],
                    "validation_CAMO_Em": metrics["E_phi_m"],
                    "validation_CAMO_MAE": metrics["MAE"],
                    "student_prob_mean": source["student_prob_mean"],
                    "teacher_prob_mean": source["teacher_prob_mean"],
                    "student_pred_area": source["student_pred_area"],
                    "teacher_pred_area": source["teacher_pred_area"],
                    "global_static_weight": source["global_static_weight"],
                    "global_teacher_weight": source["global_teacher_weight"],
                    "teacher_route_map_mean": source[
                        "teacher_route_map_mean"
                    ],
                    "core_conflict_weight_mean": source.get(
                        "core_conflict_weight_mean", 1.0
                    ),
                    "extent_bg_weight_mean": source.get(
                        "extent_bg_weight_mean", 1.0
                    ),
                    "unknown_weight_mean": source.get(
                        "unknown_weight_mean", 1.0
                    ),
                }
            )
    write_csv(
        output / "training_dynamics.csv",
        (
            "epoch",
            "model",
            "train_loss",
            "validation_CAMO_Sm",
            "validation_CAMO_Fw",
            "validation_CAMO_Em",
            "validation_CAMO_MAE",
            "student_prob_mean",
            "teacher_prob_mean",
            "student_pred_area",
            "teacher_pred_area",
            "global_static_weight",
            "global_teacher_weight",
            "teacher_route_map_mean",
            "core_conflict_weight_mean",
            "extent_bg_weight_mean",
            "unknown_weight_mean",
        ),
        dynamics_rows,
    )
    print(f"Wrote {output / 'sw_ones_vs_noecst.csv'}")
    print(f"Wrote {output / 'training_dynamics.csv'}")


if __name__ == "__main__":
    main()
