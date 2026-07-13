import argparse
import csv
import json
import re
from pathlib import Path


BASELINE = {
    "CHAMELEON": {"S": 0.7229, "Fw": 0.6082, "Fm": 0.6631, "E": 0.8374, "MAE": 0.0729},
    "CAMO": {"S": 0.7150, "Fw": 0.6315, "Fm": 0.6919, "E": 0.8156, "MAE": 0.1066},
    "COD10K": {"S": 0.7301, "Fw": 0.5823, "Fm": 0.6251, "E": 0.8268, "MAE": 0.0559},
    "NC4K": {"S": 0.7771, "Fw": 0.7006, "Fm": 0.7463, "E": 0.8611, "MAE": 0.0669},
}


def normalize_dataset(value):
    value = str(value).strip().upper()
    aliases = {
        "TE-CAMO": "CAMO",
        "CAMO": "CAMO",
        "TE-COD10K": "COD10K",
        "COD10K": "COD10K",
        "CHAMELEON": "CHAMELEON",
        "NC4K": "NC4K",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported dataset name: {value!r}")
    return aliases[value]


def parse_log(path):
    metrics = {}
    current = None
    numeric = re.compile(
        r"^\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|$"
    )
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if "[Eval] Dataset:" in line:
            current = normalize_dataset(line.split("[Eval] Dataset:", 1)[1].strip())
            continue
        match = numeric.match(line.strip())
        if current is not None and match:
            values = [float(value) for value in match.groups()]
            metrics[current] = dict(zip(("S", "Fw", "Fm", "E", "MAE"), values))
            current = None
    return metrics


def _first(row, names):
    for name in names:
        if name in row and str(row[name]).strip() != "":
            return float(row[name])
    raise KeyError(f"CSV row lacks metric columns {names}; available={sorted(row)}")


def parse_csv(path):
    metrics = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            dataset_key = next((key for key in ("dataset", "Dataset", "name") if key in row), None)
            if dataset_key is None:
                raise KeyError("CSV requires a dataset column.")
            dataset = normalize_dataset(row[dataset_key])
            metrics[dataset] = {
                "S": _first(row, ("S", "S_m", "SMeasure")),
                "Fw": _first(row, ("Fw", "F_beta^w", "WFM")),
                "Fm": _first(row, ("Fm", "F_beta^m", "F_MEAN")),
                "E": _first(row, ("E", "E_phi^m", "E_MEAN")),
                "MAE": _first(row, ("MAE", "M")),
            }
    return metrics


def global_score(values):
    return (values["S"] + values["Fw"] + values["Fm"] + values["E"] + (1.0 - values["MAE"])) / 5.0


def main():
    parser = argparse.ArgumentParser(description="Compute CACD four-dataset global score and red-line checks.")
    parser.add_argument("--input", required=True, help="eval log or CSV")
    parser.add_argument("--format", choices=["auto", "log", "csv"], default="auto")
    parser.add_argument("--out", default=None, help="Optional output JSON path")
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    kind = args.format
    if kind == "auto":
        kind = "csv" if input_path.suffix.lower() == ".csv" else "log"
    metrics = parse_csv(input_path) if kind == "csv" else parse_log(input_path)
    missing = sorted(set(BASELINE) - set(metrics))
    if missing:
        raise RuntimeError(f"Input does not contain all four datasets; missing={missing}")
    per_dataset = {}
    for dataset in BASELINE:
        score = global_score(metrics[dataset])
        baseline_score = global_score(BASELINE[dataset])
        per_dataset[dataset] = {
            **metrics[dataset],
            "G_d": score,
            "baseline_G_d": baseline_score,
            "delta_G_d": score - baseline_score,
        }
    macro = sum(row["G_d"] for row in per_dataset.values()) / 4.0
    baseline_macro = sum(global_score(row) for row in BASELINE.values()) / 4.0
    cod = metrics["COD10K"]
    nc4k = metrics["NC4K"]
    redlines = {
        "COD10K": cod["S"] >= 0.727 and cod["Fw"] >= 0.573 and cod["E"] >= 0.824 and cod["MAE"] <= 0.058,
        "NC4K": nc4k["S"] >= 0.775 and nc4k["Fw"] >= 0.691 and nc4k["E"] >= 0.859 and nc4k["MAE"] <= 0.068,
    }
    improved_count = sum(row["delta_G_d"] >= 0.001 for row in per_dataset.values())
    max_drop = min(row["delta_G_d"] for row in per_dataset.values())
    success = macro - baseline_macro >= 0.003 and improved_count >= 3 and max_drop >= -0.003 and all(redlines.values())
    report = {
        "input": str(input_path.resolve()),
        "per_dataset": per_dataset,
        "G_macro": macro,
        "baseline_G_macro": baseline_macro,
        "delta_G_macro": macro - baseline_macro,
        "datasets_improved_ge_0_001": improved_count,
        "largest_dataset_drop": max_drop,
        "redlines": redlines,
        "minimum_success": success,
    }
    for dataset, row in per_dataset.items():
        print(
            f"{dataset}: G_d={row['G_d']:.6f} | baseline={row['baseline_G_d']:.6f} | "
            f"delta={row['delta_G_d']:+.6f}"
        )
    print(f"G_macro = {macro:.6f}")
    print(f"baseline_G_macro = {baseline_macro:.6f}")
    print(f"delta_G_macro = {macro - baseline_macro:+.6f}")
    print(f"redlines = {redlines}")
    print(f"minimum_success = {success}")
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote = {out_path}")


if __name__ == "__main__":
    main()
