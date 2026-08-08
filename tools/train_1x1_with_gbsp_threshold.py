#!/usr/bin/env python3
"""Guarded launcher for the CF-BRC-HC pure-Student 1x1 experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = MAIN_ROOT / (
    "configs/dinov1_s8_dabe_clean_v1_dp_gbsp_cf_brc_hc_hard_linear_"
    "staticonly_purestudent_long45_lrfloor_2e5.py"
)
DEFAULT_OFFLINE_AUDIT = MAIN_ROOT.parent / "workdir/gbsp_threshold/eval_full6473/numerical_audit.json"
DEFAULT_TRAIN_CACHE_SUMMARY = MAIN_ROOT.parent / (
    "workdir/gbsp_threshold/crossfit_train4040/performance_summary.json"
)


def _json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _verify_gate(offline: dict, train_cache: dict) -> None:
    errors = []
    if not bool(offline.get("full_formal_evaluation", False)):
        errors.append("offline threshold evaluation is not the complete 6473-image run")
    if int(offline.get("num_failed", -1)) != 0:
        errors.append("offline threshold evaluation contains failures")
    if str(offline.get("gate", {}).get("status")) != "PASS_TO_FULL_OR_DOWNSTREAM":
        errors.append(f"offline gate={offline.get('gate')}")
    for field in ("hc_fallback_ratio", "hc_empty_ratio", "hc_large_ratio"):
        if float(offline.get(field, 1.0)) > 0.01:
            errors.append(f"offline {field} exceeds 1%")
    if int(train_cache.get("num_requested", -1)) != 4040 or int(
        train_cache.get("num_valid", -1)
    ) != 4040:
        errors.append("training crossfit cache is not complete 4040/4040")
    if int(train_cache.get("num_failed", -1)) != 0:
        errors.append("training crossfit cache contains failures")
    if float(train_cache.get("hc_fallback_ratio", 1.0)) > 0.01:
        errors.append("training-cache HC fallback ratio exceeds 1%")
    if not bool(train_cache.get("downstream_training_allowed", False)):
        errors.append("training cache explicitly blocks downstream training")
    if errors:
        raise RuntimeError("CF-BRC-HC downstream gate failed: " + "; ".join(errors))


def main(args: argparse.Namespace) -> None:
    config = Path(args.config).resolve()
    offline_path = Path(args.offline_audit).resolve()
    cache_path = Path(args.train_cache_summary).resolve()
    offline, train_cache = _json(offline_path), _json(cache_path)
    _verify_gate(offline, train_cache)
    command = [sys.executable, str(MAIN_ROOT / "train.py"), "--config", str(config)]
    if args.work_dir:
        command.extend(("--work_dir", args.work_dir))
    if args.max_train_samples is not None:
        command.extend(("--max_train_samples", str(args.max_train_samples)))
    if args.max_epochs is not None:
        command.extend(("--max_epochs", str(args.max_epochs)))
    if args.stop_after_epoch is not None:
        command.extend(("--stop_after_epoch", str(args.stop_after_epoch)))
    if args.debug_loader_only:
        command.append("--debug_loader_only")
    print(json.dumps({"status": "gate_pass", "command": command}, ensure_ascii=False, indent=2))
    if not args.dry_run:
        subprocess.run(command, cwd=MAIN_ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--offline-audit", default=str(DEFAULT_OFFLINE_AUDIT))
    parser.add_argument("--train-cache-summary", default=str(DEFAULT_TRAIN_CACHE_SUMMARY))
    parser.add_argument("--work-dir")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--stop-after-epoch", type=int)
    parser.add_argument("--debug-loader-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
