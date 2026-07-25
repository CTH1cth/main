#!/usr/bin/env python3
"""GT-free manifest and distribution audit for DABE-Clean caches."""

import argparse
import json
import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.utils import check_dabe_clean_cache, load_config  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()
    if args.max_samples == 0 or args.max_samples < -1:
        parser.error("--max-samples must be -1 or a positive integer.")

    cfg = load_config(args.config)
    _, reason, stats = check_dabe_clean_cache(
        cfg,
        max_samples=args.max_samples,
        return_stats=True,
    )
    result = {
        "schema": "dabe_clean_v2_contrec_cache_audit",
        "config": str(Path(args.config).resolve()),
        "training_gt_read": False,
        "teacher_prediction_read": False,
        "result": reason,
        "distributions": stats,
    }
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        output = Path(args.output_json).resolve()
        if MAIN_ROOT == output or MAIN_ROOT in output.parents:
            raise RuntimeError("Audit output must remain outside the source repository.")
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite audit JSON: {output}")
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
