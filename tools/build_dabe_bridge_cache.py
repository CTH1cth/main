#!/usr/bin/env python3
"""Build the GT-free DABE Bridge target cache."""

import argparse
import sys
from pathlib import Path

MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.dabe_clean_cache import (
    build_bridge_payload,
    generate_target_cache,
    parse_bool,
    select_source_rows,
)


DEFAULT_INPUT = "../datasets/cache/dabe_pu_v11_pseudo_cache/dinov1-s8"
DEFAULT_OUTPUT = "../datasets/cache/dabe_bridge_v1_pseudo_cache/dinov1-s8"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=DEFAULT_INPUT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT)
    parser.add_argument("--datasets", default="TR-CAMO,TR-COD10K")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    args = parser.parse_args()

    rows = select_source_rows(args.input_root, args.datasets, args.max_samples)
    manifest, manifest_rows = generate_target_cache(
        rows,
        args.output_root,
        build_bridge_payload,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print("TRAIN_GT_READ=False")
    print("TRAIN_GT_WRITTEN=False")
    print(f"cache_kind=dabe_bridge_v1 samples={len(manifest_rows)}")
    print(f"output_root={Path(args.output_root).resolve()}")
    print(f"manifest={manifest.resolve()}")


if __name__ == "__main__":
    main()
