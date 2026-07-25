#!/usr/bin/env python3
"""Build GT-free DABE Clean or v2 continuous-recovery caches."""

import argparse
import sys
from pathlib import Path

MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.dabe_clean_cache import (
    attach_feature_cache_rows,
    build_clean_payload,
    build_contrec_payload,
    generate_target_cache,
    parse_bool,
    select_source_rows,
)
from common.utils import feature_manifest_path, load_config


DEFAULT_INPUT = "../datasets/cache/dabe_pu_v11_pseudo_cache/dinov1-s8"
DEFAULT_OUTPUT = "../datasets/cache/dabe_clean_v1_pseudo_cache/dinov1-s8"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--force-new-version", action="store_true")
    parser.add_argument("--input-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    args = parser.parse_args()

    cfg = load_config(args.config) if args.config else None
    if args.force_new_version and cfg is None:
        parser.error("--force-new-version requires --config.")
    if cfg is not None and args.force_new_version:
        expected = "dabe_clean_v2_contrec"
        actual = str(
            getattr(cfg, "DABE_CLEAN_EXPECTED_PAYLOAD_VERSION", "")
        )
        if actual != expected:
            parser.error(
                "--force-new-version requires a v2-contrec config with "
                f"DABE_CLEAN_EXPECTED_PAYLOAD_VERSION={expected!r}."
            )
        input_root = args.input_root or str(
            getattr(cfg, "DABE_CLEAN_SOURCE_ROOT", DEFAULT_INPUT)
        )
        output_root = args.output_root or str(cfg.DABE_CLEAN_ROOT)
        datasets = args.datasets or ",".join(str(x) for x in cfg.TRAIN_DATASETS)
        rows = select_source_rows(input_root, datasets, args.max_samples)
        rows = attach_feature_cache_rows(
            rows,
            feature_manifest_path(cfg, "train"),
            margin_tau=float(cfg.ECST_CLEAN_MARGIN_TAU),
        )
        builder = build_contrec_payload
        cache_kind = "dabe_clean_v2_contrec"
    else:
        input_root = args.input_root or DEFAULT_INPUT
        output_root = args.output_root or DEFAULT_OUTPUT
        datasets = args.datasets or "TR-CAMO,TR-COD10K"
        rows = select_source_rows(input_root, datasets, args.max_samples)
        builder = build_clean_payload
        cache_kind = "dabe_clean_v1"
    manifest, manifest_rows = generate_target_cache(
        rows,
        output_root,
        builder,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print("TRAIN_GT_READ=False")
    print("TRAIN_GT_WRITTEN=False")
    print("TEACHER_PREDICTION_READ=False")
    print(f"cache_kind={cache_kind} samples={len(manifest_rows)}")
    print(f"output_root={Path(output_root).resolve()}")
    print(f"manifest={manifest.resolve()}")


if __name__ == "__main__":
    main()
