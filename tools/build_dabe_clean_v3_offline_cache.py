#!/usr/bin/env python3
"""Build deterministic GT-free DABE-Clean v3 offline-consolidation caches."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from common.dabe_clean_offline import (  # noqa: E402
    DABE_CLEAN_OFFLINE_MODES,
    DABE_CLEAN_OFFLINE_PAYLOAD_VERSION,
)
from common.dabe_clean_offline_cache import (  # noqa: E402
    generate_offline_cache,
    select_joined_rows,
)
from common.dabe_clean_cache import parse_bool  # noqa: E402
from common.utils import load_config  # noqa: E402


DEFAULT_SOURCE = (
    "../datasets/cache/dabe_clean_v1_a1_residual_only_pseudo_cache/dinov1-s8"
)
DEFAULT_SEMANTIC_SOURCE = (
    "../datasets/cache/dabe_clean_v2_contrec_a1_residual_only_pseudo_cache/"
    "dinov1-s8"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--semantic-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--datasets", default=None)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if str(getattr(cfg, "DABE_CLEAN_VERSION", "")) != "v3_offline_consolidation":
        parser.error("--config must be a DABE-Clean v3 offline configuration.")
    expected = str(getattr(cfg, "DABE_CLEAN_EXPECTED_PAYLOAD_VERSION", ""))
    if expected != DABE_CLEAN_OFFLINE_PAYLOAD_VERSION:
        parser.error(
            "DABE_CLEAN_EXPECTED_PAYLOAD_VERSION must be "
            f"{DABE_CLEAN_OFFLINE_PAYLOAD_VERSION!r}."
        )
    mode = str(getattr(cfg, "DABE_CLEAN_OFFLINE_MODE", "")).strip().lower()
    if mode not in DABE_CLEAN_OFFLINE_MODES:
        parser.error(
            f"Unsupported DABE_CLEAN_OFFLINE_MODE={mode!r}; "
            f"allowed={sorted(DABE_CLEAN_OFFLINE_MODES)}."
        )
    if float(getattr(cfg, "DABE_BC_LAMBDA", -1.0)) != 0.0:
        parser.error("Pure-offline v3 requires DABE_BC_LAMBDA=0.0.")
    forbidden_training_flags = {
        "USE_ECST": bool(getattr(cfg, "USE_ECST", False)),
        "USE_ECST_MINIMAL": bool(getattr(cfg, "USE_ECST_MINIMAL", False)),
        "USE_ECST_CLEAN": bool(getattr(cfg, "USE_ECST_CLEAN", False)),
        "DABE_CLEAN_USE_LEGACY_ECST_REGIONS": bool(
            getattr(cfg, "DABE_CLEAN_USE_LEGACY_ECST_REGIONS", False)
        ),
    }
    enabled = [name for name, value in forbidden_training_flags.items() if value]
    if enabled or str(getattr(cfg, "TEACHER_ROUTING_MODE", "none")) != "none":
        parser.error(
            "Pure-offline cache configs must disable all ECST routing; "
            f"enabled={enabled}, TEACHER_ROUTING_MODE="
            f"{getattr(cfg, 'TEACHER_ROUTING_MODE', None)!r}."
        )
    source_root = args.source_root or str(
        getattr(cfg, "DABE_CLEAN_OFFLINE_SOURCE_ROOT", DEFAULT_SOURCE)
    )
    semantic_root = args.semantic_root or str(
        getattr(
            cfg,
            "DABE_CLEAN_OFFLINE_SEMANTIC_ROOT",
            DEFAULT_SEMANTIC_SOURCE,
        )
    )
    output_root = args.output_root or str(cfg.DABE_CLEAN_ROOT)
    datasets = args.datasets or ",".join(str(item) for item in cfg.TRAIN_DATASETS)
    rows = select_joined_rows(
        source_root, semantic_root, datasets, max_samples=args.max_samples
    )
    if args.max_samples < 0 and len(rows) != 4040:
        raise RuntimeError(
            "Full DABE-Clean v3 cache generation requires exactly 4040 rows; "
            f"selected={len(rows)}. Use a positive --max-samples for diagnostics."
        )
    manifest, manifest_rows = generate_offline_cache(
        rows,
        output_root,
        offline_mode=mode,
        workers=args.workers,
        overwrite=args.overwrite,
    )
    print("TRAIN_GT_READ=False")
    print("TEACHER_CHECKPOINT_READ=False")
    print("STUDENT_CHECKPOINT_READ=False")
    print("EPOCH_DEPENDENT=False")
    print("HISTORY_READ=False")
    print(f"payload_version={DABE_CLEAN_OFFLINE_PAYLOAD_VERSION}")
    print(f"offline_mode={mode}")
    print(f"samples={len(manifest_rows)}")
    print(f"output_root={Path(output_root).resolve()}")
    print(f"manifest={manifest.resolve()}")


if __name__ == "__main__":
    main()
