#!/usr/bin/env python3
"""Entrypoint for the frozen four-stage GBSP teaser sample miner."""

import sys
from pathlib import Path


MAIN_ROOT = Path(__file__).resolve().parents[1]
if str(MAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(MAIN_ROOT))

from tools.gbsp_teaser_sample_mining.pipeline import main  # noqa: E402


if __name__ == "__main__":
    main()
