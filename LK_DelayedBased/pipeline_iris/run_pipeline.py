#!/usr/bin/env python3
"""
Iris pipeline entry point.

Runs a fab-parameter sweep over the Iris reservoir benchmark, then
optionally trains a fab-recommender model on the results.

All outputs land under pipeline_iris/outputs/.

Usage examples
--------------
# Quick sweep (4 configs, ~2 min on CPU; seconds on GPU):
python pipeline_iris/run_pipeline.py --quick --sweep-only

# Quick sweep with GPU acceleration:
python pipeline_iris/run_pipeline.py --quick --gpu

# Full sweep + recommender training:
python pipeline_iris/run_pipeline.py

# Standard grid (slow, ~7 000 configs):
python pipeline_iris/run_pipeline.py --standard-grid --gpu

# Latin-Hypercube sampling with 48 points:
python pipeline_iris/run_pipeline.py --lhc --n-lhc 48 --gpu

See pipeline_common.run_cli for all available flags.
"""

from __future__ import annotations

import sys
from pathlib import Path

PIPE = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPE.parent))

from pipeline_common import run_cli  # noqa: E402

if __name__ == "__main__":
    run_cli(PIPE)
