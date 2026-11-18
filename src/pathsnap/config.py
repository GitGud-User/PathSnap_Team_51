"""Project-wide paths and constants.

The data root can be overridden with the PATHSNAP_DATA_DIR environment variable,
e.g. when the evaluator data is copied somewhere else on Laptop 1.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("PATHSNAP_DATA_DIR", PROJECT_ROOT / "data"))
RESULTS_DIR = PROJECT_ROOT / "results"

# Partition name -> folder name under DATA_DIR
PARTITIONS = {
    "train": "train",
    "student_test": "student_test",
    "evaluator_test": "evaluator_test",
}

# Partitions whose reference routes may be used during development.
# evaluator_test routes are only ever read by the server-side evaluator (Stage 9).
DEV_PARTITIONS = ("train", "student_test")

RANDOM_SEED = 42
