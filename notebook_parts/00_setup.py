import json
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 120)

SEED = 42
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

EXPECTED_NODES = 256
PCA_VARIANCE_TARGET = 0.90
K_VALUES = range(2, 11)
N_INIT = 20
N_STABILITY = 30
N_PERMUTATIONS = 2000

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("RAW directory:", RAW_DIR.resolve())
print("raw_trace.json files:", len(list(RAW_DIR.rglob("raw_trace.json"))))
