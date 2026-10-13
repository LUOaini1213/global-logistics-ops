"""Shared helpers: confidence intervals, JSON output, thread limits."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CONFIG = ROOT / "config"
DATA = ROOT / "data"


def limit_threads(n: int = 2) -> None:
    """Keep BLAS/OpenMP pools small: this runs on a shared machine."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, str(n))


def mean_ci(x, level: float = 0.95) -> dict:
    """Student-t confidence interval of the mean over independent replications."""
    a = np.asarray(x, dtype=float)
    n = a.size
    m = float(a.mean())
    if n < 2:
        return {"mean": m, "lo": m, "hi": m, "n": int(n), "sd": 0.0}
    sd = float(a.std(ddof=1))
    half = float(stats.t.ppf(0.5 + level / 2, n - 1) * sd / math.sqrt(n))
    return {"mean": m, "lo": m - half, "hi": m + half, "n": int(n), "sd": sd}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_default)
        f.write("\n")


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o)}")
