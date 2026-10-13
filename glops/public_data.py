"""Part 5 - optional public-data enrichment (MPA monthly container throughput).

Used only as a *proxy* seasonality profile for a demand-seasonality stress test of
the inventory policy. Port throughput is not SKU demand; the README says so.
Licence: Singapore Open Data Licence v1.0 (data/public/LICENCE_SODL_v1.0.txt).
"""
from __future__ import annotations

import csv

import numpy as np

from .common import DATA, read_json

CSV = DATA / "public" / "container_throughput_monthly.csv"
META = DATA / "public" / "container_throughput_monthly.meta.json"


def load():
    months, vals = [], []
    with open(CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            months.append(row["month"])
            vals.append(float(row["container_throughput_thousand_teu"]))
    return months, np.array(vals)


def meta() -> dict:
    return read_json(META)


def seasonal_index(first_year: int = 2010, last_year: int = 2025) -> np.ndarray:
    """Multiplicative month-of-year index (mean 1) by ratio to a centred 2x12 moving
    average, averaged over complete years first_year..last_year."""
    months, v = load()
    years = np.array([int(m[:4]) for m in months])
    mon = np.array([int(m[5:7]) for m in months])
    keep = (years >= first_year - 1) & (years <= last_year + 1)
    y, mo, vv = years[keep], mon[keep], v[keep]
    ma12 = np.convolve(vv, np.ones(12) / 12, mode="valid")          # len n-11
    cma = (ma12[:-1] + ma12[1:]) / 2                                # centred at t=6..n-7
    ratio = np.full(len(vv), np.nan)
    ratio[6:6 + len(cma)] = vv[6:6 + len(cma)] / cma
    sel = (y >= first_year) & (y <= last_year)
    idx = np.array([np.nanmean(ratio[sel & (mo == m)]) for m in range(1, 13)])
    return idx / idx.mean()


def annual_totals() -> dict:
    months, v = load()
    out: dict = {}
    for m, x in zip(months, v):
        out.setdefault(int(m[:4]), []).append(x)
    return {y: float(np.sum(x)) for y, x in out.items() if len(x) == 12}
