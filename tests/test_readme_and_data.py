"""README numbers are pinned to results/summary.json; public data snapshot is intact."""
import csv

import numpy as np
import pytest

from glops import public_data, readme_numbers
from glops.common import DATA, RESULTS, ROOT, read_json


def test_readme_numbers_match_results():
    summary = read_json(RESULTS / "summary.json")
    expected = readme_numbers.numbers(summary, readme_numbers.load_mutation())
    found = readme_numbers.extract((ROOT / "README.md").read_text(encoding="utf-8"))
    assert found, "README has no generated-number markers"
    assert set(found) == set(expected), set(found) ^ set(expected)
    for k, v in expected.items():
        assert found[k] == v, f"README {k}: {found[k]!r} != results {v!r}"


def test_readme_numbers_json_in_sync():
    summary = read_json(RESULTS / "summary.json")
    assert read_json(RESULTS / "readme_numbers.json") == readme_numbers.numbers(
        summary, readme_numbers.load_mutation())


def test_public_snapshot_and_licence():
    rows = list(csv.DictReader(open(DATA / "public" / "container_throughput_monthly.csv", encoding="utf-8")))
    months = [r["month"] for r in rows]
    assert months == sorted(months) and len(set(months)) == len(months)
    assert months[0] == "1995-01"
    # unit check: 2024 total equals SingStat's 41,124.1 thousand TEU
    assert public_data.annual_totals()[2024] == pytest.approx(41124.1, abs=1.0)
    lic = (DATA / "public" / "LICENCE_SODL_v1.0.txt").read_text(encoding="utf-8")
    assert "Singapore Open Data Licence" in lic and "Attribution" in lic
    meta = public_data.meta()
    assert "Singapore Open Data Licence version 1.0" in meta["attribution"]


def test_seasonal_index_normalised():
    idx = public_data.seasonal_index()
    assert idx.shape == (12,) and np.isclose(idx.mean(), 1.0)
    assert idx.argmin() == 1          # February (Lunar New Year) is the weakest month
