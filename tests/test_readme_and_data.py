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


def test_testing_doc_counts_match_mutation_reports():
    """docs/TESTING.md quotes the mutant counts in prose; pin them to the reports and the tool."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("mutation_test", ROOT / "tools" / "mutation_test.py")
    mt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mt)
    doc = (ROOT / "docs" / "TESTING.md").read_text(encoding="utf-8")
    now, first = read_json(RESULTS / "mutation_report.json"), read_json(RESULTS / "mutation_report_round1.json")
    part5 = sum(f in ("glops/rollout.py", "config/rollout.json") for f, *_ in mt.MUTANTS)
    real = len(mt.MUTANTS) - first["total"] - part5
    assert now["total"] == len(mt.MUTANTS) and now["caught"] == now["total"] and not now.get("invalid")
    assert f"({now['caught']}/{now['total']} caught, 0 invalid)" in doc
    assert f"## Round 1: {first['total']} mutants" in doc and f"caught {first['caught']}/{first['total']}" in doc
    assert f"{real} more mutants" in doc and f"{part5} more mutants" in doc


def test_check_results_tolerates_float_noise_only():
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_results", ROOT / "tools" / "check_results.py")
    cr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cr)
    D = cr.Diff(1e-6, 1e-9)
    D.tree("j", {"a": 1.0, "b": [2.0, "x"], "c": True}, {"a": 1.0 + 1e-12, "b": [2.0, "x"], "c": True})
    D.csv("c", "k,v\na,0.123456\n", "k,v\na,0.123456\n")
    assert D.errors == []
    for old, new in (({"a": 1.0}, {"a": 1.001}), ({"a": "x"}, {"a": "y"}), ({"a": 1}, {"b": 1}),
                     ({"a": [1, 2]}, {"a": [1]}), ({"a": True}, {"a": False})):
        D = cr.Diff(1e-6, 1e-9)
        D.tree("j", old, new)
        assert D.errors, (old, new)
    for old, new in (("k,v\na,1.0\n", "k,v\na,1.1\n"), ("k,v\na,1\n", "k,w\na,1\n"),
                     ("k,v\na,1\n", "k,v\na,1\nb,2\n")):
        D = cr.Diff(1e-6, 1e-9)
        D.csv("c", old, new)
        assert D.errors, (old, new)
