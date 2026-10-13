import pytest

from glops import reverse as rv

NET = rv.load_network()


def test_base_case_feasible_and_optimal():
    r = rv.build_and_solve(NET)
    assert r["status"] == "Optimal"
    assert rv.check_flows(NET, r["flows"]) == []
    assert rv.evaluate_flows(NET, r["flows"]) == pytest.approx(r["net_cost"], rel=1e-9)


def test_lp_beats_or_matches_greedy():
    g = rv.greedy_baseline(NET)
    assert rv.check_flows(NET, g) == []
    assert rv.build_and_solve(NET)["net_cost"] <= rv.evaluate_flows(NET, g) + 1e-6


def test_check_flows_catches_violation():
    r = rv.build_and_solve(NET)
    f = {k: dict(v) for k, v in r["flows"].items()}
    k = next(iter(f["x"]))
    f["x"][k] += 10.0
    assert rv.check_flows(NET, f)


@pytest.mark.parametrize("centre", ["Penang refurb", "Singapore refurb"])
def test_capacity_dual_matches_finite_difference(centre):
    base = rv.build_and_solve(NET)
    cap = NET["refurb_centres"][centre]["capacity"]
    up = rv.build_and_solve(rv.with_changes(NET, refurb_capacity={centre: cap + 1}))
    assert up["net_cost"] - base["net_cost"] == pytest.approx(base["duals"][f"refurb_cap|{centre}"], abs=1e-6)


def test_no_value_no_refurbishment():
    n = rv.with_changes(NET, resale_price=0.0)
    n["market"]["parts_value"] = 0.0
    r = rv.build_and_solve(n)
    assert sum(r["flows"]["x"].values()) == pytest.approx(0.0, abs=1e-6)


def test_cross_border_flag_respected():
    r = rv.build_and_solve(rv.with_changes(NET, allow_cross_border_recycling=False))
    assert r["status"] == "Optimal"
    assert r["summary"]["cross_border_waste_units"] == pytest.approx(0.0, abs=1e-9)
    assert r["net_cost"] >= rv.build_and_solve(NET)["net_cost"] - 1e-6


def test_overflow_only_when_capacity_short():
    assert rv.build_and_solve(NET)["summary"]["overflow"] == pytest.approx(0.0, abs=1e-6)
    big = rv.build_and_solve(rv.with_changes(NET, volume_mult=2.0))
    assert big["status"] == "Optimal" and big["summary"]["overflow"] > 0


def test_higher_price_never_reduces_refurbishment():
    prev = -1.0
    for p in (20, 40, 60, 100, 150, 250):
        s = rv.build_and_solve(rv.with_changes(NET, resale_price=p))["summary"]
        tot = s["refurb_share_A"] * s["returns"]["A"] + s["refurb_share_B"] * s["returns"]["B"]
        assert tot >= prev - 1e-6
        prev = tot
