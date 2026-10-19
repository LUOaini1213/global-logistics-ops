import copy
import json

import numpy as np
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


@pytest.fixture(scope="module")
def optimal_flows():
    return rv.build_and_solve(NET)["flows"]


@pytest.mark.parametrize("part", ["x", "y", "f", "s", "p", "o"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, "1", None, True, 1j, [1]])
def test_flow_input_rejects_nonfinite_and_non_numeric_quantities(optimal_flows, part, value):
    flows = copy.deepcopy(optimal_flows)
    flows[part][next(iter(flows[part]))] = value
    bad = rv.check_flows(NET, flows)
    assert bad and any(part in message and "finite real" in message for message in bad)


@pytest.mark.parametrize("part,key", [
    ("x", ("Ghost hub", "Ghost refurb", "A")),
    ("x", ("Singapore", "Singapore refurb", "C")),
    ("x", ("Singapore", "Singapore refurb", "Z")),
    ("x", ("Singapore",)),
    ("x", "Singapore"),
    ("y", ("Ghost hub", "Ghost recycler", "A")),
    ("y", ("Singapore", "SG recycler", "Z")),
    ("y", ("Singapore", "SG recycler", "A", "extra")),
    ("f", ("Ghost refurb", "Ghost recycler")),
    ("f", ("Singapore", "SG recycler")),
    ("s", "Ghost refurb"),
    ("p", ("Singapore refurb",)),
    ("o", ("Ghost hub", "A")),
    ("o", ("Singapore", "fail")),
    ("o", ("Singapore refurb", "A")),
])
def test_flow_input_rejects_illegal_keys_even_at_zero(optimal_flows, part, key):
    flows = copy.deepcopy(optimal_flows)
    flows[part][key] = 0.0
    bad = rv.check_flows(NET, flows)
    assert bad and any("invalid" in message and part in message for message in bad)


@pytest.mark.parametrize("value", [None, [], {"x": []}, {"extra": {}}])
def test_flow_input_requires_known_mapping_families(value):
    bad = rv.check_flows(NET, value)
    assert bad and all(isinstance(message, str) for message in bad)


@pytest.mark.parametrize("tol", [np.nan, np.inf, -1, "0.001", True])
def test_flow_input_rejects_invalid_tolerance(optimal_flows, tol):
    assert any("tolerance" in message for message in rv.check_flows(NET, optimal_flows, tol=tol))


def balanced_small_network():
    """Hand-balanced plan: ten A returns yield eight resales and two failures; five C returns recycle."""
    net = {
        "hubs": {"hub": {"country": "SG", "returns": {"A": 10, "B": 0, "C": 5}}},
        "refurb_centres": {"repair": {"country": "SG", "capacity": 20, "yield": {"A": 0.8, "B": 0.5}}},
        "recyclers": {"domestic": {"country": "SG", "capacity": 20},
                      "abroad": {"country": "MY", "capacity": 20}},
        "market": {"resale_demand_cap": 20},
        "cross_border": {"allow_cross_border_recycling": False},
    }
    flows = {"x": {("hub", "repair", "A"): 10}, "y": {("hub", "domestic", "C"): 5},
             "f": {("repair", "domestic"): 2}, "s": {"repair": 8}}
    return net, flows


@pytest.mark.parametrize("part", ["y", "f"])
def test_forbidden_waste_is_rejected_even_when_balances_and_capacities_hold(part):
    net, flows = balanced_small_network()
    assert rv.check_flows(net, flows) == []
    old = next(iter(flows[part]))
    foreign = (old[0], "abroad", *old[2:])
    flows[part][foreign] = flows[part].pop(old)
    assert any("cross-border" in message for message in rv.check_flows(net, flows))
    net["cross_border"]["allow_cross_border_recycling"] = True
    assert rv.check_flows(net, flows) == []


def test_cross_border_repair_and_zero_waste_arcs_remain_allowed():
    net, flows = balanced_small_network()
    net["refurb_centres"]["repair"]["country"] = "MY"
    flows["f"] = {("repair", "abroad"): 2, ("repair", "domestic"): 0}
    flows["y"]["hub", "abroad", "C"] = 0
    assert rv.check_flows(net, flows) == []


@pytest.mark.parametrize("scenario", ["A", "B", "all_returns", "one_recycler", "all_recyclers"])
def test_zero_activity_summaries_are_finite_for_solver_and_greedy(scenario):
    net = copy.deepcopy(NET)
    if scenario in ("A", "B", "all_returns"):
        grades = ("A", "B", "C") if scenario == "all_returns" else (scenario,)
        for hub in net["hubs"].values():
            for grade in grades:
                hub["returns"][grade] = 0
    else:
        names = list(net["recyclers"]) if scenario == "all_recyclers" else ["SG recycler"]
        for name in names:
            net["recyclers"][name]["capacity"] = 0
    result = rv.build_and_solve(net)
    assert result["status"] == "Optimal"
    for flows in (result["flows"], rv.greedy_baseline(net)):
        assert rv.check_flows(net, flows) == []
        with np.errstate(divide="raise", invalid="raise"):
            summary = rv.summarise(net, flows)
        json.dumps(summary, allow_nan=False)
        for grade in ("A", "B"):
            if summary["returns"][grade] == 0:
                assert summary[f"refurb_share_{grade}"] == 0.0
        for name, facility in net["recyclers"].items():
            if facility["capacity"] == 0:
                assert summary["recycler_utilisation"][name] == 0.0


def test_sparse_zero_flow_is_valid_but_throughput_at_closed_facility_is_not():
    empty = rv.with_changes(NET, volume_mult=0)
    for facility in empty["refurb_centres"].values():
        facility["capacity"] = 0
    for facility in empty["recyclers"].values():
        facility["capacity"] = 0
    assert rv.check_flows(empty, {}) == []
    summary = rv.summarise(empty, {})
    json.dumps(summary, allow_nan=False)
    assert summary["refurb_share_A"] == summary["refurb_share_B"] == 0.0
    assert set(summary["refurb_utilisation"].values()) == {0.0}
    assert set(summary["recycler_utilisation"].values()) == {0.0}
    net, flows = balanced_small_network()
    net["recyclers"]["domestic"]["capacity"] = 0
    assert "recycler cap domestic" in rv.check_flows(net, flows)
    with pytest.raises(ValueError, match="infeasible.*recycler cap domestic"):
        rv.summarise(net, flows)
