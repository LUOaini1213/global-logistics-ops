import numpy as np
import pytest

from glops import inventory as inv

ITEM = inv.Item(demand_mean=40.0, demand_sd=12.0, unit_value=400.0, holding_rate=0.25,
                backorder_cost=6.0)
DET = inv.Mode("det", 5.0, 0.0, 1.0, 100.0)


def test_normal_loss_known_values():
    assert inv.normal_loss(0.0) == pytest.approx(0.3989423, abs=1e-6)
    assert inv.normal_loss(1.0) == pytest.approx(0.0833155, abs=1e-6)
    assert inv.normal_loss(-1.0) == pytest.approx(1.0833155, abs=1e-6)   # G(-z) = G(z) + z


@pytest.mark.parametrize("L", [0, 1, 5])
def test_deterministic_demand_timing_is_exact(L):
    """With sd = 0 the protection interval is exactly L + 1 days."""
    item = inv.Item(40.0, 0.0, 100.0, 0.25, 1.0)
    mode = inv.Mode("d", float(L), 0.0, 0.0, 0.0)
    S = np.array([40.0 * (L + 1), 40.0 * L, 40.0 * (L + 0.5), 40.0 * (L - 1)])
    r = inv.simulate("base_stock", {"S": S}, item, mode, 500, 50, 2, 0)
    assert r["fill_rate"][0] == pytest.approx(1.0)
    assert r["fill_rate"][1] == pytest.approx(0.0, abs=1e-12)
    assert r["fill_rate"][2] == pytest.approx(0.5)
    # end-of-day backorders (after receipts) = E[(D_L - S)^+]
    assert r["backorder"][1] == pytest.approx(0.0, abs=1e-12)
    assert r["backorder"][3] == pytest.approx(40.0)
    assert r["on_hand"][0] == pytest.approx(40.0)          # S - L*mu left after the receipt
    assert r["pipeline"][0] == pytest.approx(40.0 * L)


def _run(policy, params, mode=DET, reps=3, days=1500, seed=1, **kw):
    return inv.simulate(policy, params, ITEM, mode, days, 100, reps, seed, **kw)


@pytest.mark.parametrize("policy,params", [
    ("base_stock", {"S": np.array([200.0, 300.0])}),
    ("sS", {"s": np.array([150.0, 250.0]), "S": np.array([400.0, 500.0])}),
    ("rQ", {"r": np.array([150.0, 250.0]), "Q": np.array([200.0, 300.0])}),
])
@pytest.mark.parametrize("mode", [DET, inv.Mode("rand", 8.0, 3.0, 1.0, 100.0),
                                inv.Mode("instant", 0.0, 0.0, 1.0, 100.0)])
def test_invariants(policy, params, mode):
    r = _run(policy, params, mode)
    assert np.all(r["final_balance_error"] < 1e-6)          # NI_end = NI_0 - demand + receipts
    assert np.all(r["pipeline_error"] < 1e-6)               # on-order == scheduled arrivals
    assert np.all(r["on_hand"] >= 0) and np.all(r["backorder"] >= 0)
    assert np.all((r["fill_rate"] >= 0) & (r["fill_rate"] <= 1))
    # long-run conservation: what is ordered is what is demanded (+/- one cycle)
    rel = np.abs(r["units_ordered_per_day"] - r["demand_per_day"]) / r["demand_per_day"]
    assert np.all(rel < 0.05)
    # Little's law on the pipeline: avg on-order ~ order rate x mean lead time
    assert np.allclose(r["pipeline"], r["units_ordered_per_day"] * mode.lt_mean, rtol=0.12)


def test_rq_orders_are_multiples_of_q():
    Q = 173.0
    r = _run("rQ", {"r": np.array([300.0]), "Q": np.array([Q])})
    per_order = r["units_ordered_per_day"] / np.maximum(r["orders_per_day"], 1e-12)
    k = per_order / Q
    assert np.allclose(k, np.round(k), atol=1e-9) and np.all(k >= 1)


def test_mixed_same_day_and_future_receipts_preserve_inventory(monkeypatch):
    """Same-day supply clears today's backlog, but cannot undo a lost immediate fill."""
    item = inv.Item(10.0, 0.0, 100.0, 0.25, 2.0)
    mode = inv.Mode("mixed", 1.0, 0.0, 0.0, 0.0)
    monkeypatch.setattr(inv, "draw_lead_times",
                        lambda rng, mode, shape: np.array([[0, 1, 0, 1]]))
    result = inv.simulate("base_stock", {"S": np.array([10.0])}, item, mode,
                          n_days=4, warmup=0, reps=1, seed=0)
    # End-of-day stock: 10, 0, 10, 0; pipeline: 0, 10, 0, 10.
    assert result["fill_rate"][0, 0] == pytest.approx(0.75)
    assert result["on_hand"][0, 0] == pytest.approx(5.0)
    assert result["pipeline"][0, 0] == pytest.approx(5.0)
    assert result["backorder"][0, 0] == pytest.approx(0.0)
    assert result["final_balance_error"][0, 0] == pytest.approx(0.0)
    assert result["pipeline_error"][0, 0] == pytest.approx(0.0)


@pytest.mark.parametrize("mean,sd", [(-1, 0), (1, -1), (0, 1),
                                     (np.nan, 0), (1, np.nan), (np.inf, 0), (1, np.inf)])
def test_invalid_lead_time_distribution_is_rejected(mean, sd):
    mode = inv.Mode("invalid", mean, sd, 0.0, 0.0)
    with pytest.raises(ValueError, match="lead time"):
        inv.draw_lead_times(np.random.default_rng(0), mode, (1, 3))


def test_ss_orders_less_often_than_base_stock():
    bs = _run("base_stock", {"S": np.array([300.0])})
    ss = _run("sS", {"s": np.array([300.0]), "S": np.array([700.0])})
    assert ss["orders_per_day"].mean() < 0.2 * bs["orders_per_day"].mean()


def test_more_stock_more_service():
    S = np.array([200.0, 240.0, 280.0, 320.0])
    r = _run("base_stock", {"S": S})
    fr = r["fill_rate"].mean(axis=1)
    oh = r["on_hand"].mean(axis=1)
    assert np.all(np.diff(fr) > 0) and np.all(np.diff(oh) > 0)


def test_costs_add_up():
    r = _run("rQ", {"r": np.array([250.0]), "Q": np.array([300.0])})
    parts = sum(r[f"cost_{k}"] for k in ("holding", "pipeline", "backorder", "ordering", "freight"))
    assert np.allclose(parts, r["cost_total"])
    assert np.allclose(r["cost_holding"], r["on_hand"] * ITEM.h)
    assert np.allclose(r["cost_freight"], r["units_ordered_per_day"] * DET.freight_per_unit)


def test_backorders_are_charged_not_hidden():
    """Starved system: stock never goes negative; the shortfall shows up as backorders."""
    r = _run("base_stock", {"S": np.array([0.0])})
    assert np.all(r["on_hand"] < 5.0)
    assert np.all(r["backorder"] > 4 * 40.0)
    assert np.all(r["cost_backorder"] == pytest.approx(r["backorder"] * ITEM.backorder_cost))


@pytest.mark.parametrize("z", [0.0, 1.5])
def test_analytic_vs_simulation(z):
    """Exact normal-demand base-stock formulas vs simulation (allow negative demand so the
    normal model is exact). Tolerance: 4 standard errors across replications."""
    L = 5
    S = ITEM.demand_mean * (L + 1) + z * ITEM.demand_sd * np.sqrt(L + 1)
    mode = inv.Mode("d", float(L), 0.0, 0.0, 0.0)
    r = inv.simulate("base_stock", {"S": np.array([S])}, ITEM, mode, 8000, 100, 8, 3,
                     allow_negative=True)
    a = inv.analytic_base_stock(S, ITEM, L)
    for m in ("fill_rate", "csl", "on_hand", "backorder"):
        x = r[m][0]
        se = x.std(ddof=1) / np.sqrt(len(x))
        assert abs(x.mean() - a[m]) < 4 * se + 1e-9, (m, x.mean(), a[m], se)


def test_seasonal_bucket_fill():
    T = 1600
    bucket = (np.arange(T) // 400) % 4
    r = _run("base_stock", {"S": np.array([280.0])}, bucket=bucket, days=1500)
    assert r["bucket_fill_rate"].shape == (1, 4)
    assert np.all((r["bucket_fill_rate"] >= 0) & (r["bucket_fill_rate"] <= 1))


def test_holding_cost_convention():
    """Holding cost per unit-day = value x annual rate / 365 (documented convention)."""
    assert ITEM.h == pytest.approx(400.0 * 0.25 / 365.0)


def test_rq_lifts_position_above_reorder_point():
    """Deterministic demand: if every order lifts IP above r = mu (L+1), stock never runs out."""
    item = inv.Item(40.0, 0.0, 100.0, 0.25, 1.0)
    mode = inv.Mode("d", 5.0, 0.0, 0.0, 0.0)
    r = inv.simulate("rQ", {"r": np.array([240.0]), "Q": np.array([200.0])}, item, mode, 600, 60, 1, 0)
    assert r["fill_rate"][0, 0] == pytest.approx(1.0)
    assert r["orders_per_day"][0, 0] == pytest.approx(40.0 / 200.0, rel=0.02)
