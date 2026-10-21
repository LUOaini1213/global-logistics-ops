"""Part 5 - overseas rollout: distributions, lead-time model, market rule, ship-date planning,
container loading, mode-policy cost arithmetic and consolidation."""
import copy
import math

import numpy as np
import pytest

from glops import rollout as ro
from glops.common import read_json

CFG = ro.load_config()
LPI = ro.load_lpi()
ROUTES = ro.load_sea_routes()
AIRPORTS = ro.load_airports()
SHARES = ro.load_shares()
MARKETS = CFG["markets"]
CTS = ro.container_types(CFG)
CLR = CFG["containers"]["clearance_m"]


# ----------------------------------------------------------------- distributions and fits

def test_quartile_fit_recovers_lognormal():
    mu, s = 1.3, 0.7
    d = ro.lognorm_from_quartiles(math.exp(mu - ro.Z75 * s), math.exp(mu), math.exp(mu + ro.Z75 * s))
    assert d.a == pytest.approx(mu) and d.b == pytest.approx(s)


def test_median_mean_fit_recovers_lognormal():
    mu, s = 0.4, 1.1
    d = ro.lognorm_from_median_mean(math.exp(mu), math.exp(mu + s * s / 2))
    assert d.a == pytest.approx(mu) and d.b == pytest.approx(s)
    assert d.mean() == pytest.approx(math.exp(mu + s * s / 2))


@pytest.mark.parametrize("args", [(3.0, 2.0, 4.0), (0.0, 1.0, 2.0), (2.0, 2.0, 2.0), (1.0, 3.0, 2.0)])
def test_quartile_fit_rejects_bad_input(args):
    with pytest.raises(ValueError):
        ro.lognorm_from_quartiles(*args)


@pytest.mark.parametrize("args", [(2.0, 2.0), (2.0, 1.5), (0.0, 1.0)])
def test_median_mean_fit_rejects_bad_input(args):
    with pytest.raises(ValueError):
        ro.lognorm_from_median_mean(*args)


@pytest.mark.parametrize("d", [ro.Dist("const", 2.5), ro.Dist("lognorm", 0.8, 0.6), ro.Dist("gamma", 4.0, 1.5),
                               ro.Dist("gamma", 10.0, 5.0, 0.6), ro.Dist("lognorm", 1.0, 0.9, 0.3)])
def test_dist_sample_moments_match_analytic(d):
    x = d.sample(np.random.default_rng(3), 400_000)
    se = math.sqrt(d.var() / x.size) if d.var() > 0 else 1e-12
    assert abs(x.mean() - d.mean()) < 4 * se + 1e-12
    if d.var() > 0:
        m4 = np.mean((x - x.mean()) ** 4)
        assert abs(x.var(ddof=1) - d.var()) < 4 * math.sqrt((m4 - x.var() ** 2) / x.size)


def test_mixture_variance_formula():
    d = ro.Dist("gamma", 8.0, 2.0, 0.25)          # Bernoulli(0.25) x Gamma(mean 8, sd 2)
    assert d.mean() == pytest.approx(2.0)
    assert d.var() == pytest.approx(0.25 * (4.0 + 64.0) - 4.0)


def test_scaled_lognormal_scales_quantiles():
    d = ro.Dist("lognorm", 1.0, 0.5)
    assert math.exp(d.scaled(1.25).a) == pytest.approx(1.25 * math.exp(d.a))
    assert d.scaled(1.25).b == d.b and d.scaled(0.75).mean() == pytest.approx(0.75 * d.mean())


@pytest.mark.parametrize("bad", [("weird", 1.0, 0.0, 1.0), ("gamma", -1.0, 0.5, 1.0), ("gamma", 1.0, -0.5, 1.0),
                                 ("const", 1.0, 0.0, 1.5), ("gamma", 0.0, 1.0, 1.0)])
def test_dist_rejects_bad_parameters(bad):
    with pytest.raises(ValueError):
        ro.Dist(*bad)


def test_haversine_known_distance():
    assert ro.haversine_km(0, 0, 0, 1) == pytest.approx(111.195, abs=0.01)
    assert ro.haversine_km(22.6, 114.3, 22.6, 114.3) == 0.0


# ----------------------------------------------------------------- lead-time model

@pytest.mark.parametrize("m", MARKETS, ids=[m["iso3"] for m in MARKETS])
@pytest.mark.parametrize("mode", ["sea", "air"])
def test_lead_time_monte_carlo_matches_analytic_sum(m, mode):
    comps = (ro.sea_components(m, CFG, LPI, ROUTES) if mode == "sea" else ro.air_components(m, CFG, LPI, AIRPORTS))
    S = ro.sample_components(comps, np.random.default_rng(11), 200_000)
    T = np.sum(list(S.values()), axis=0)
    mean, var = ro.analytic_moments(comps)
    assert abs(T.mean() - mean) < 4 * math.sqrt(var / T.size)
    m4 = np.mean((T - T.mean()) ** 4)
    assert abs(T.var(ddof=1) - var) < 4 * math.sqrt((m4 - T.var() ** 2) / T.size)
    for k, x in S.items():                                     # every component on its own, too
        d = comps[k]
        if d.var() > 0:
            assert abs(x.mean() - d.mean()) < 5 * math.sqrt(d.var() / x.size), k


@pytest.mark.parametrize("m", MARKETS, ids=[m["iso3"] for m in MARKETS])
def test_sea_component_inputs(m):
    c = ro.sea_components(m, CFG, LPI, ROUTES)
    nm = ROUTES[m["port_locode"]]["base"]
    v, call = CFG["sea"]["speed_kn"], CFG["sea"]["port_call_days_per_1000nm"]
    assert c["ocean"].mean() == pytest.approx(nm / (24 * v) + call * nm / 1000)
    row = LPI[m["iso3"]]
    assert c["transshipment"].mean() == pytest.approx(row["transshipment_days_mean_2024"])   # mean is the LPI value
    assert c["transshipment"].p == pytest.approx(min(1.0, row["transshipments_mean_2024"]))
    d = c["import_dwell"]                                     # fitted quartile ratio = published ratio
    q1, q3 = row["ct_import_dwell_q1_2024"], row["ct_import_dwell_q3_2024"]
    assert math.exp(2 * ro.Z75 * d.b) == pytest.approx(q3 / q1)
    assert d.a == pytest.approx(np.mean(np.log([q1, row["ct_import_dwell_median_2024"], q3])))
    ex = c["export_dwell"]
    china = LPI["CHN"]
    assert math.exp(ex.a) == pytest.approx(math.exp(np.mean(np.log([china[f"ct_export_dwell_{s}_2024"]
                                                                     for s in ("q1", "median", "q3")]))))


def test_air_component_inputs():
    m = next(x for x in MARKETS if x["iso3"] == "ITA")
    c = ro.air_components(m, CFG, LPI, AIRPORTS)
    km = ro.haversine_km(*AIRPORTS["SZX"], *AIRPORTS["MXP"])
    assert 8500 < km < 9500
    assert c["flight"].a == pytest.approx(km * CFG["air"]["routing_factor"] / CFG["air"]["block_speed_kmh"] / 24)
    assert c["import_dwell"].mean() == pytest.approx(LPI["ITA"]["air_import_dwell_mean_2024"])
    assert math.exp(c["import_dwell"].a) == pytest.approx(LPI["ITA"]["air_import_dwell_median_2024"])


def test_sensitivity_switches():
    m = next(x for x in MARKETS if x["iso3"] == "DZA")
    base = ro.sea_components(m, CFG, LPI, ROUTES)
    assert ro.sea_components(m, CFG, LPI, ROUTES, transship=False)["transshipment"].mean() == 0.0
    cape = ro.sea_components(m, CFG, LPI, ROUTES, route="avoid_red_sea")
    assert cape["ocean"].mean() > base["ocean"].mean() + 10        # Cape detour adds > 10 days
    up = ro.sea_components(m, CFG, LPI, ROUTES, dwell_scale=1.25)
    assert up["import_dwell"].mean() == pytest.approx(1.25 * base["import_dwell"].mean())
    alt = ro.sea_components(m, CFG, LPI, ROUTES, fit="median_mean")
    assert alt["import_dwell"].mean() == pytest.approx(LPI["DZA"]["ct_import_dwell_mean_2024"])
    with pytest.raises(ValueError):
        ro.sea_components(m, CFG, LPI, ROUTES, fit="nope")


def test_lead_time_summary_shares_and_tail():
    comps = {"export_dwell": ro.Dist("const", 2.0), "ocean": ro.Dist("const", 10.0),
             "import_dwell": ro.Dist("const", 3.0), "inland": ro.Dist("const", 5.0)}
    s = ro.lead_time_summary(ro.sample_components(comps, np.random.default_rng(0), 100), "sea")
    assert s["mean"] == pytest.approx(20.0) and s["p95"] == pytest.approx(20.0)
    assert s["share_port_customs_mean"] == pytest.approx(0.25)
    assert s["share_dest_port_customs_mean"] == pytest.approx(0.15)
    assert s["on_time_if_plan_on_mean"] == pytest.approx(1.0)


# ----------------------------------------------------------------- data and market rule

def test_market_rule_reproduces_config():
    regions, year = CFG["selection"]["regions"], CFG["selection"]["lpi_year"]
    assert ro.select_markets(SHARES, LPI, regions, year) == [m["iso3"] for m in MARKETS]
    assert ro.select_markets(SHARES, LPI, regions, year, "share_2025_pct") == \
        ["IDN", "PAK", "LBN", "DZA", "NGA", "ARG", "ITA"]               # stated in the README
    for iso in ("UZB", "RUS", "UGA", "IRN", "UKR"):                     # README: dropped for missing data
        assert iso not in LPI or not ro.has_dwell_data(LPI[iso], year)
        assert any(r["iso3"] == iso for r in SHARES)


def test_committed_shares_are_derived_only():
    assert [r["rank"] for r in SHARES] == list(range(1, len(SHARES) + 1)) and len(SHARES) == 40
    comb = [r["share_2024_2025_pct"] for r in SHARES]
    assert comb == sorted(comb, reverse=True) and sum(comb) < 100
    with open(ro.RDATA / "comtrade_cn_hs851761_partner_shares.csv", encoding="utf-8") as f:
        header = f.readline()
    assert "value" not in header.lower() and "usd" not in header.lower()


def test_sources_meta_and_routes():
    meta = read_json(ro.RDATA / "sources.meta.json")
    assert meta["accessed"] == "2026-10-21"
    assert {"comtrade_2024", "comtrade_2025", "lpi20", "lpi_survey", "ourairports"} <= set(meta["raw_sha256"])
    for m in MARKETS:
        r = ROUTES[m["port_locode"]]
        assert r["great_circle"] <= r["base"] <= r["avoid_red_sea"]
        assert r["base"] <= 2.0 * r["great_circle"]
    for loc in ("DZALG", "ITGOA"):
        assert "suez" in ROUTES[loc]["passages_base"] and ROUTES[loc]["avoid_red_sea"] > ROUTES[loc]["base"] + 3000


# ----------------------------------------------------------------- ship-date planning

def test_quantile_guarantees_coverage():
    assert ro.quantile(np.array([1.0, 2.0, 3.0, 4.0]), 0.9) == 4.0
    x = np.random.default_rng(5).lognormal(1.0, 1.0, 9973)
    for p in (0.5, 0.8, 0.95, 0.99):
        assert np.mean(x <= ro.quantile(x, p)) >= p


def test_rollout_plan_at_p95_is_on_time_out_of_sample():
    m = next(x for x in MARKETS if x["iso3"] == "NGA")
    comps = ro.sea_components(m, CFG, LPI, ROUTES)
    T = np.sum(list(ro.sample_components(comps, np.random.default_rng(1), 200_000).values()), axis=0)
    r = ro.simulate_rollout(comps, np.random.default_rng(2), 12, 6, 1500, ro.quantile(T, 0.95))
    assert abs(r["on_time"] - 0.95) < 0.01
    assert r["week_all_on_time"] < r["on_time"]                 # shared shipments: whole weeks fail together
    r_mean = ro.simulate_rollout(comps, np.random.default_rng(2), 12, 6, 1500, float(T.mean()))
    assert r_mean["on_time"] < 0.7


def test_rollout_deterministic_timing():
    comps = {"ocean": ro.Dist("const", 20.0), "import_dwell": ro.Dist("const", 5.0), "inland": ro.Dist("const", 3.0)}
    rng = np.random.default_rng(0)
    assert ro.simulate_rollout(comps, rng, 4, 3, 5, 28.0)["on_time"] == 1.0
    assert ro.simulate_rollout(comps, rng, 4, 3, 5, 27.9)["on_time"] == 0.0


# ----------------------------------------------------------------- container loading

def _random_units(seed, heavy=False):
    rng = np.random.default_rng(seed)
    kinds = [u["id"] for u in CFG["site_kit"]["units"]]
    keep = {k for k in kinds if rng.random() < 0.8} or {kinds[0]}
    us = ro.kit_units(CFG, int(rng.integers(1, 25)), keep)
    if heavy:     # make weight bind: 2-3 t pallets
        us = [ro.Unit(u.uid, u.kind, u.l, u.w, u.h, u.kg * 9, u.top_ok, u.max_top_kg * 9, u.value) for u in us]
    return us


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("ct", ["20GP", "40GP", "40HC"])
def test_pack_is_feasible_and_above_lower_bound(seed, ct):
    for heavy in (False, True):
        us = _random_units(seed, heavy)
        boxes = ro.pack(us, CTS[ct], CLR)
        assert ro.check_plan(boxes, us, CLR) == []
        assert sorted(u.uid for b in boxes for u in b.units) == sorted(u.uid for u in us)
        assert all(b.kg <= b.ctype.payload for b in boxes)
        assert all(s.height <= CTS[ct].door_height - CLR + 1e-9 for b in boxes for s in b.stacks)
        assert ro.lower_bound(us, CTS[ct])["best"] <= len(boxes)


def test_weekly_batch_needs_two_boxes_and_bound_proves_it():
    us = ro.kit_units(CFG, CFG["schedule"]["sites_per_week"])
    for ct in ("40GP", "40HC"):
        assert len(ro.pack(us, CTS[ct], CLR)) == 2
        assert ro.lower_bound(us, CTS[ct])["floor"] == 2              # README: 29.0 m2 > 28.0 m2
    one = ro.kit_units(CFG, 1)
    assert len(ro.pack(one, CTS["20GP"], CLR)) == 1 and ro.lower_bound(one, CTS["20GP"])["best"] == 1


def test_lower_bound_floor_argument_by_hand():
    cab = ro.Unit("c", "cabinet", 1.2, 1.0, 1.9, 300, False, 0)        # alone on the floor
    bat = ro.Unit("b", "battery", 1.2, 1.0, 0.8, 600, False, 400)      # floor, carries one
    tops = [ro.Unit(f"t{i}", "radio", 1.2, 1.0, 1.0, 200, True, 300) for i in range(5)]
    ct = ro.CType("toy", 3.6, 1.0, 2.6, 1.0, 2.6, 1e6)                 # floor 3.6 m2 = 3 pallets
    lb = ro.lower_bound([cab, bat] + tops, ct)
    assert lb["floor"] == math.ceil((1.2 + 1.2 + (5 * 1.2 - 1.2) / 2) / 3.6 - 1e-9) == 2
    # three carriers each take one top unit: everything fits one toy box, and the bound must say 1
    bats = [ro.Unit(f"b{i}", "battery", 1.2, 1.0, 0.8, 600, False, 400) for i in range(3)]
    rads = [ro.Unit(f"r{i}", "radio", 1.2, 1.0, 1.0, 200, True, 300) for i in range(3)]
    assert ro.lower_bound(bats + rads, ct)["floor"] == 1
    boxes = ro.pack(bats + rads, ct, 0.05)
    assert len(boxes) == 1 and ro.check_plan(boxes, bats + rads, 0.05) == []


def test_checker_rejects_bad_plans():
    us = ro.kit_units(CFG, 2)
    ct = CTS["40HC"]
    good = ro.pack(us, ct, CLR)
    assert ro.check_plan(good, us, CLR) == []
    bad = copy.deepcopy(good)
    bad[0].stacks[1].x, bad[0].stacks[1].y = bad[0].stacks[0].x, bad[0].stacks[0].y
    assert any("overlap" in e for e in ro.check_plan(bad, us, CLR))
    assert any("exactly once" in e for e in ro.check_plan(good, us[:-1], CLR))
    bad = copy.deepcopy(good)
    bad[0].stacks[0].x = ct.length - 0.5
    assert any("outside" in e for e in ro.check_plan(bad, us, CLR))
    heavy = ro.Unit("h", "x", 1.2, 1.0, 1.0, 30000, True, 0)
    hb = [ro.Box(ct, [ro.Placed([heavy], 0, 0, 1.2, 1.0)])]
    assert any("overweight" in e for e in ro.check_plan(hb, [heavy], CLR))
    a = ro.Unit("a", "x", 1.2, 1.0, 1.3, 100, True, 500)
    b = ro.Unit("b", "x", 1.2, 1.0, 1.3, 100, True, 500)
    tall = [ro.Box(ct, [ro.Placed([a, b], 0, 0, 1.2, 1.0)])]
    assert any("too high" in e for e in ro.check_plan(tall, [a, b], CLR))
    light = ro.Unit("l", "x", 1.2, 1.0, 0.5, 50, True, 100)
    heavy_top = ro.Unit("t", "x", 1.2, 1.0, 0.5, 400, True, 0)
    crush = [ro.Box(ct, [ro.Placed([light, heavy_top], 0, 0, 1.2, 1.0)])]
    assert any("illegal stacking" in e for e in ro.check_plan(crush, [light, heavy_top], CLR))


def test_stacking_rules():
    ct = CTS["40GP"]
    base = ro.Unit("b", "x", 1.2, 1.0, 1.0, 300, True, 300)
    assert ro.can_stack(base, ro.Unit("u", "x", 1.0, 1.2, 1.0, 300, True, 0), ct, CLR)   # rotated footprint ok
    assert not ro.can_stack(base, ro.Unit("u", "x", 1.2, 1.0, 1.0, 301, True, 0), ct, CLR)  # too heavy
    assert not ro.can_stack(base, ro.Unit("u", "x", 1.2, 1.0, 1.3, 100, True, 0), ct, CLR)  # door height
    assert not ro.can_stack(base, ro.Unit("u", "x", 1.2, 0.8, 0.5, 100, True, 0), ct, CLR)  # footprint differs
    assert not ro.can_stack(base, ro.Unit("u", "x", 1.2, 1.0, 0.5, 100, False, 0), ct, CLR)  # may not ride on top
    with pytest.raises(ValueError):
        ro.pack([ro.Unit("t", "x", 1.2, 1.0, 2.4, 100, True, 0)], ct, CLR)                # taller than the door


def test_load_batch_picks_cheapest_plan():
    lb = ro.load_batch(ro.kit_units(CFG, 6), CFG, "IDN")
    assert lb["cost"] == min(lb["all_costs"].values())
    assert lb["n_containers"] == len(lb["boxes"]) and 0 < lb["fill_volume"] < 1
    c = CFG["costs"]
    assert ro.container_cost("20GP", "IDN", CFG) == pytest.approx(
        c["ocean_40hc_usd"]["IDN"] * c["ocean_type_factor"]["20GP"] + c["per_container_handling_usd"])


# ----------------------------------------------------------------- costs and policies

KITS = {"full": ro.Kit.of(ro.kit_units(CFG, 1)),
        "bulky": ro.Kit.of(ro.kit_units(CFG, 1, {u["id"] for u in CFG["site_kit"]["units"] if not u["critical"]})),
        "critical": ro.Kit.of(ro.kit_units(CFG, 1, {u["id"] for u in CFG["site_kit"]["units"] if u["critical"]}))}
FR = {"market": "ITA", "sea_full": 1000.0, "sea_bulky": 800.0, "sites": 4}


def test_air_chargeable_weight():
    assert ro.air_chargeable_kg(ro.Kit(0, 100.0, 1.0), 6000) == pytest.approx(1e6 / 6000)   # volumetric
    assert ro.air_chargeable_kg(ro.Kit(0, 500.0, 1.0), 6000) == 500.0                       # actual
    k = KITS["critical"]
    assert ro.air_freight(k, "ITA", CFG) == pytest.approx(max(k.kg, k.cbm * 1e6 / 6000) * CFG["costs"]["air_usd_per_kg"]["ITA"])


def test_readme_quoted_lpi_values():
    """README quotes these LPI 2.0 (2024) values for Algeria in prose."""
    assert LPI["DZA"]["ct_import_dwell_median_2024"] == 16.7
    assert LPI["DZA"]["air_import_dwell_median_2024"] == 11.8


def test_expedite_pays_the_full_single_shipment_fee():
    T = _const_T(sea=40.0, gate=37.0, air=10.0, air2=9.0)
    c, k = CFG["costs"], KITS["full"]
    r = ro.evaluate_policy("sea_expedite", 0.95, T, KITS, FR, CFG)
    expected = (1000 + c["per_delivery_inland_usd"] + ro.holding(k, 40.0, CFG, False)
                + ro.air_freight(k, "ITA", CFG) * (1 + c["expedite_premium"]) + c["per_shipment_air_usd"]
                + c["per_delivery_inland_usd"] + ro.holding(k, 9.0, CFG, False)
                + ro.holding(k, c["redeploy_days"], CFG, True) + ro.holding(k, 40.0 - 39.0, CFG, True))
    assert r["cost"] == pytest.approx(expected)


def test_holding_arithmetic():
    k, c = KITS["full"], CFG["costs"]
    assert ro.holding(k, 10.0, CFG, False) == pytest.approx(k.value * c["holding_rate_per_year"] / 365 * 10)
    assert ro.holding(k, 10.0, CFG, True) == pytest.approx(
        10 * (k.value * c["holding_rate_per_year"] / 365 + k.cbm * c["storage_usd_per_m3_day"]))


def _const_T(sea=40.0, gate=37.0, air=10.0, air2=10.0, n=1000):
    return {"sea": np.full(n, sea), "sea_gate": np.full(n, gate), "air": np.full(n, air), "air2": np.full(n, air2)}


def test_sea_and_air_policy_cost_by_hand():
    c, k = CFG["costs"], KITS["full"]
    r = ro.evaluate_policy("sea", 0.95, _const_T(), KITS, FR, CFG)
    assert r["on_time"] == 1.0 and r["lead_days"] == 40.0
    assert r["cost"] == pytest.approx(1000 + c["per_delivery_inland_usd"] + ro.holding(k, 40.0, CFG, False))
    r = ro.evaluate_policy("air", 0.95, _const_T(), KITS, FR, CFG)
    assert r["cost"] == pytest.approx(ro.air_freight(k, "ITA", CFG) + c["per_shipment_air_usd"] / 4
                                      + c["per_delivery_inland_usd"] + ro.holding(k, 10.0, CFG, False))
    T = _const_T()                                                   # early arrival is charged with storage
    T["sea"] = np.r_[np.full(500, 30.0), np.full(500, 40.0)]
    r = ro.evaluate_policy("sea", 0.95, T, KITS, FR, CFG)
    early = ro.holding(k, 10.0, CFG, True) / 2
    assert r["cost"] == pytest.approx(1000 + c["per_delivery_inland_usd"] + ro.holding(k, 35.0, CFG, False) + early)


def test_split_policy_joint_on_time_and_cost():
    rng = np.random.default_rng(4)
    T = {"sea": rng.gamma(9.0, 5.0, 200_000), "air": rng.gamma(9.0, 1.0, 200_000)}
    r = ro.evaluate_policy("split", 0.9, T, KITS, FR, CFG)
    assert abs(r["on_time"] - 0.9) < 0.005                           # legs at sqrt(0.9) each, independent
    Tc = _const_T()
    r = ro.evaluate_policy("split", 0.95, Tc, KITS, FR, CFG)
    c = CFG["costs"]
    assert r["cost"] == pytest.approx(800 + ro.air_freight(KITS["critical"], "ITA", CFG) + c["per_shipment_air_usd"] / 4
                                      + 2 * c["per_delivery_inland_usd"] + ro.holding(KITS["bulky"], 40.0, CFG, False)
                                      + ro.holding(KITS["critical"], 10.0, CFG, False))


def test_expedite_rule():
    # decision day = L_sea - L_air = 40 - 10 = 30; gate-out at 37 > 30 -> expedite, air kit lands on day 40
    T = _const_T(sea=40.0, gate=37.0, air=10.0, air2=9.0)
    r = ro.evaluate_policy("sea_expedite", 0.95, T, KITS, FR, CFG)
    assert r["expedite_rate"] == 1.0 and r["on_time"] == 1.0
    T = _const_T(sea=40.0, gate=25.0)                                # out of port before day 30: no expedite
    r = ro.evaluate_policy("sea_expedite", 0.95, T, KITS, FR, CFG)
    assert r["expedite_rate"] == 0.0
    assert r["cost"] == pytest.approx(ro.evaluate_policy("sea", 0.95, T, KITS, FR, CFG)["cost"])
    late = {"sea": np.r_[np.full(900, 40.0), np.full(100, 80.0)], "sea_gate": np.r_[np.full(900, 25.0), np.full(100, 70.0)],
            "air": np.full(1000, 10.0), "air2": np.full(1000, 9.0)}
    r_sea = ro.evaluate_policy("sea", 0.85, late, KITS, FR, CFG)        # plans on 40 days
    r_exp = ro.evaluate_policy("sea_expedite", 0.85, late, KITS, FR, CFG)
    assert r_sea["on_time"] == pytest.approx(0.9) and r_exp["on_time"] == 1.0   # the air kit rescues the 10%
    assert r_exp["expedite_rate"] == pytest.approx(0.1)


def test_cheapest_at_target_is_cheapest_feasible():
    rng = np.random.default_rng(8)
    T = {"sea": rng.gamma(20.0, 2.0, 50_000), "air": rng.gamma(10.0, 1.0, 50_000)}
    T["sea_gate"], T["air2"] = T["sea"] - 3.0, rng.gamma(10.0, 1.0, 50_000)
    grid = [0.5, 0.7, 0.9, 0.95, 0.99]
    for pol in ("sea", "sea_expedite"):
        b = ro.cheapest_at_target(pol, T, KITS, FR, CFG, 0.95, grid)
        feas = [ro.evaluate_policy(pol, p, T, KITS, FR, CFG) for p in grid]
        feas = [r for r in feas if r["on_time"] >= 0.95]
        assert b["cost"] == pytest.approx(min(r["cost"] for r in feas))


def test_consolidation_trade_off():
    m = next(x for x in MARKETS if x["iso3"] == "ITA")
    comps = ro.sea_components(m, CFG, LPI, ROUTES)
    T = np.sum(list(ro.sample_components(comps, np.random.default_rng(1), 50_000).values()), axis=0)
    res = [ro.consolidation(k, comps, np.random.default_rng(k), T, KITS["full"], CFG, "ITA", 6, 12, 300, 0.95)
           for k in (1, 2, 4)]
    assert [r["containers"] for r in res] == sorted([r["containers"] for r in res], reverse=True)
    assert res[0]["early_days"] < res[1]["early_days"] < res[2]["early_days"]   # later weeks wait longer
    assert all(r["on_time"] >= 0.94 for r in res)
    assert res[0]["freight_per_site"] > res[1]["freight_per_site"]
    with pytest.raises(ValueError):
        ro.consolidation(5, comps, np.random.default_rng(0), T, KITS["full"], CFG, "ITA", 6, 12, 10, 0.95)
