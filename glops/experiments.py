"""Experiment drivers. Each run_* writes its raw results under results/<part>/ and
returns a small dict of headline numbers that run_all.py collects into
results/summary.json (the single source for the README numbers)."""
from __future__ import annotations

import csv
import time

import numpy as np

from . import inventory as inv
from . import landed as lc
from . import public_data as pdata
from . import reverse as rv
from . import slotting as sl
from .common import CONFIG, RESULTS, mean_ci, read_json, write_json

SLOT_LARGE_SEEDS = list(range(20))
SLOT_SMALL_SEEDS = list(range(10))
SLOT_DEV_SEEDS = [1000, 1001, 1002]          # lambda tuning only; disjoint from evaluation
LAMBDA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0]
LS_ITERS = 60000


def _write_csv(path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


def _ci_round(c, nd=2):
    return {k: (round(v, nd) if isinstance(v, float) else v) for k, v in c.items()}


# =========================================================================== slotting

def run_slotting(log=print) -> dict:
    out_dir = RESULTS / "slotting"
    # ---- lambda tuning on dev seeds (never on evaluation seeds)
    tune = []
    for lam in LAMBDA_GRID:
        travel = []
        for seed in SLOT_DEV_SEEDS:
            inst = sl.generate(seed)
            slots = inst.layout.slots()
            a = sl.assign_heuristic(inst, slots, np.random.default_rng(seed), lam=lam, iters=LS_ITERS)
            travel.append(sl.evaluate(inst, slots, a))
        tune.append({"lambda": lam, "dev_mean_travel_m": float(np.mean(travel))})
    lam_star = min(tune, key=lambda r: r["dev_mean_travel_m"])["lambda"]
    log(f"  slotting: lambda* = {lam_star} (dev seeds)")

    # ---- large instance: 1,000 SKUs, 1,200 slots, 20 seeds
    rows = []
    methods = ["random", "abc", "turnover", "heuristic"]
    for seed in SLOT_LARGE_SEEDS:
        inst = sl.generate(seed)
        slots = inst.layout.slots()
        rng = np.random.default_rng(10_000 + seed)
        t0 = time.time()
        assigns = {"random": sl.assign_random(inst, slots, rng),
                   "abc": sl.assign_abc(inst, slots, rng),
                   "turnover": sl.assign_turnover(inst, slots, rng)}
        t1 = time.time()
        assigns["heuristic"] = sl.local_search(inst, slots, assigns["turnover"], lam_star, rng, LS_ITERS)
        t_heur = time.time() - t1
        for m in methods:
            a = assigns[m]
            assert sl.feasible(inst, slots, a), (seed, m)
            rows.append({"seed": seed, "method": m,
                         "travel_s_shape_m": sl.evaluate(inst, slots, a, policy="s_shape"),
                         "travel_return_m": sl.evaluate(inst, slots, a, policy="return"),
                         "proxy": sl.proxy_objective(inst, slots, a, lam_star),
                         "runtime_s": t_heur if m == "heuristic" else (t1 - t0) / 3})
    _write_csv(out_dir / "large_instance.csv", rows)
    large = _slot_summary(rows, methods, SLOT_LARGE_SEEDS)
    log(f"  slotting large: heuristic vs random {large['red_vs_random']['heuristic']['s_shape']['mean']:.1f}%")

    # ---- small instance: MILP (HiGHS) vs heuristic, 10 seeds
    lay = sl.Layout(aisles=3, bays=3, levels=2)
    par = sl.GeneratorParams(n_sku=30, n_kits=5, n_train_orders=1500, n_test_orders=2000, max_pairs=30)
    srows, milp_rows = [], []
    for seed in SLOT_SMALL_SEEDS:
        inst = sl.generate(seed, lay, par)
        slots = lay.slots()
        rng = np.random.default_rng(20_000 + seed)
        assigns = {"random": sl.assign_random(inst, slots, rng),
                   "abc": sl.assign_abc(inst, slots, rng),
                   "turnover": sl.assign_turnover(inst, slots, rng)}
        assigns["heuristic"] = sl.local_search(inst, slots, assigns["turnover"], lam_star, rng, 20000)
        t0 = time.time()
        assigns["milp"], info = sl.assign_milp(inst, slots, lam=lam_star, time_limit=120, threads=1)
        info.update({"seed": seed, "runtime_s": time.time() - t0,
                     "heuristic_proxy": sl.proxy_objective(inst, slots, assigns["heuristic"], lam_star),
                     "milp_proxy": sl.proxy_objective(inst, slots, assigns["milp"], lam_star)})
        info["heuristic_gap_pct"] = 100 * (info["heuristic_proxy"] - info["milp_proxy"]) / info["milp_proxy"]
        milp_rows.append(info)
        for m, a in assigns.items():
            assert sl.feasible(inst, slots, a), (seed, m)
            srows.append({"seed": seed, "method": m,
                          "travel_s_shape_m": sl.evaluate(inst, slots, a, policy="s_shape"),
                          "travel_return_m": sl.evaluate(inst, slots, a, policy="return"),
                          "proxy": sl.proxy_objective(inst, slots, a, lam_star), "runtime_s": 0.0})
    _write_csv(out_dir / "small_instance.csv", srows)
    _write_csv(out_dir / "small_instance_milp.csv", milp_rows)
    small = _slot_summary(srows, ["random", "abc", "turnover", "heuristic", "milp"], SLOT_SMALL_SEEDS)
    small["milp"] = {
        "n_optimal": sum(r["status"] == "Optimal" for r in milp_rows),
        "n": len(milp_rows),
        "max_mip_gap": max(r["gap"] for r in milp_rows),
        "heuristic_gap_pct": _ci_round(mean_ci([r["heuristic_gap_pct"] for r in milp_rows]), 3),
        "max_heuristic_gap_pct": max(r["heuristic_gap_pct"] for r in milp_rows),
        "mean_runtime_s": float(np.mean([r["runtime_s"] for r in milp_rows])),
        "n_binaries": milp_rows[0]["n_binaries"], "n_pairs": milp_rows[0]["n_pairs"],
    }
    res = {"lambda_tuning": tune, "lambda_star": lam_star, "large": large, "small": small,
           "large_instance": {"skus": sl.GeneratorParams().n_sku, "slots": sl.Layout().n_slots,
                              "seeds": len(SLOT_LARGE_SEEDS),
                              "test_orders_per_seed": sl.GeneratorParams().n_test_orders},
           "small_instance": {"skus": par.n_sku, "slots": lay.n_slots, "seeds": len(SLOT_SMALL_SEEDS)}}
    write_json(out_dir / "summary.json", res)
    return res


def _slot_summary(rows, methods, seeds) -> dict:
    tab = {(r["seed"], r["method"]): r for r in rows}
    out = {"travel": {}, "red_vs_random": {}, "red_vs_abc": {}}
    for m in methods:
        out["travel"][m] = {p: _ci_round(mean_ci([tab[s, m][f"travel_{p}_m"] for s in seeds]))
                            for p in ("s_shape", "return")}
        for base in ("random", "abc"):
            if m == base:
                continue
            out[f"red_vs_{base}"].setdefault(m, {})
            for p in ("s_shape", "return"):
                red = [100 * (tab[s, base][f"travel_{p}_m"] - tab[s, m][f"travel_{p}_m"]) /
                       tab[s, base][f"travel_{p}_m"] for s in seeds]
                out[f"red_vs_{base}"][m][p] = _ci_round(mean_ci(red))
    return out


# =========================================================================== inventory

def _scenario():
    sc = read_json(CONFIG / "inventory_scenario.json")
    item = inv.Item(**{k: v for k, v in sc["item"].items() if not k.startswith("_")})
    modes = {n: inv.Mode(n, **{k: v for k, v in m.items() if not k.startswith("_")})
             for n, m in sc["modes"].items()}
    return sc, item, modes


def run_inventory_validation(log=print) -> dict:
    sc, item, _ = _scenario()
    v = sc["validation"]
    L = v["lead_time_days"]
    mode = inv.Mode("deterministic", float(L), 0.0, 0.0, 0.0)
    zs = np.array(v["z"])
    S = item.demand_mean * (L + 1) + zs * item.demand_sd * np.sqrt(L + 1)
    sim = inv.simulate("base_stock", {"S": S}, item, mode, v["n_days"], v["warmup"], v["reps"],
                       v["seed"], allow_negative=True)
    rows, n_pass, n_tot = [], 0, 0
    for k, s in enumerate(S):
        a = inv.analytic_base_stock(float(s), item, L)
        for metric in ("fill_rate", "csl", "on_hand", "backorder"):
            c = mean_ci(sim[metric][k])
            se = c["sd"] / np.sqrt(c["n"])
            zdev = (c["mean"] - a[metric]) / se if se > 0 else 0.0
            ok = abs(zdev) < 3.0
            n_pass += ok
            n_tot += 1
            rows.append({"z": float(zs[k]), "S": float(s), "metric": metric, "analytic": a[metric],
                         "sim_mean": c["mean"], "sim_lo": c["lo"], "sim_hi": c["hi"],
                         "abs_diff": abs(c["mean"] - a[metric]), "z_dev": float(zdev),
                         "within_95ci": bool(c["lo"] <= a[metric] <= c["hi"]), "pass_3se": bool(ok)})
    _write_csv(RESULTS / "inventory" / "validation_analytic_vs_sim.csv", rows)
    fr = [r for r in rows if r["metric"] == "fill_rate"]
    res = {"n_checks": n_tot, "n_pass_3se": int(n_pass),
           "n_within_95ci": int(sum(r["within_95ci"] for r in rows)),
           "max_abs_fill_rate_diff_pp": 100 * max(r["abs_diff"] for r in fr),
           "max_balance_error": float(sim["final_balance_error"].max()),
           "max_pipeline_error": float(sim["pipeline_error"].max()),
           "lead_time": L, "reps": v["reps"], "days": v["n_days"]}

    # classic random-lead-time sigma formula vs simulation (approximation, no pass/fail)
    approx = []
    _, _, modes = _scenario()
    for name, m in modes.items():
        mu, sig = inv.protection_sigma(item, m)
        for z in (1.0, 1.645, 2.326):
            s = mu + z * sig
            r = inv.simulate("base_stock", {"S": np.array([s])}, item, m, 8000, 400, 10, 11)
            approx.append({"mode": name, "z": z, "target_csl": float(inv.stats.norm.cdf(z)),
                           "sim_csl": float(r["csl"].mean()), "sim_fill_rate": float(r["fill_rate"].mean())})
    _write_csv(RESULTS / "inventory" / "random_leadtime_formula_check.csv", approx)
    res["random_lt_formula"] = approx
    log(f"  inventory validation: {n_pass}/{n_tot} within 3 SE")
    return res


def run_inventory(log=print) -> dict:
    sc, item, modes = _scenario()
    sim_cfg = sc["simulation"]
    zgrid = np.round(np.linspace(-0.5, 3.0, 15), 3)
    rows = []
    best = {}
    for mname, mode in modes.items():
        mu, sig = inv.protection_sigma(item, mode)
        Q = inv.eoq(item, mode)
        level = mu + zgrid * sig
        cfgs = {"base_stock": {"S": level},
                "sS": {"s": level, "S": level + Q},
                "rQ": {"r": level, "Q": np.full_like(level, Q)}}
        for pol, params in cfgs.items():
            r = inv.simulate(pol, params, item, mode, sim_cfg["n_days"], sim_cfg["warmup"],
                             sim_cfg["reps"], sim_cfg["seed"])
            for k, z in enumerate(zgrid):
                fr = mean_ci(r["fill_rate"][k])
                tc = mean_ci(r["cost_total"][k])
                rows.append({"mode": mname, "policy": pol, "z": float(z), "Q": float(Q) if pol != "base_stock" else 0.0,
                             "fill_rate": fr["mean"], "fill_lo": fr["lo"], "fill_hi": fr["hi"],
                             "cost_total": tc["mean"], "cost_lo": tc["lo"], "cost_hi": tc["hi"],
                             **{f"cost_{c}": float(r[f"cost_{c}"][k].mean())
                                for c in ("holding", "pipeline", "backorder", "ordering", "freight")},
                             "on_hand": float(r["on_hand"][k].mean()),
                             "orders_per_day": float(r["orders_per_day"][k].mean()),
                             "max_balance_error": float(r["final_balance_error"][k].max())})
    _write_csv(RESULTS / "inventory" / "tradeoff.csv", rows)

    def cheapest(mname, pol, min_fill):
        c = [r for r in rows if r["mode"] == mname and r["policy"] == pol and r["fill_rate"] >= min_fill]
        return min(c, key=lambda r: r["cost_total"]) if c else None

    for mname in modes:
        for pol in ("base_stock", "sS", "rQ"):
            b = cheapest(mname, pol, 0.95)
            o = cheapest(mname, pol, 0.0)
            best[f"{mname}|{pol}"] = {
                "at_95": None if b is None else {k: b[k] for k in ("z", "fill_rate", "cost_total", "cost_lo", "cost_hi", "on_hand")},
                "unconstrained": {k: o[k] for k in ("z", "fill_rate", "cost_total", "on_hand")}}
    oc, ai = best["ocean|rQ"]["at_95"], best["air|rQ"]["at_95"]
    res = {"best": best, "zgrid": zgrid.tolist(),
           "ocean_vs_air_rq_95": {"ocean_cost_per_day": oc["cost_total"], "air_cost_per_day": ai["cost_total"],
                                  "air_minus_ocean_pct": 100 * (ai["cost_total"] - oc["cost_total"]) / oc["cost_total"],
                                  "ocean_on_hand": oc["on_hand"], "air_on_hand": ai["on_hand"]},
           "base_stock_vs_rQ_ocean_95_pct": 100 * (best["ocean|base_stock"]["at_95"]["cost_total"] - oc["cost_total"]) / oc["cost_total"]}

    # ---- break-even unit value: below it ocean is cheaper, above it air (r,Q @ >=95% fill)
    be_rows = []
    for value in [50, 100, 150, 200, 250, 300, 400, 600]:
        it = inv.Item(**{**item.__dict__, "unit_value": float(value)})
        for mname, mode in modes.items():
            mu, sig = inv.protection_sigma(it, mode)
            Q = inv.eoq(it, mode)
            zz = np.round(np.linspace(-0.5, 2.0, 21), 3)
            r = inv.simulate("rQ", {"r": mu + zz * sig, "Q": np.full_like(zz, Q)}, it, mode,
                             sim_cfg["n_days"], sim_cfg["warmup"], 5, sim_cfg["seed"])
            fr, tc = r["fill_rate"].mean(axis=1), r["cost_total"].mean(axis=1)
            ok = fr >= 0.95
            be_rows.append({"unit_value": value, "mode": mname, "cost_at_95": float(tc[ok].min()),
                            "z": float(zz[ok][np.argmin(tc[ok])])})
    _write_csv(RESULTS / "inventory" / "breakeven_unit_value.csv", be_rows)
    diff = [(r["unit_value"], r["cost_at_95"] - next(q["cost_at_95"] for q in be_rows
             if q["unit_value"] == r["unit_value"] and q["mode"] == "air"))
            for r in be_rows if r["mode"] == "ocean"]
    breakeven = None
    for (v0, d0), (v1, d1) in zip(diff, diff[1:]):
        if d0 <= 0 < d1:   # ocean cheaper at v0, air cheaper at v1
            breakeven = v0 + (v1 - v0) * (-d0) / (d1 - d0)
    res["breakeven_unit_value"] = breakeven
    res["breakeven_table"] = be_rows

    # ---- seasonality stress test with the public-data proxy profile
    idx = pdata.seasonal_index()
    T = sim_cfg["n_days"] + sim_cfg["warmup"]
    day_month = ((np.arange(T) / (365.25 / 12)).astype(int)) % 12
    season = idx[day_month]
    mode = modes["ocean"]
    mu, sig = inv.protection_sigma(item, mode)
    z95 = oc["z"]
    Q = inv.eoq(item, mode)
    pr = {"r": np.array([mu + z95 * sig]), "Q": np.array([Q])}
    flat = inv.simulate("rQ", pr, item, mode, sim_cfg["n_days"], sim_cfg["warmup"],
                        sim_cfg["reps"], sim_cfg["seed"], bucket=day_month)
    sea = inv.simulate("rQ", pr, item, mode, sim_cfg["n_days"], sim_cfg["warmup"],
                       sim_cfg["reps"], sim_cfg["seed"], season=season, bucket=day_month)
    seas = sea["bucket_fill_rate"][0].tolist()
    res["seasonality"] = {"index": idx.tolist(), "flat_fill_rate": float(flat["fill_rate"].mean()),
                          "flat_monthly_fill": flat["bucket_fill_rate"][0].tolist(),
                          "seasonal_monthly_fill": seas, "worst_month": int(np.argmin(seas)) + 1,
                          "worst_month_fill": float(min(seas)),
                          "overall_seasonal_fill": float(sea["fill_rate"].mean())}
    log(f"  inventory: ocean rQ@95% {oc['cost_total']:.1f}/day, air {ai['cost_total']:.1f}/day")
    write_json(RESULTS / "inventory" / "summary.json", res)
    return res


# =========================================================================== landed cost

def run_landed(log=print) -> dict:
    markup = 0.05
    opts = lc.Options(seller_markup=markup)
    rows, by = [], {}
    for sh in lc.example_shipments():
        for rule in lc.rules_for(sh):
            r = lc.landed_cost(sh, rule, opts)
            by[sh.name, rule] = r
            rows.append({"shipment": sh.name, "lane": sh.lane, "mode": sh.mode, "regime": sh.regime,
                         "rule": rule, "insurance_clause": r["insurance_clause"],
                         "price": r["price"], "buyer_paid_separately": r["buyer_paid_separately"],
                         "customs_value": r["customs_value"], "duty": r["duty"], "tax": r["tax"],
                         "landed_gross": r["landed_gross"], "landed_net": r["landed_net"],
                         "landed_net_per_unit": r["landed_net_per_unit"],
                         "seller_margin_on_logistics": r["seller_margin_on_logistics"],
                         "warnings": " | ".join(r["warnings"])})
    _write_csv(RESULTS / "landed" / "landed_cost_matrix.csv", rows)
    # who-pays matrix as resolved by the model (for the report)
    rules = lc.load_rules()
    payer_rows = []
    for rule in lc.ALL_RULES:
        p = lc.resolve_payers(rule, lc.Options(), rules)
        payer_rows.append({"rule": rule, **p})
    _write_csv(RESULTS / "landed" / "payer_matrix.csv", payer_rows)

    # conservation check: zero markup + equal insurance rates => same net landed cost (non-DDP)
    cons = []
    for sh in lc.example_shipments():
        vals = [lc.landed_cost(sh, rule, lc.Options(seller_markup=0.0, insurance_rates={"A": 0.0015, "C": 0.0015}))["landed_net"]
                for rule in lc.rules_for(sh) if rule != "DDP"]
        cons.append(max(vals) - min(vals))
    fcl = lambda rule: by["FCL radio units", rule]
    lcl_unit = by["LCL optics + PSUs", "FCA"]["landed_net_per_unit"]
    air_unit = by["Air optics + PSUs", "FCA"]["landed_net_per_unit"]
    non_ddp = [fcl(r)["landed_net"] for r in lc.rules_for(lc.example_shipments()[0]) if r != "DDP"]
    res = {
        "markup": markup,
        "conservation_max_spread_usd": float(max(cons)),
        "fcl_sg": {"exw_net": fcl("EXW")["landed_net"], "fca_net": fcl("FCA")["landed_net"],
                   "ddp_net": fcl("DDP")["landed_net"],
                   "ddp_premium_pct": 100 * (fcl("DDP")["landed_net"] - fcl("FCA")["landed_net"]) / fcl("FCA")["landed_net"],
                   "non_ddp_spread_pct": 100 * (max(non_ddp) - min(non_ddp)) / min(non_ddp),
                   "gst": fcl("FCA")["tax"]},
        "air_vs_lcl_per_unit": {"lcl": lcl_unit, "air": air_unit, "air_premium_pct": 100 * (air_unit - lcl_unit) / lcl_unit},
        "us_fob_basis_vs_cif_basis_duty": None,
    }
    # valuation-basis effect: the same US shipment valued CIF-style instead of FOB-style
    sh = lc.example_shipments()[4]
    regs = lc.load_regimes()
    alt = dict(regs)
    alt["US_CIF_COUNTERFACTUAL"] = dict(regs["US_STYLE_EXAMPLE"], valuation_basis="CIF")
    sh_alt = lc.Shipment(**{**sh.__dict__, "regime": "US_CIF_COUNTERFACTUAL"})
    d_fob = lc.landed_cost(sh, "FCA", opts)["duty"]
    d_cif = lc.landed_cost(sh_alt, "FCA", opts, regimes=alt)["duty"]
    res["us_fob_basis_vs_cif_basis_duty"] = {"fob_basis": d_fob, "cif_basis": d_cif}
    write_json(RESULTS / "landed" / "summary.json", res)
    log(f"  landed: DDP premium {res['fcl_sg']['ddp_premium_pct']:.2f}% (unrecoverable GST)")
    return res


# =========================================================================== reverse logistics

def run_reverse(log=print) -> dict:
    net = rv.load_network()
    base = rv.build_and_solve(net)
    assert base["status"] == "Optimal"
    assert not rv.check_flows(net, base["flows"])
    greedy = rv.greedy_baseline(net)
    g_cost = rv.evaluate_flows(net, greedy)
    out_dir = RESULTS / "reverse"
    flow_rows = [{"type": part, "key": " -> ".join(map(str, k)), "units": float(q)}
                 for part in ("x", "y", "f", "s", "p", "o") for k, q in base["flows"][part].items() if q > 1e-6]
    _write_csv(out_dir / "base_flows.csv", flow_rows)

    sweeps = []
    for p in [10, 20, 30, 40, 50, 60, 80, 100, 125, 150, 200, 250]:
        r = rv.build_and_solve(rv.with_changes(net, resale_price=p))
        sweeps.append({"sweep": "resale_price", "value": p, "net_value": -r["net_cost"], **_sw(r)})
    cap0 = net["refurb_centres"]["Penang refurb"]["capacity"]
    for c in [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]:
        r = rv.build_and_solve(rv.with_changes(net, refurb_capacity={"Penang refurb": c}))
        sweeps.append({"sweep": "penang_capacity", "value": c, "net_value": -r["net_cost"], **_sw(r),
                       "dual_penang_cap": -r["duals"]["refurb_cap|Penang refurb"]})
    for m in [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]:
        r = rv.build_and_solve(rv.with_changes(net, volume_mult=m))
        sweeps.append({"sweep": "volume_mult", "value": m, "net_value": -r["net_cost"], **_sw(r)})
    for m in [0.5, 1.0, 1.5, 2.0, 3.0]:
        r = rv.build_and_solve(rv.with_changes(net, transport_mult=m))
        sweeps.append({"sweep": "transport_mult", "value": m, "net_value": -r["net_cost"], **_sw(r)})
    nocross = rv.build_and_solve(rv.with_changes(net, allow_cross_border_recycling=False))
    sweeps.append({"sweep": "no_cross_border_recycling", "value": 1, "net_value": -nocross["net_cost"], **_sw(nocross)})
    for row in sweeps:
        row.setdefault("dual_penang_cap", "")
    _write_csv(out_dir / "sensitivity.csv", sweeps)

    # finite-difference check of the capacity shadow price
    fd = -(rv.build_and_solve(rv.with_changes(net, refurb_capacity={"Penang refurb": cap0 + 1}))["net_cost"]
           - base["net_cost"])
    s = base["summary"]
    res = {"net_value": -base["net_cost"], "greedy_net_value": -g_cost,
           "lp_vs_greedy_gain_pct": 100 * ((-base["net_cost"]) - (-g_cost)) / abs(g_cost),
           "summary": s,
           "shadow_price_penang_capacity": -base["duals"]["refurb_cap|Penang refurb"],
           "shadow_price_market_cap": -base["duals"]["market"],
           "finite_difference_penang": fd,
           "no_cross_border": {"net_value": -nocross["net_cost"],
                               "loss_pct": 100 * (base["net_cost"] - nocross["net_cost"]) / base["net_cost"]},
           "refurb_threshold_price": min(r["value"] for r in sweeps
                                          if r["sweep"] == "resale_price" and r["refurb_share_A"] >= 0.999)}
    write_json(out_dir / "summary.json", res)
    log(f"  reverse: LP net value {res['net_value']:.0f} vs greedy {res['greedy_net_value']:.0f}")
    return res


def _sw(r):
    s = r["summary"]
    return {"refurb_share_A": float(s["refurb_share_A"]), "refurb_share_B": float(s["refurb_share_B"]),
            "resold": float(s["resold"]), "recycled": float(s["recycled"]), "overflow": float(s["overflow"]),
            "cross_border_waste": float(s["cross_border_waste_units"])}


# =========================================================================== public data

def run_public(log=print) -> dict:
    idx = pdata.seasonal_index()
    tot = pdata.annual_totals()
    meta = pdata.meta()
    res = {"seasonal_index": idx.tolist(), "annual_totals_thousand_teu": {str(k): v for k, v in tot.items()},
           "meta": meta, "feb_index": float(idx[1])}
    write_json(RESULTS / "public" / "summary.json", res)
    return res
