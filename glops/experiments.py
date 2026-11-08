"""Experiment drivers. Each run_* writes its raw results under results/<part>/ and
returns a small dict of headline numbers that run_all.py collects into
results/summary.json (the single source for the README numbers)."""
from __future__ import annotations

import csv
import math
import time

import numpy as np
from scipy import stats

from . import inventory as inv
from . import landed as lc
from . import online_retail as orr
from . import public_data as pdata
from . import real_orders as rlo
from . import reverse as rv
from . import rollout as ro
from . import slotting as sl
from .common import CONFIG, RESULTS, mean_ci, read_json, write_json

TIMINGS = RESULTS / "timings"   # wall-clock runtimes, git-ignored (they differ on every run)

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
    rows, timing_rows = [], []
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
                         "proxy": sl.proxy_objective(inst, slots, a, lam_star)})
            timing_rows.append({"seed": seed, "method": m,
                                "runtime_s": t_heur if m == "heuristic" else (t1 - t0) / 3})
    _write_csv(out_dir / "large_instance.csv", rows)
    # wall-clock runtimes change on every run, so they go to the git-ignored results/timings/
    # and every committed result file is reproducible (CI: tools/check_results.py)
    _write_csv(TIMINGS / "slotting_large_instance.csv", timing_rows)
    large = _slot_summary(rows, methods, SLOT_LARGE_SEEDS)
    log(f"  slotting large: heuristic vs random {large['red_vs_random']['heuristic']['s_shape']['mean']:.1f}%")

    # ---- small instance: MILP (HiGHS) vs heuristic, 10 seeds
    lay = sl.Layout(aisles=3, bays=3, levels=2)
    par = sl.GeneratorParams(n_sku=30, n_kits=5, n_train_orders=1500, n_test_orders=2000, max_pairs=30)
    srows, milp_rows, milp_timing = [], [], []
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
        milp_timing.append({"seed": seed, "runtime_s": time.time() - t0})
        info.update({"seed": seed,
                     "heuristic_proxy": sl.proxy_objective(inst, slots, assigns["heuristic"], lam_star),
                     "milp_proxy": sl.proxy_objective(inst, slots, assigns["milp"], lam_star)})
        info["heuristic_gap_pct"] = 100 * (info["heuristic_proxy"] - info["milp_proxy"]) / info["milp_proxy"]
        milp_rows.append(info)
        for m, a in assigns.items():
            assert sl.feasible(inst, slots, a), (seed, m)
            srows.append({"seed": seed, "method": m,
                          "travel_s_shape_m": sl.evaluate(inst, slots, a, policy="s_shape"),
                          "travel_return_m": sl.evaluate(inst, slots, a, policy="return"),
                          "proxy": sl.proxy_objective(inst, slots, a, lam_star)})
    _write_csv(out_dir / "small_instance.csv", srows)
    _write_csv(out_dir / "small_instance_milp.csv", milp_rows)
    _write_csv(TIMINGS / "slotting_small_instance_milp.csv", milp_timing)
    small = _slot_summary(srows, ["random", "abc", "turnover", "heuristic", "milp"], SLOT_SMALL_SEEDS)
    small["milp"] = {
        "n_optimal": sum(r["status"] == "Optimal" for r in milp_rows),
        "n": len(milp_rows),
        "max_mip_gap": max(r["gap"] for r in milp_rows),
        "heuristic_gap_pct": _ci_round(mean_ci([r["heuristic_gap_pct"] for r in milp_rows]), 3),
        "max_heuristic_gap_pct": max(r["heuristic_gap_pct"] for r in milp_rows),
        # wall-clock runtimes stay in results/timings/ (git-ignored), not in any committed result
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


# =========================================================================== overseas rollout (Part 5)

SENS_VARIANTS = [
    ("base", {}),
    ("dwell fit: median + mean", {"fit": "median_mean"}),
    ("destination dwell x 0.75", {"dwell_scale": 0.75}),
    ("destination dwell x 1.25", {"dwell_scale": 1.25}),
    ("LPI 2.0 year 2023", {"year": 2023}),
    ("sea speed 14 kn", {"speed_kn": 14.0}),
    ("sea speed 18 kn", {"speed_kn": 18.0}),
    ("no transshipment", {"transship": False}),
    ("Cape route (avoid Red Sea)", {"route": "avoid_red_sea"}),
]
POLICIES = ["sea", "split", "sea_expedite", "air"]


def _rng(cfg: dict, *key: int) -> np.random.Generator:
    return np.random.default_rng([cfg["mc"]["seed"], *key])


def _floor_stacks(units: list, cts: dict, tiers: int) -> dict:
    """The argument behind the floor lower bound, per footprint: units that may not ride on top
    need their own floor position, each carrier takes at most one unit on top, the remaining
    stackable units share positions ``tiers`` high. Also: by how much a 40-foot box is too short
    for one more row of pallets placed lengthwise."""
    groups = {}
    for u in units:
        g = groups.setdefault(round(u.footprint, 6), {"own": 0, "carriers": 0, "stackable": 0})
        g["own" if not u.top_ok else "stackable"] += 1
        g["carriers"] += (not u.top_ok and u.max_top_kg > 0)
    stacks = {fp: g["own"] + math.ceil(max(0, g["stackable"] - g["carriers"] * (tiers - 1)) / tiers)
              for fp, g in groups.items()}
    pallet = max(stacks, key=lambda fp: stacks[fp])
    floor_m2 = sum(fp * n for fp, n in stacks.items())
    hc = cts["40HC"]
    pl = max(max(u.l, u.w) for u in units if round(u.footprint, 6) == pallet)
    rows = math.floor(hc.length / pl + 1e-9) + 1          # the first row count that does not fit
    return {"pallet_stacks": stacks[pallet], "other_stacks": sum(n for fp, n in stacks.items() if fp != pallet),
            "floor_m2": floor_m2, "floor_40ft_m2": hc.floor, "pallet_length_m": pl,
            "pallet_rows_not_fitting": rows, "rows_short_mm": 1000 * (rows * pl - hc.length)}


def run_rollout(log=print) -> dict:
    cfg, lpi, routes, ap, shares = (ro.load_config(), ro.load_lpi(), ro.load_sea_routes(),
                                    ro.load_airports(), ro.load_shares())
    mc, out_dir = cfg["mc"], RESULTS / "rollout"
    target, markets = mc["on_time_target"], cfg["markets"]
    sites, weeks = cfg["schedule"]["sites_per_week"], cfg["schedule"]["weeks"]
    year = cfg["selection"]["lpi_year"]

    # ---- 0. market selection (UN Comtrade shares + LPI 2.0 data availability) and data checks
    sel = ro.select_markets(shares, lpi, cfg["selection"]["regions"], year)
    if sel != [m["iso3"] for m in markets]:
        raise ValueError(f"config markets {[m['iso3'] for m in markets]} != selection rule {sel}")
    sel25 = ro.select_markets(shares, lpi, cfg["selection"]["regions"], year, "share_2025_pct")
    share = {r["iso3"]: r for r in shares}
    fit_rows = []
    for iso, kind in [(m["iso3"], "import") for m in markets] + [(cfg["origin"]["lpi_iso3"], "export")]:
        row = lpi[iso]
        d = ro.container_dwell(row, kind, year, "quartiles")
        pub = row[f"ct_{kind}_dwell_mean_{year}"]
        fit_rows.append({"iso3": iso, "dwell": kind, "q1": row[f"ct_{kind}_dwell_q1_{year}"],
                         "median": row[f"ct_{kind}_dwell_median_{year}"], "q3": row[f"ct_{kind}_dwell_q3_{year}"],
                         "published_mean": pub, "fitted_mean": d.mean(), "mu": d.a, "sigma": d.b,
                         "fitted_p95": math.exp(d.a + ro.Z95 * d.b),
                         "mean_rel_err_pct": 100 * (d.mean() - pub) / pub})
    _write_csv(out_dir / "dwell_fit_check.csv", fit_rows)
    ct = [r for r in lpi.values() if r.get(f"ct_import_dwell_mean_{year}") is not None
          and r.get(f"ct_import_dwell_median_{year}") is not None]
    av = [r for r in lpi.values() if r.get(f"air_import_dwell_mean_{year}") is not None
          and r.get(f"air_import_dwell_p80_minus_p20_{year}") is not None]
    data_checks = {"ct_mean_ge_median": sum(r[f"ct_import_dwell_mean_{year}"] >= r[f"ct_import_dwell_median_{year}"]
                                            for r in ct), "ct_rows": len(ct),
                   "air_mean_below_fifth_of_range": sum(r[f"air_import_dwell_mean_{year}"]
                                                        < 0.2 * r[f"air_import_dwell_p80_minus_p20_{year}"] for r in av),
                   "air_rows": len(av)}
    key = f"ct_import_dwell_median_{year}"
    both = sorted((k, r["lpi2023_customs"], r[key]) for k, r in lpi.items()
                  if r.get("lpi2023_customs") is not None and r.get(key) is not None)
    rho = stats.spearmanr([b[1] for b in both], [b[2] for b in both])
    _write_csv(out_dir / "customs_score_vs_dwell.csv",
               [{"iso3": k, "lpi2023_customs": c, "ct_import_dwell_median": d, "selected": k in sel}
                for k, c, d in both])

    # ---- a + b. lead time per market and mode; ship-date planning checked out of sample
    lead_rows, comp_rows, plan_rows, comps_of, T_of = [], [], [], {}, {}
    for i, m in enumerate(markets):
        for j, mode in enumerate(("sea", "air")):
            comps = (ro.sea_components(m, cfg, lpi, routes) if mode == "sea"
                     else ro.air_components(m, cfg, lpi, ap))
            S = ro.sample_components(comps, _rng(cfg, 1, i, j), mc["n_lead"])
            T = np.sum(list(S.values()), axis=0)
            comps_of[m["iso3"], mode], T_of[m["iso3"], mode] = comps, T
            s = ro.lead_time_summary(S, mode, target)
            am, av = ro.analytic_moments(comps)
            m4 = float(np.mean((T - T.mean()) ** 4))
            lead_rows.append({"market": m["iso3"], "mode": mode, "mean": s["mean"], "analytic_mean": am,
                              "mean_z": (s["mean"] - am) / math.sqrt(av / T.size),
                              "sd": s["sd"], "analytic_sd": math.sqrt(av),
                              "var_z": (s["sd"] ** 2 - av) / math.sqrt((m4 - s["sd"] ** 4) / T.size),
                              "p50": s["p50"], "p95": s["p95"], "buffer_95": s["buffer_95"],
                              "on_time_if_plan_on_mean": s["on_time_if_plan_on_mean"],
                              "share_port_customs_mean": s["share_port_customs_mean"],
                              "share_dest_port_customs_mean": s["share_dest_port_customs_mean"],
                              "share_port_customs_tail": s["share_port_customs_tail"]})
            comp_rows += [{"market": m["iso3"], "mode": mode, "component": k, "mean_days": v,
                           "analytic_mean_days": comps[k].mean()} for k, v in s["component_means"].items()]
            oos = {tag: ro.simulate_rollout(comps, _rng(cfg, 2, i, j, t), weeks, sites, mc["reps_rollout"], lead)
                   for t, (tag, lead) in enumerate((("p95", s["p95"]), ("mean", s["mean"])))}
            c95 = mean_ci(oos["p95"]["per_rep"])
            plan_rows.append({"market": m["iso3"], "mode": mode, "ship_days_before_need_p95": s["p95"],
                              "ship_days_before_need_mean": s["mean"], "buffer_days": s["buffer_95"],
                              "on_time_oos_plan_p95": oos["p95"]["on_time"], "on_time_oos_lo": c95["lo"],
                              "on_time_oos_hi": c95["hi"],
                              "week_all_sites_on_time_p95": oos["p95"]["week_all_on_time"],
                              "on_time_oos_plan_mean": oos["mean"]["on_time"],
                              "worst_week_on_time_plan_mean": oos["mean"]["worst_week_mean"]})
    _write_csv(out_dir / "lead_time.csv", lead_rows)
    _write_csv(out_dir / "lead_time_components.csv", comp_rows)
    _write_csv(out_dir / "ship_date_plan.csv", plan_rows)
    log("  rollout: lead times and ship-date plans done")

    # ---- sensitivity of the sea P95 to the LPI mapping and other assumptions
    sens_rows = []
    for i, m in enumerate(markets):
        for v, (name, kw) in enumerate(SENS_VARIANTS):
            comps = ro.sea_components(m, cfg, lpi, routes, **kw)
            T = np.sum(list(ro.sample_components(comps, _rng(cfg, 3, i, v), mc["n_sens"]).values()), axis=0)
            q = ro.quantile(T, target)
            sens_rows.append({"market": m["iso3"], "variant": name, "mean": float(T.mean()), "p95": q,
                              "buffer_95": q - float(T.mean())})
    base_p95 = {r["market"]: r["p95"] for r in sens_rows if r["variant"] == "base"}
    for r in sens_rows:
        r["p95_change_days"] = r["p95"] - base_p95[r["market"]]
    _write_csv(out_dir / "sensitivity_sea_p95.csv", sens_rows)

    # ---- d. container loading: one weekly batch, plus random instances against the lower bound
    cts = ro.container_types(cfg)
    clr, tiers = cfg["containers"]["clearance_m"], cfg["containers"]["max_tiers"]
    week_units = ro.kit_units(cfg, sites)
    load_rows = []
    for name, ct in cts.items():
        boxes = ro.pack(week_units, ct, clr, tiers)
        if ro.check_plan(boxes, week_units, clr, tiers):
            raise AssertionError(f"weekly batch {name}: infeasible load plan")
        lb = ro.lower_bound(week_units, ct, tiers)
        load_rows.append({"container": name, "heuristic": len(boxes), "lb_volume": lb["volume"],
                          "lb_weight": lb["weight"], "lb_floor": lb["floor"], "lower_bound": lb["best"],
                          "fill_volume": sum(u.volume for u in week_units) / (len(boxes) * ct.volume),
                          "fill_weight": sum(u.kg for u in week_units) / (len(boxes) * ct.payload)})
    _write_csv(out_dir / "loading_weekly_batch.csv", load_rows)
    weekly_floor = _floor_stacks(week_units, cts, tiers)
    rng = _rng(cfg, 4)
    kinds = [u["id"] for u in cfg["site_kit"]["units"]]
    inst_rows = []
    for k in range(mc["loading_random_instances"]):
        n_s = int(rng.integers(1, 31))
        keep = {kd for kd in kinds if rng.random() < 0.8} or {kinds[0]}
        us = ro.kit_units(cfg, n_s, keep)
        for name in ("20GP", "40HC"):
            boxes = ro.pack(us, cts[name], clr, tiers)
            errs = ro.check_plan(boxes, us, clr, tiers)
            if errs:
                raise AssertionError(f"instance {k} {name}: {errs[:3]}")
            lb = ro.lower_bound(us, cts[name], tiers)["best"]
            inst_rows.append({"instance": k, "container": name, "sites": n_s, "units": len(us),
                              "heuristic": len(boxes), "lower_bound": lb, "ratio": len(boxes) / lb})
    _write_csv(out_dir / "loading_random_instances.csv", inst_rows)
    hc = [r for r in inst_rows if r["container"] == "40HC"]

    # ---- c. mode policy: cost per site vs on-time rate
    crit = {u["id"] for u in cfg["site_kit"]["units"] if u["critical"]}
    bulky = set(kinds) - crit
    kits = {"full": ro.Kit.of(ro.kit_units(cfg, 1)), "bulky": ro.Kit.of(ro.kit_units(cfg, 1, bulky)),
            "critical": ro.Kit.of(ro.kit_units(cfg, 1, crit))}
    fine = [round(float(x), 2) for x in np.arange(0.50, 0.995, 0.01)]
    pol_rows, best_rows, batches = [], [], {}
    for i, m in enumerate(markets):
        iso = m["iso3"]
        rs, ra = _rng(cfg, 5, i, 0), _rng(cfg, 5, i, 1)
        Ss = ro.sample_components(comps_of[iso, "sea"], rs, mc["n_lead"])
        Ts = np.sum(list(Ss.values()), axis=0)
        Ta = np.sum(list(ro.sample_components(comps_of[iso, "air"], ra, mc["n_lead"]).values()), axis=0)
        Ta2 = np.sum(list(ro.sample_components(comps_of[iso, "air"], ra, mc["n_lead"]).values()), axis=0)
        T = {"sea": Ts, "sea_gate": Ts - Ss["inland"], "air": Ta, "air2": Ta2}
        # fresh draws (own seed stream) to re-score each chosen plan out of sample
        fs, fa = _rng(cfg, 7, i, 0), _rng(cfg, 7, i, 1)
        Fs = ro.sample_components(comps_of[iso, "sea"], fs, mc["n_lead"])
        Fts = np.sum(list(Fs.values()), axis=0)
        T_fresh = {"sea": Fts, "sea_gate": Fts - Fs["inland"],
                   "air": np.sum(list(ro.sample_components(comps_of[iso, "air"], fa, mc["n_lead"]).values()), axis=0),
                   "air2": np.sum(list(ro.sample_components(comps_of[iso, "air"], fa, mc["n_lead"]).values()), axis=0)}
        sf, sb = ro.sea_freight_per_site(cfg, iso, sites), ro.sea_freight_per_site(cfg, iso, sites, bulky)
        batches[iso] = sf["batch"]
        fr = {"market": iso, "sea_full": sf["per_site"], "sea_bulky": sb["per_site"], "sites": sites}
        for pol in POLICIES:
            for p in mc["plan_quantiles"]:
                r = ro.evaluate_policy(pol, p, T, kits, fr, cfg)
                pol_rows.append({"market": iso, "policy": pol, "plan_quantile": p, "on_time": r["on_time"],
                                 "cost_per_site": r["cost"], "lead_days": r["lead_days"],
                                 "expedite_rate": r.get("expedite_rate", 0.0)})
            b = ro.cheapest_at_target(pol, T, kits, fr, cfg, target, fine)
            f = ro.evaluate_policy(pol, b["plan_quantile"], T_fresh, kits, fr, cfg, plan=b["plan"])
            best_rows.append({"market": iso, "policy": pol, "plan_quantile": b["plan_quantile"],
                              "on_time": b["on_time"], "cost_per_site": b["cost"], "lead_days": b["lead_days"],
                              "expedite_rate": b.get("expedite_rate", 0.0),
                              "target_missed": b.get("target_missed", False),
                              "on_time_fresh": f["on_time"], "cost_per_site_fresh": f["cost"]})
    _write_csv(out_dir / "policy_frontier.csv", pol_rows)
    _write_csv(out_dir / "policy_at_target.csv", best_rows)
    log("  rollout: mode policies done")

    # ---- e. consolidation frequency (sea)
    cons_rows = []
    for i, m in enumerate(markets):
        for k in mc["consolidation_weeks"]:
            r = ro.consolidation(k, comps_of[m["iso3"], "sea"], _rng(cfg, 6, i, k), T_of[m["iso3"], "sea"],
                                 kits["full"], cfg, m["iso3"], sites, weeks, mc["reps_rollout"], target)
            cons_rows.append({"market": m["iso3"], **r})
    _write_csv(out_dir / "consolidation.csv", cons_rows)

    # ---- summary (source of the README numbers)
    def cheapest(iso):
        ok = [r for r in best_rows if r["market"] == iso and not r["target_missed"]]
        return min(ok, key=lambda r: r["cost_per_site"])["policy"]

    def rng_of(rows, k):
        return {"min": min(r[k] for r in rows), "max": max(r[k] for r in rows)}

    sea_rows = [r for r in lead_rows if r["mode"] == "sea"]
    res = {
        "markets": [m["iso3"] for m in markets], "market_names": {m["iso3"]: m["name"] for m in markets},
        "selection": {"rule_2024_2025": sel, "rule_2025_only": sel25,
                      "cover_share_pct": sum(share[k]["share_2024_2025_pct"] for k in sel),
                      "shares": {k: share[k]["share_2024_2025_pct"] for k in sel}},
        "dwell_fit": {"max_abs_mean_rel_err_pct": max(abs(r["mean_rel_err_pct"]) for r in fit_rows),
                      "rows": fit_rows},
        "data_checks": data_checks,
        "customs_score_vs_dwell": {"spearman_rho": float(rho.statistic), "p_value": float(rho.pvalue),
                                   "n": len(both)},
        "lead": {f"{r['market']}|{r['mode']}": {k: v for k, v in r.items() if k not in ("market", "mode")}
                 for r in lead_rows},
        "lead_check": {"max_abs_mean_z": max(abs(r["mean_z"]) for r in lead_rows),
                       "max_abs_var_z": max(abs(r["var_z"]) for r in lead_rows)},
        "plan": {f"{r['market']}|{r['mode']}": {k: v for k, v in r.items() if k not in ("market", "mode")}
                 for r in plan_rows},
        "plan_range": {mode: {k: rng_of([r for r in plan_rows if r["mode"] == mode], k)
                              for k in ("on_time_oos_plan_mean", "on_time_oos_plan_p95", "buffer_days")}
                       for mode in ("sea", "air")},
        "sea_port_share": {k: rng_of(sea_rows, k) for k in ("share_port_customs_mean", "share_port_customs_tail",
                                                           "share_dest_port_customs_mean")},
        "sensitivity": {f"{r['market']}|{r['variant']}": r["p95_change_days"] for r in sens_rows},
        "loading": {"weekly_batch": load_rows, "weekly_floor": weekly_floor, "weekly_mix": batches[markets[0]["iso3"]]["mix"],
                    "weekly_fill_volume": batches[markets[0]["iso3"]]["fill_volume"],
                    "random_40hc": {"n": len(hc), "equal_lb": sum(r["heuristic"] == r["lower_bound"] for r in hc),
                                    "max_gap_boxes": max(r["heuristic"] - r["lower_bound"] for r in hc),
                                    "mean_ratio": float(np.mean([r["ratio"] for r in hc]))}},
        "policy": {f"{r['market']}|{r['policy']}": {k: v for k, v in r.items() if k not in ("market", "policy")}
                   for r in best_rows},
        "policy_best": {m["iso3"]: cheapest(m["iso3"]) for m in markets},
        "policy_fresh": {
            "n_draws": mc["n_lead"],
            "on_time": {pol: rng_of([r for r in best_rows if r["policy"] == pol], "on_time_fresh")
                        for pol in POLICIES},
            "max_abs_cost_change_pct": max(abs(100 * (r["cost_per_site_fresh"] - r["cost_per_site"])
                                               / r["cost_per_site"]) for r in best_rows),
            "policy_best_fresh": {m["iso3"]: min((r for r in best_rows if r["market"] == m["iso3"]),
                                                 key=lambda r: r["cost_per_site_fresh"])["policy"]
                                  for m in markets}},
        "kit": {"critical_value_share": kits["critical"].value / kits["full"].value,
                "critical_weight_share": kits["critical"].kg / kits["full"].kg,
                "buffer_usd_per_site_day": float(ro.holding(kits["full"], 1.0, cfg, True)),
                "holding_rate_per_year": cfg["costs"]["holding_rate_per_year"],
                "expedite_premium": cfg["costs"]["expedite_premium"]},
        "consolidation": {f"{r['market']}|{r['k_weeks']}": {k: v for k, v in r.items()
                                                            if k not in ("market", "k_weeks")}
                          for r in cons_rows},
        "config": {"sites_per_week": sites, "weeks": weeks, "n_lead": mc["n_lead"],
                   "reps_rollout": mc["reps_rollout"], "speed_kn": cfg["sea"]["speed_kn"], "target": target},
    }
    write_json(out_dir / "summary.json", res)
    log(f"  rollout: cheapest policy at >= {target:.0%} on time: {res['policy_best']}")
    return res


# =========================================================================== real orders (Part 1b)

def run_real_orders(log=print) -> dict:
    """Parts 1 and 2 re-checked on real orders (UCI Online Retail II, committed derived tables)."""
    out_dir = RESULTS / "real_orders"
    meta = orr.load_meta()
    real, real_rows = rlo.slotting_real(log)
    syn, syn_rows = rlo.slotting_synthetic_relaxed(log)
    _write_csv(out_dir / "slotting_real_by_seed.csv", real_rows)
    _write_csv(out_dir / "slotting_synthetic_relaxed_by_seed.csv", syn_rows)
    sc, item, modes = _scenario()
    inv_res = rlo.inventory_real(item, modes, sc["simulation"], log)
    _write_csv(out_dir / "inventory_replay.csv", inv_res["skus"])
    res = {"data": {k: meta[k] for k in ("dataset", "uci_id", "page", "accessed", "licence", "citation",
                                         "periods", "cleaning", "orders", "coverage", "calendar")},
           "lambda": rlo.LAMBDA, "method_seeds": len(rlo.SEEDS), "n_boot": rlo.N_BOOT,
           "slotting": real, "synthetic_relaxed": syn, "inventory": inv_res}
    write_json(out_dir / "summary.json", res)
    return res

