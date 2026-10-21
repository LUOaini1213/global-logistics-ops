"""README numbers are generated from results/summary.json, never typed by hand.

The README contains markers ``<!--n:KEY-->VALUE<!--/n-->``. ``run_all.py`` rewrites
VALUE from ``numbers(summary)``; tests/test_readme_numbers.py re-derives every value
from results/summary.json and fails if the README disagrees.
"""
from __future__ import annotations

import re

from .common import RESULTS, read_json

MARK = re.compile(r"<!--n:([a-z0-9_]+)-->(.*?)<!--/n-->", re.S)


def numbers(S: dict, mutation: dict | None = None) -> dict:
    sl, inv, val, land, rev = (S[k] for k in ("slotting", "inventory", "inventory_validation",
                                               "landed", "reverse"))
    L, sm = sl["large"], sl["small"]

    def ci(c, nd=1):
        return f"{c['mean']:.{nd}f}% (95% CI {c['lo']:.{nd}f}-{c['hi']:.{nd}f})"

    oc = inv["ocean_vs_air_rq_95"]
    return {
        "slot_skus": f"{sl['large_instance']['skus']:,}",
        "slot_slots": f"{sl['large_instance']['slots']:,}",
        "slot_seeds": str(sl["large_instance"]["seeds"]),
        "slot_heur_vs_random": ci(L["red_vs_random"]["heuristic"]["s_shape"]),
        "slot_heur_vs_abc": ci(L["red_vs_abc"]["heuristic"]["s_shape"]),
        "slot_turn_vs_abc": ci(L["red_vs_abc"]["turnover"]["s_shape"]),
        "slot_abc_vs_random": ci(L["red_vs_random"]["abc"]["s_shape"]),
        "slot_heur_vs_random_return": ci(L["red_vs_random"]["heuristic"]["return"]),
        "slot_milp_optimal": f"{sm['milp']['n_optimal']}/{sm['milp']['n']}",
        "slot_heur_gap": f"{sm['milp']['heuristic_gap_pct']['mean']:.2f}%",
        "slot_heur_gap_max": f"{sm['milp']['max_heuristic_gap_pct']:.2f}%",
        "slot_lambda": f"{sl['lambda_star']}",
        "inv_val_pass": f"{val['n_pass_3se']}/{val['n_checks']}",
        "inv_val_ci": f"{val['n_within_95ci']}/{val['n_checks']}",
        "inv_val_maxdiff": f"{val['max_abs_fill_rate_diff_pp']:.2f} pp",
        "inv_ocean_cost": f"USD {oc['ocean_cost_per_day']:.0f}/day",
        "inv_air_cost": f"USD {oc['air_cost_per_day']:.0f}/day",
        "inv_breakeven": f"USD {inv['breakeven_unit_value']:.0f}",
        "inv_bs_penalty": f"{inv['base_stock_vs_rQ_ocean_95_pct']:.0f}%",
        "inv_lt_ocean_csl": f"{100 * next(r['sim_csl'] for r in val['random_lt_formula'] if r['mode'] == 'ocean' and r['z'] == 1.0):.1f}%",
        "land_ddp_premium": f"{land['fcl_sg']['ddp_premium_pct']:.1f}%",
        "land_non_ddp_spread": f"{land['fcl_sg']['non_ddp_spread_pct']:.2f}%",
        "land_conservation": f"{land['conservation_max_spread_usd']:.1e} USD",
        "land_air_premium": f"{land['air_vs_lcl_per_unit']['air_premium_pct']:.1f}%",
        "rev_value": f"USD {rev['net_value']:,.0f}",
        "rev_gain": f"{rev['lp_vs_greedy_gain_pct']:.1f}%",
        "rev_shadow": f"USD {rev['shadow_price_penang_capacity']:.2f}",
        "rev_fd": f"USD {rev['finite_difference_penang']:.2f}",
        "rev_nocross": f"{rev['no_cross_border']['loss_pct']:.1f}%",
        "rev_threshold": f"USD {rev['refurb_threshold_price']}",
        "mut_caught": (f"{mutation['caught']}/{mutation['total']}"
                       + (f" ({mutation['invalid']} invalid, not run)" if mutation.get("invalid") else "")
                       if mutation else "not run yet"),
        **rollout_numbers(S["rollout"]),
    }


def _rng_pct(r: dict, nd: int = 0) -> str:
    return f"{100 * r['min']:.{nd}f}-{100 * r['max']:.{nd}f}%"


def _rng_days(vals: list) -> str:
    lo, hi = min(vals), max(vals)
    return f"{lo:+.0f} to {hi:+.0f} days" if round(lo) != round(hi) else f"{lo:+.0f} days"


def rollout_numbers(R: dict) -> dict:
    mk = R["markets"]
    n = {
        "ro_n_markets": str(len(mk)),
        "ro_cover": f"{R['selection']['cover_share_pct']:.1f}%",
        "ro_rho": f"{R['customs_score_vs_dwell']['spearman_rho']:.2f}",
        "ro_rho_n": str(R["customs_score_vs_dwell"]["n"]),
        "ro_m_ge_md": f"{R['data_checks']['ct_mean_ge_median']} of {R['data_checks']['ct_rows']}",
        "ro_av_bad": f"{R['data_checks']['air_mean_below_fifth_of_range']} of {R['data_checks']['air_rows']}",
        "ro_fit_err": f"{R['dwell_fit']['max_abs_mean_rel_err_pct']:.0f}%",
        "ro_check_z": f"{R['lead_check']['max_abs_mean_z']:.1f} / {R['lead_check']['max_abs_var_z']:.1f}",
        "ro_ontime_mean_sea": _rng_pct(R["plan_range"]["sea"]["on_time_oos_plan_mean"]),
        "ro_ontime_mean_air": _rng_pct(R["plan_range"]["air"]["on_time_oos_plan_mean"]),
        "ro_oos_p95_sea": _rng_pct(R["plan_range"]["sea"]["on_time_oos_plan_p95"], 1),
        "ro_oos_p95_air": _rng_pct(R["plan_range"]["air"]["on_time_oos_plan_p95"], 1),
        "ro_buf_sea": f"{R['plan_range']['sea']['buffer_days']['min']:.0f}-{R['plan_range']['sea']['buffer_days']['max']:.0f} days",
        "ro_buf_air": f"{R['plan_range']['air']['buffer_days']['min']:.0f}-{R['plan_range']['air']['buffer_days']['max']:.0f} days",
        "ro_port_mean": _rng_pct(R["sea_port_share"]["share_port_customs_mean"]),
        "ro_port_tail": _rng_pct(R["sea_port_share"]["share_port_customs_tail"]),
        "ro_load_mix": R["loading"]["weekly_mix"],
        "ro_load_fill": f"{100 * R['loading']['weekly_fill_volume']:.0f}%",
        "ro_load_eq": f"{R['loading']['random_40hc']['equal_lb']}/{R['loading']['random_40hc']['n']}",
        "ro_load_gap": f"{R['loading']['random_40hc']['max_gap_boxes']}",
        "ro_best_sea": f"{sum(v == 'sea' for v in R['policy_best'].values())}/{len(mk)}",
    }
    sens = R["sensitivity"]
    for key, var in (("transship", "no transshipment"), ("dwell_up", "destination dwell x 1.25"),
                     ("dwell_down", "destination dwell x 0.75"), ("fit", "dwell fit: median + mean"),
                     ("year", "LPI 2.0 year 2023"), ("speed14", "sea speed 14 kn")):
        n[f"ro_sens_{key}"] = _rng_days([sens[f"{m}|{var}"] for m in mk])
    for m in ("DZA", "ITA"):
        n[f"ro_sens_cape_{m.lower()}"] = f"{sens[f'{m}|Cape route (avoid Red Sea)']:+.0f} days"
    for m in mk:
        s, a = R["lead"][f"{m}|sea"], R["lead"][f"{m}|air"]
        k = m.lower()
        n.update({f"ro_{k}_sea_mean": f"{s['mean']:.0f}", f"ro_{k}_sea_p95": f"{s['p95']:.0f}",
                  f"ro_{k}_sea_buf": f"{s['buffer_95']:.0f}", f"ro_{k}_port": f"{100 * s['share_port_customs_mean']:.0f}%",
                  f"ro_{k}_tail": f"{100 * s['share_port_customs_tail']:.0f}%",
                  f"ro_{k}_air_mean": f"{a['mean']:.0f}", f"ro_{k}_air_p95": f"{a['p95']:.0f}"})
        for pol in ("sea", "split", "sea_expedite", "air"):
            n[f"ro_{k}_cost_{pol}"] = f"{R['policy'][f'{m}|{pol}']['cost_per_site']:,.0f}"
    cons = R["consolidation"]
    weeks = sorted({int(key.split("|")[1]) for key in cons})
    for w in weeks:
        rows = [cons[f"{m}|{w}"] for m in mk]
        n[f"ro_cons_{w}_containers"] = str(sum(r["containers"] for r in rows))
        n[f"ro_cons_{w}_mix"] = rows[0]["mix_per_sailing"]
        n[f"ro_cons_{w}_fill"] = f"{100 * rows[0]['fill_volume']:.0f}%"
        n[f"ro_cons_{w}_early"] = f"{sum(r['early_days'] for r in rows) / len(rows):.0f}"
        n[f"ro_cons_{w}_cost"] = f"{sum(r['cost_per_site'] for r in rows) / len(rows):,.0f}"
    best_w = [min(weeks, key=lambda w: cons[f"{m}|{w}"]["cost_per_site"]) for m in mk]
    n["ro_cons_best"] = f"{best_w.count(2)}/{len(mk)}"
    return n


def extract(text: str) -> dict:
    return {k: v for k, v in MARK.findall(text)}


def apply(text: str, nums: dict) -> str:
    def sub(m):
        k = m.group(1)
        if k not in nums:
            raise KeyError(f"README marker {k!r} has no generated value")
        return f"<!--n:{k}-->{nums[k]}<!--/n-->"
    return MARK.sub(sub, text)


def load_mutation() -> dict | None:
    p = RESULTS / "mutation_report.json"
    return read_json(p) if p.exists() else None
