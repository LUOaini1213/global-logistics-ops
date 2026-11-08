"""README numbers are generated from results/summary.json, never typed by hand.

The README contains markers ``<!--n:KEY-->VALUE<!--/n-->``. ``run_all.py`` rewrites
VALUE from ``numbers(summary)``; tests/test_readme_and_data.py re-derives every value
from results/summary.json and fails if the README disagrees.
"""
from __future__ import annotations

import datetime as dt
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
        **real_numbers(S["real_orders"]),
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
    kit, fl, fr = R["kit"], R["loading"]["weekly_floor"], R["policy_fresh"]
    n.update({
        "ro_kit_value_share": f"{100 * kit['critical_value_share']:.0f}%",
        "ro_kit_weight_share": f"{100 * kit['critical_weight_share']:.0f}%",
        "ro_buffer_day": f"USD {kit['buffer_usd_per_site_day']:.0f}",
        "ro_hold_rate": f"{100 * kit['holding_rate_per_year']:.0f}%/yr",
        "ro_exp_premium": f"{100 * kit['expedite_premium']:.0f}%",
        "ro_floor_pallet": str(fl["pallet_stacks"]), "ro_floor_other": str(fl["other_stacks"]),
        "ro_floor_m2": f"{fl['floor_m2']:.1f} m2", "ro_floor_40": f"{fl['floor_40ft_m2']:.1f} m2",
        "ro_rows_short": str(fl["pallet_rows_not_fitting"]), "ro_pallet_len": f"{fl['pallet_length_m']:.1f} m",
        "ro_short_mm": f"{fl['rows_short_mm']:.0f} mm",
        "ro_fresh_n": f"{fr['n_draws']:,}",
        "ro_fresh_sea": _rng_pct(fr["on_time"]["sea"], 1), "ro_fresh_air": _rng_pct(fr["on_time"]["air"], 1),
        "ro_fresh_split": _rng_pct(fr["on_time"]["split"], 1),
        "ro_fresh_expedite": _rng_pct(fr["on_time"]["sea_expedite"], 1),
        "ro_fresh_cost": f"{fr['max_abs_cost_change_pct']:.2f}%",
        "ro_fresh_same_best": f"{sum(fr['policy_best_fresh'][m] == R['policy_best'][m] for m in mk)}/{len(mk)}",
    })
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


def real_numbers(R: dict) -> dict:
    """Part 1b (real orders, Online Retail II)."""
    d, s, x, iv = R["data"], R["slotting"], R["synthetic_relaxed"], R["inventory"]

    def ci(c, nd=1):
        return f"{c['mean']:.{nd}f}% (95% CI {c['lo']:.{nd}f}-{c['hi']:.{nd}f})"

    def pct(v, nd=1):
        return f"{100 * v:.{nd}f}%"

    rm, cl, od = d["cleaning"]["removed"], d["cleaning"], d["orders"]
    out = {
        "real_rows_in": f"{cl['rows_in']:,}", "real_rows_kept": f"{cl['rows_kept']:,}",
        "real_rm_overlap": f"{rm['duplicate_sheet_overlap']:,}", "real_rm_cancel": f"{rm['cancellation']:,}",
        "real_rm_missing": f"{rm['missing_stock_code']:,}", "real_rm_nonproduct": f"{rm['non_product_code']:,}",
        "real_rm_qty": f"{rm['non_positive_quantity']:,}", "real_rm_price": f"{rm['non_positive_price']:,}",
        "real_codes_normalised": f"{cl['codes_normalised']:,}", "real_lines_merged": f"{cl['repeat_lines_merged']:,}",
        "real_invoices": f"{od['invoices']:,}", "real_products": f"{od['products']:,}",
        "real_train_products": f"{od['train_products']:,}", "real_cutoff_lines": f"{od['top_sku_min_train_lines']:,}",
        "real_train_orders": f"{od['train_orders_kept']:,}", "real_test_orders": f"{od['test_orders_kept']:,}",
        "real_test_dropped": f"{od['test_orders_without_top_sku']:,}",
        "real_train_period": f"{d['periods']['train'][0]} to "
                             f"{dt.date.fromisoformat(d['periods']['train'][1]) - dt.timedelta(days=1)}",
        "real_test_period": f"{d['periods']['test'][0]} to {d['periods']['test'][1]}",
        "real_coverage": pct(d["coverage"]["test"]["share"]), "real_train_coverage": pct(d["coverage"]["train"]["share"]),
        "real_test_days": f"{s['n_test_days']}", "real_test_weeks": f"{s['n_test_weeks']}",
        "real_seeds": f"{R['method_seeds']}", "real_boot": f"{R['n_boot']:,}", "real_lambda": f"{R['lambda']}",
        "real_heur_vs_random": ci(s["red_vs_random"]["heuristic"]["s_shape"]),
        "real_heur_vs_abc": ci(s["red_vs_abc"]["heuristic"]["s_shape"]),
        "real_turn_vs_abc": ci(s["red_vs_abc"]["turnover"]["s_shape"]),
        "real_abc_vs_random": ci(s["red_vs_random"]["abc"]["s_shape"]),
        "real_heur_vs_random_return": ci(s["red_vs_random"]["heuristic"]["return"]),
        "real_heur_vs_turnover": ci(s["red_vs_turnover"]["heuristic"]["s_shape"]),
        "real_week_ci": "{lo:.1f}-{hi:.1f}%".format(**s["week_blocks_heur_vs_random"]["s_shape"]),
        "real_seed_range": "{:.1f}-{:.1f}%".format(*s["seed_range_heur_vs_random"]),
        "real_travel_random": f"{s['travel']['random']['s_shape']['mean']:.0f} m",
        "real_travel_heur": f"{s['travel']['heuristic']['s_shape']['mean']:.0f} m",
        "srx_seeds": f"{x['seeds']}",
        "srx_heur_vs_random": ci(x["red_vs_random"]["heuristic"]["s_shape"]),
        "srx_heur_vs_abc": ci(x["red_vs_abc"]["heuristic"]["s_shape"]),
        "srx_turn_vs_abc": ci(x["red_vs_abc"]["turnover"]["s_shape"]),
        "srx_abc_vs_random": ci(x["red_vs_random"]["abc"]["s_shape"]),
        "srx_heur_vs_random_return": ci(x["red_vs_random"]["heuristic"]["return"]),
        "srx_heur_vs_turnover": ci(x["red_vs_turnover"]["heuristic"]["s_shape"]),
    }
    for who, st in (("real", s["structure"]), ("syn", x["structure"])):
        out.update({f"{who}_lines_mean": f"{st['lines_per_order_mean']:.1f}",
                    f"{who}_lines_median": f"{st['lines_per_order_median']:.0f}",
                    f"{who}_single_share": pct(st["single_line_share"]),
                    f"{who}_top20": pct(st["top20_pick_share"]), f"{who}_gini": f"{st['gini']:.2f}",
                    f"{who}_aff_orders": pct(st["affinity_order_share"]),
                    f"{who}_aff_weight": f"{st['affinity_weight_per_pick']:.2f}",
                    f"{who}_best_partner": pct(st["best_partner_rate"]),
                    f"{who}_aisles": f"{st['pick_aisles_random']:.1f}"})
    for k, (b, bx) in enumerate(zip(s["by_size"], x["by_size"])):
        lo, hi = b["lines"]
        out[f"size{k}_lines"] = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi else f"{lo}+")
        out[f"real_size{k}_share"] = pct(b["share_orders"], 0)
        out[f"real_size{k}_red"] = ci(b["red_heur_vs_random"])
        out[f"syn_size{k}_share"] = pct(bx["share_orders"], 0)
        out[f"syn_size{k}_red"] = (f"{bx['red_heur_vs_random_pooled']:.1f}%" if bx["orders"] >= 100
                                   else f"- (n = {bx['orders']})")
    g = iv["groups"]
    for grp, short in (("steady", "st"), ("intermittent", "in")):
        n = g[grp]["ocean_normal"]["n"]
        for mode in ("ocean", "air"):
            for meth, m in (("normal", "n"), ("bootstrap", "b")):
                c = g[grp][f"{mode}_{meth}"]
                out[f"ri_{short}_{mode}_{m}"] = pct(c["heldout_mean"])
                out[f"ri_{short}_{mode}_{m}_in"] = pct(c["insample_mean"])
                out[f"ri_{short}_{mode}_{m}_meet"] = f"{c['n_meet']}/{n}"
        out[f"ri_{short}_change"] = f"{g[grp]['median_demand_change_pct']:+.0f}%"
        out[f"ri_{short}_season"] = pct(g[grp]["mean_season_share"], 0)
        out[f"ri_{short}_adi"] = f"{g[grp]['mean_adi']:.2f}"
    out.update({"real_skus": f"{s['n_sku']:,}", "real_slots": f"{s['n_slots']:,}",
                "ri_adi_st_max": f"{iv['adi_steady_max']}", "ri_adi_in_min": f"{iv['adi_intermittent_min']}"})
    out.update({"ri_n": f"{g['steady']['ocean_normal']['n']}", "ri_target": pct(iv["target"], 0),
                "ri_q_ocean": f"{iv['q_days']['ocean']:.1f}", "ri_q_air": f"{iv['q_days']['air']:.1f}",
                "ri_warmup": f"{iv['warmup_days']}", "ri_reps": f"{iv['replay_reps']}"})
    return out


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
