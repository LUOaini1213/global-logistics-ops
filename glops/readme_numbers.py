"""README numbers are generated from results/summary.json, never typed by hand.

The README contains markers ``<!--n:KEY-->VALUE<!--/n-->``. ``run_all.py`` rewrites
VALUE from ``numbers(summary)``; tests/test_readme_numbers.py re-derives every value
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
        "mut_caught": f"{mutation['caught']}/{mutation['total']}" if mutation else "not run yet",
        **real_numbers(S["real_orders"]),
    }


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
