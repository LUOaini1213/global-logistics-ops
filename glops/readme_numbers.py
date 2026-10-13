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
        "mut_caught": f"{mutation['caught']}/{mutation['total']}" if mutation else "not run yet",
    }


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
