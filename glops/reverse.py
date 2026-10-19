"""Part 4 - reverse logistics (returns -> refurbish / harvest / recycle) as an LP.

Network data: config/reverse_network.json (ILLUSTRATIVE). Monthly flows, USD.

Decision variables
  x[h,r,g]  grade g in {A,B} returns sent from hub h to refurb centre r
  y[h,c,g]  grade g in {A,B,C} returns sent from hub h to recycler c
  f[r,c]    units failing refurbishment at r, sent to recycler c
  s[r]      refurbished units from r sold on the secondary market
  p[r]      refurbished units from r harvested for parts
  o[h,g] / o[r,'fail']  overflow to uncapacitated, expensive spot recycling at the source
Constraints
  every return is processed (supply rows), refurb and recycler capacities,
  refurb output = sum yield_g * x  (= s + p),  failures = sum (1 - yield_g) * x,
  total resale <= market cap.  Cross-border recycling flows can be switched off.
Objective: minimise transport + processing + compliance - resale - parts revenue.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from numbers import Real

import numpy as np

from .common import CONFIG, read_json


def load_network():
    return read_json(CONFIG / "reverse_network.json")


class _LP:
    def __init__(self):
        self.cost, self.names, self.rows = [], [], []

    def var(self, name, cost):
        self.names.append(name)
        self.cost.append(float(cost))
        return len(self.cost) - 1

    def row(self, name, idx, coef, lo, hi):
        self.rows.append((name, list(idx), list(coef), lo, hi))

    def solve(self):
        import highspy
        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        h.setOptionValue("threads", 1)
        inf = highspy.kHighsInf
        n = len(self.cost)
        h.addVars(n, np.zeros(n), np.full(n, inf))
        h.changeColsCost(n, np.arange(n, dtype=np.int32), np.array(self.cost))
        for _, idx, coef, lo, hi in self.rows:
            h.addRow(-inf if lo is None else lo, inf if hi is None else hi, len(idx),
                     np.array(idx, dtype=np.int32), np.array(coef, dtype=float))
        h.run()
        status = h.modelStatusToString(h.getModelStatus())
        if status != "Optimal":
            return {"status": status}
        sol = h.getSolution()
        return {"status": status, "x": np.array(sol.col_value),
                "duals": {r[0]: float(d) for r, d in zip(self.rows, sol.row_dual)},
                "objective": float(h.getInfo().objective_function_value)}


def build_and_solve(net: dict) -> dict:
    hubs, refs, recs = net["hubs"], net["refurb_centres"], net["recyclers"]
    tr, r2c, cb, mk = (net["transport_per_unit"], net["refurb_to_recycler_per_unit"],
                       net["cross_border"], net["market"])
    allow_x = cb["allow_cross_border_recycling"]
    lp = _LP()
    X, Y, F, S, P, O = {}, {}, {}, {}, {}, {}
    ov = net["overflow"]["cost_per_unit"]
    for h, hv in hubs.items():
        for r, rv in refs.items():
            border = cb["returns_for_repair_per_unit"] if hv["country"] != rv["country"] else 0.0
            for g in ("A", "B"):
                X[h, r, g] = lp.var(f"x|{h}|{r}|{g}", tr[h][r] + border + rv["cost"][g])
        for c, cv in recs.items():
            cross = hv["country"] != cv["country"]
            if cross and not allow_x:
                continue
            border = cb["waste_for_recycling_per_unit"] if cross else 0.0
            for g in ("A", "B", "C"):
                Y[h, c, g] = lp.var(f"y|{h}|{c}|{g}", tr[h][c] + border + cv["net_cost"])
        for g in ("A", "B", "C"):
            O[h, g] = lp.var(f"o|{h}|{g}", ov)
    for r, rv in refs.items():
        for c, cv in recs.items():
            cross = rv["country"] != cv["country"]
            if cross and not allow_x:
                continue
            border = cb["waste_for_recycling_per_unit"] if cross else 0.0
            F[r, c] = lp.var(f"f|{r}|{c}", r2c[r][c] + border + cv["net_cost"])
        O[r, "fail"] = lp.var(f"o|{r}|fail", ov)
        S[r] = lp.var(f"s|{r}", -mk["resale_price"])
        P[r] = lp.var(f"p|{r}", -mk["parts_value"])

    for h, hv in hubs.items():
        for g, qty in hv["returns"].items():
            idx = [X[k] for k in X if k[0] == h and k[2] == g] + \
                [Y[k] for k in Y if k[0] == h and k[2] == g] + [O[h, g]]
            lp.row(f"supply|{h}|{g}", idx, [1.0] * len(idx), float(qty), float(qty))
    for r, rv in refs.items():
        xs = [k for k in X if k[1] == r]
        lp.row(f"refurb_cap|{r}", [X[k] for k in xs], [1.0] * len(xs), None, float(rv["capacity"]))
        lp.row(f"refurb_out|{r}", [S[r], P[r]] + [X[k] for k in xs],
               [1.0, 1.0] + [-rv["yield"][k[2]] for k in xs], 0.0, 0.0)
        fs = [F[k] for k in F if k[0] == r] + [O[r, "fail"]]
        lp.row(f"refurb_fail|{r}", fs + [X[k] for k in xs],
               [1.0] * len(fs) + [-(1 - rv["yield"][k[2]]) for k in xs], 0.0, 0.0)
    for c, cv in recs.items():
        idx = [Y[k] for k in Y if k[1] == c] + [F[k] for k in F if k[1] == c]
        lp.row(f"recycle_cap|{c}", idx, [1.0] * len(idx), None, float(cv["capacity"]))
    lp.row("market", list(S.values()), [1.0] * len(S), None, float(mk["resale_demand_cap"]))

    res = lp.solve()
    if res["status"] != "Optimal":
        return {"status": res["status"]}
    v = res["x"]
    flows = {"x": {k: v[i] for k, i in X.items()}, "y": {k: v[i] for k, i in Y.items()},
             "f": {k: v[i] for k, i in F.items()}, "s": {k: v[i] for k, i in S.items()},
             "p": {k: v[i] for k, i in P.items()}, "o": {k: v[i] for k, i in O.items()}}
    return {"status": "Optimal", "net_cost": res["objective"], "flows": flows,
            "duals": res["duals"], "summary": summarise(net, flows)}


def evaluate_flows(net: dict, flows: dict) -> float:
    """Net cost of a given flow plan with the LP's cost coefficients (for baselines)."""
    hubs, refs, recs = net["hubs"], net["refurb_centres"], net["recyclers"]
    tr, r2c, cb, mk = (net["transport_per_unit"], net["refurb_to_recycler_per_unit"],
                       net["cross_border"], net["market"])
    cost = 0.0
    for (h, r, g), q in flows["x"].items():
        border = cb["returns_for_repair_per_unit"] if hubs[h]["country"] != refs[r]["country"] else 0.0
        cost += q * (tr[h][r] + border + refs[r]["cost"][g])
    for (h, c, g), q in flows["y"].items():
        border = cb["waste_for_recycling_per_unit"] if hubs[h]["country"] != recs[c]["country"] else 0.0
        cost += q * (tr[h][c] + border + recs[c]["net_cost"])
    for (r, c), q in flows["f"].items():
        border = cb["waste_for_recycling_per_unit"] if refs[r]["country"] != recs[c]["country"] else 0.0
        cost += q * (r2c[r][c] + border + recs[c]["net_cost"])
    cost += net["overflow"]["cost_per_unit"] * sum(flows.get("o", {}).values())
    cost -= mk["resale_price"] * sum(flows["s"].values())
    cost -= mk["parts_value"] * sum(flows["p"].values())
    return float(cost)


_FLOW_PARTS = ("x", "y", "f", "s", "p", "o")


def _finite_real(value) -> bool:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _validate_flow_inputs(net: dict, flows, tol) -> tuple[dict, list]:
    """Validate the plan's domain before sums: NaN residuals and unknown arcs cannot certify feasibility."""
    clean = {part: {} for part in _FLOW_PARTS}
    bad = []
    if not _finite_real(tol) or tol < 0:
        return clean, ["tolerance must be a finite nonnegative real number"]
    if not isinstance(flows, Mapping):
        return clean, ["flows must be a mapping of known flow families"]
    for part in flows:
        if part not in _FLOW_PARTS:
            bad.append(f"unknown flow family {part!r}")
    hubs, refs, recs = net["hubs"], net["refurb_centres"], net["recyclers"]
    for part in _FLOW_PARTS:
        entries = flows.get(part, {})
        if not isinstance(entries, Mapping):
            bad.append(f"invalid {part}: flow entries must be a mapping")
            continue
        for key, value in entries.items():
            valid = False
            if part in ("s", "p"):
                valid = isinstance(key, str) and key in refs
            else:
                size = 3 if part in ("x", "y") else 2
                if isinstance(key, tuple) and len(key) == size and all(isinstance(k, str) for k in key):
                    if part == "x":
                        valid = key[0] in hubs and key[1] in refs and key[2] in ("A", "B")
                    elif part == "y":
                        valid = key[0] in hubs and key[1] in recs and key[2] in ("A", "B", "C")
                    elif part == "f":
                        valid = key[0] in refs and key[1] in recs
                    elif part == "o":
                        valid = ((key[0] in hubs and key[1] in ("A", "B", "C"))
                                 or (key[0] in refs and key[1] == "fail"))
            if not valid:
                bad.append(f"invalid {part} flow key {key!r}: check nodes, grade and key shape")
            if not _finite_real(value):
                bad.append(f"invalid {part} quantity at {key!r}: expected a finite real number")
            elif valid:
                clean[part][key] = float(value)
                if part in ("y", "f") and value > tol and not net["cross_border"]["allow_cross_border_recycling"]:
                    source = hubs[key[0]] if part == "y" else refs[key[0]]
                    if source["country"] != recs[key[1]]["country"]:
                        bad.append(f"cross-border recycling is disabled: {part} {key!r}")
    return clean, bad


def check_flows(net: dict, flows: dict, tol: float = 1e-6) -> list:
    """Return readable violations (empty = feasible), accepting sparse plans with omitted zero-flow families."""
    flows, bad = _validate_flow_inputs(net, flows, tol)
    if bad:
        return bad
    hubs, refs, recs = net["hubs"], net["refurb_centres"], net["recyclers"]
    for h, hv in hubs.items():
        for g, qty in hv["returns"].items():
            out = sum(q for k, q in flows["x"].items() if k[0] == h and k[2] == g) + \
                sum(q for k, q in flows["y"].items() if k[0] == h and k[2] == g) + \
                flows.get("o", {}).get((h, g), 0.0)
            if abs(out - qty) > tol * max(1, qty):
                bad.append(f"supply {h} {g}: {out} != {qty}")
    for r, rv in refs.items():
        inflow = sum(q for k, q in flows["x"].items() if k[1] == r)
        if inflow > rv["capacity"] + tol:
            bad.append(f"refurb cap {r}")
        good = sum(q * rv["yield"][k[2]] for k, q in flows["x"].items() if k[1] == r)
        if abs(flows["s"].get(r, 0) + flows["p"].get(r, 0) - good) > 1e-4:
            bad.append(f"refurb output {r}")
        fail = sum(q * (1 - rv["yield"][k[2]]) for k, q in flows["x"].items() if k[1] == r)
        if abs(sum(q for k, q in flows["f"].items() if k[0] == r) +
               flows.get("o", {}).get((r, "fail"), 0.0) - fail) > 1e-4:
            bad.append(f"refurb failures {r}")
    for c, cv in recs.items():
        inflow = sum(q for k, q in flows["y"].items() if k[1] == c) + \
            sum(q for k, q in flows["f"].items() if k[1] == c)
        if inflow > cv["capacity"] + tol:
            bad.append(f"recycler cap {c}")
    if sum(flows["s"].values()) > net["market"]["resale_demand_cap"] + tol:
        bad.append("market cap")
    for part in ("x", "y", "f", "s", "p", "o"):
        if any(q < -tol for q in flows.get(part, {}).values()):
            bad.append(f"negative {part}")
    return bad


def greedy_baseline(net: dict) -> dict:
    """Rule of thumb: each hub sends A then B grade to its nearest refurb centre while
    capacity lasts; everything else goes to the nearest recycler with capacity;
    refurb failures go to the recycler nearest the refurb centre; refurbished
    units are sold until the market cap, then harvested."""
    hubs, refs, recs = net["hubs"], net["refurb_centres"], net["recyclers"]
    tr, r2c = net["transport_per_unit"], net["refurb_to_recycler_per_unit"]
    allow_x = net["cross_border"]["allow_cross_border_recycling"]
    rcap = {r: v["capacity"] for r, v in refs.items()}
    ccap = {c: v["capacity"] for c, v in recs.items()}
    flows = {"x": {}, "y": {}, "f": {}, "s": {}, "p": {}, "o": {}}

    def recycle(src_country, dist, qty, key_fn, overflow_key):
        for c in sorted(recs, key=lambda c: dist[c]):
            if not allow_x and recs[c]["country"] != src_country:
                continue
            q = min(qty, ccap[c])
            if q > 0:
                k = key_fn(c)
                flows[k[0]][k[1]] = flows[k[0]].get(k[1], 0) + q
                ccap[c] -= q
                qty -= q
            if qty <= 1e-9:
                return
        if qty > 1e-9:
            flows["o"][overflow_key] = flows["o"].get(overflow_key, 0.0) + qty

    leftovers = []
    for g in ("A", "B"):
        for h, hv in hubs.items():
            qty = hv["returns"][g]
            for r in sorted(refs, key=lambda r: tr[h][r]):
                q = min(qty, rcap[r])
                if q > 0:
                    flows["x"][h, r, g] = flows["x"].get((h, r, g), 0) + q
                    rcap[r] -= q
                    qty -= q
            if qty > 0:
                leftovers.append((h, g, qty))
    for r, rv in refs.items():
        fail = sum(q * (1 - rv["yield"][k[2]]) for k, q in flows["x"].items() if k[1] == r)
        if fail > 0:
            recycle(rv["country"], r2c[r], fail, lambda c, r=r: ("f", (r, c)), (r, "fail"))
    for h, hv in hubs.items():
        recycle(hv["country"], tr[h], hv["returns"]["C"], lambda c, h=h: ("y", (h, c, "C")), (h, "C"))
    for h, g, qty in leftovers:
        recycle(hubs[h]["country"], tr[h], qty, lambda c, h=h, g=g: ("y", (h, c, g)), (h, g))
    cap = net["market"]["resale_demand_cap"]
    for r, rv in refs.items():
        good = sum(q * rv["yield"][k[2]] for k, q in flows["x"].items() if k[1] == r)
        sold = min(good, cap)
        cap -= sold
        flows["s"][r], flows["p"][r] = sold, good - sold
    return flows


def summarise(net: dict, flows: dict) -> dict:
    """Summarise a feasible plan; a zero supply/capacity ratio is 0.0 (no activity, not performance evidence)."""
    bad = check_flows(net, flows)
    if bad:
        raise ValueError("cannot summarise infeasible flows: " + "; ".join(bad))
    flows = {part: {key: float(q) for key, q in flows.get(part, {}).items()} for part in _FLOW_PARTS}

    def ratio(activity, total):
        # Feasibility was checked first: zero capacity cannot hide positive throughput behind an epsilon.
        return float(activity / total) if total > 0 else 0.0

    ret = {g: sum(h["returns"][g] for h in net["hubs"].values()) for g in ("A", "B", "C")}
    refurb_in = {g: sum(q for k, q in flows["x"].items() if k[2] == g) for g in ("A", "B")}
    util = {r: ratio(sum(q for k, q in flows["x"].items() if k[1] == r), v["capacity"])
            for r, v in net["refurb_centres"].items()}
    rec_util = {c: ratio(sum(q for k, q in flows["y"].items() if k[1] == c) +
                         sum(q for k, q in flows["f"].items() if k[1] == c), v["capacity"])
                for c, v in net["recyclers"].items()}
    cross_waste = sum(q for k, q in flows["y"].items()
                      if net["hubs"][k[0]]["country"] != net["recyclers"][k[1]]["country"])
    cross_waste += sum(q for k, q in flows["f"].items()
                       if net["refurb_centres"][k[0]]["country"] != net["recyclers"][k[1]]["country"])
    return {
        "returns": ret,
        "refurb_share_A": ratio(refurb_in["A"], ret["A"]),
        "refurb_share_B": ratio(refurb_in["B"], ret["B"]),
        "resold": sum(flows["s"].values()),
        "parts": sum(flows["p"].values()),
        "recycled": sum(flows["y"].values()) + sum(flows["f"].values()),
        "overflow": sum(flows.get("o", {}).values()),
        "refurb_utilisation": util, "recycler_utilisation": rec_util,
        "cross_border_waste_units": cross_waste,
    }


def with_changes(net: dict, **kw) -> dict:
    """Copy of the network with scenario changes applied."""
    n = copy.deepcopy(net)
    if "resale_price" in kw:
        n["market"]["resale_price"] = kw["resale_price"]
    if "refurb_capacity" in kw:
        for r, cap in kw["refurb_capacity"].items():
            n["refurb_centres"][r]["capacity"] = cap
    if "volume_mult" in kw:
        for hv in n["hubs"].values():
            hv["returns"] = {g: q * kw["volume_mult"] for g, q in hv["returns"].items()}
    if "transport_mult" in kw:
        for tab in (n["transport_per_unit"], n["refurb_to_recycler_per_unit"]):
            for k, row in tab.items():
                if not k.startswith("_"):
                    tab[k] = {d: c * kw["transport_mult"] for d, c in row.items()}
    if "allow_cross_border_recycling" in kw:
        n["cross_border"]["allow_cross_border_recycling"] = kw["allow_cross_border_recycling"]
    if "market_cap" in kw:
        n["market"]["resale_demand_cap"] = kw["market_cap"]
    return n
