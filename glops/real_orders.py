"""Part 1b - the slotting (Part 1) and inventory (Part 2) analyses re-run on REAL orders.

Data: UCI "Online Retail II" (CC BY 4.0), reduced by ``glops/online_retail.py`` to the committed
tables in data/public/online_retail_ii/. Training = 2009-12-01..2010-11-30, held out =
2010-12-01..2011-12-09; nothing from the held-out period is used to build any slotting or policy.

Slotting. The 1,000 SKUs on most training order lines go into the same 1,200-slot layout with
the same four methods and the same routing simulation as Part 1, and lambda as tuned on the
synthetic development seeds (not retuned). Velocity and co-pick affinity are estimated by
``slotting.estimate_stats`` on training orders only. SKU sizes and weights are not in the data
and the retailer sells small giftware, so the level rules are RELAXED (any SKU may use any
level); a synthetic control run with the same relaxation separates the effect of the real demand
from the effect of dropping the rules. One real dataset gives no seeds to average over, so
uncertainty is a paired bootstrap over held-out trading days (all methods share the resampled
days); the methods' own randomness (random and ABC placement, swap moves) is averaged over
several seeds.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from . import inventory as inv
from . import online_retail as orr
from . import slotting as sl
from .common import mean_ci

LAMBDA = 0.5                    # = results/slotting lambda_star (synthetic dev seeds); a test pins it
LS_ITERS = 60000                # = experiments.LS_ITERS
SEEDS = list(range(10))         # method randomness on the real instance
SEED_BASE = 30_000
SYN_SEEDS = list(range(20))     # = experiments.SLOT_LARGE_SEEDS (synthetic control)
N_BOOT = 2000
BOOT_SEED = 20261021
METHODS = ("random", "abc", "turnover", "heuristic")
POLICIES = ("s_shape", "return")
SIZE_BUCKETS = ((1, 1), (2, 3), (4, 8), (9, 20), (21, 10 ** 6))   # order lines (held-out)
ANY_LEVEL = 99


def real_instance(d=orr.DERIVED) -> tuple[sl.Instance, list]:
    """Instance built from the committed tables; returns (instance, held-out order dates)."""
    skus = orr.load_skus(d)
    _, _, train = orr.load_orders("train", d)
    _, test_dates, test = orr.load_orders("test", d)
    n = len(skus)
    params = sl.GeneratorParams(n_sku=n, n_kits=0, n_train_orders=len(train), n_test_orders=len(test))
    inst = sl.Instance(sl.Layout(), params, -1, category=np.full(n, "real"), size=np.full(n, "S"),
                       weight=np.full(n, np.nan), max_level=np.full(n, ANY_LEVEL),
                       popularity=np.full(n, np.nan), kits=[], train_orders=train, test_orders=test)
    sl.estimate_stats(inst, inst.train_orders)          # training orders only
    return inst, test_dates


def run_methods(inst, slots, rng, lam: float = LAMBDA, iters: int = LS_ITERS) -> dict:
    """The four Part 1 methods, called exactly as experiments.run_slotting calls them."""
    a = {"random": sl.assign_random(inst, slots, rng), "abc": sl.assign_abc(inst, slots, rng),
         "turnover": sl.assign_turnover(inst, slots, rng)}
    a["heuristic"] = sl.local_search(inst, slots, a["turnover"], lam, rng, iters)
    return a


# ------------------------------------------------------------------ statistics

def gini(x) -> float:
    """Gini coefficient of non-negative values (0 = all equal, (n-1)/n = one item has all)."""
    v = np.sort(np.asarray(x, dtype=float))
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    i = np.arange(1, n + 1)
    return float(((2 * i - n - 1) * v).sum() / (n * v.sum()))


def top_share(x, frac: float = 0.2) -> float:
    """Share of the total carried by the largest ``frac`` of items."""
    v = np.sort(np.asarray(x, dtype=float))[::-1]
    k = int(round(frac * len(v)))
    return float(v[:k].sum() / v.sum())


def aisles_per_order(slots, assign, orders) -> np.ndarray:
    sizes = np.array([len(o) for o in orders])
    oid = np.repeat(np.arange(len(orders)), sizes)
    key = np.unique(oid * 1000 + slots["aisle"][assign[np.concatenate(orders)]])
    return np.bincount(key // 1000, minlength=len(orders))


def structure(inst, slots, random_assign) -> dict:
    """Popularity skew and co-pick structure, computed identically for real and synthetic data."""
    v = inst.velocity
    tr = np.array([len(o) for o in inst.train_orders])
    te = np.array([len(o) for o in inst.test_orders])
    best = np.zeros(inst.n_sku)                      # strongest partner's co-pick count per SKU
    for (i, j), w in zip(inst.pairs.tolist(), inst.pair_w.tolist()):
        best[i], best[j] = max(best[i], w), max(best[j], w)
    return {"lines_per_order_mean": float(te.mean()), "lines_per_order_median": float(np.median(te)),
            "single_line_share": float((te == 1).mean()),
            "top20_pick_share": top_share(v), "gini": gini(v),
            "affinity_order_share": float(((tr >= 2) & (tr <= 8)).mean()),
            "affinity_pairs": int(len(inst.pair_w)),
            "affinity_weight_per_pick": float(inst.pair_w.sum() / v.sum()),
            "best_partner_rate": float(best.sum() / v.sum()),
            "pick_aisles_random": float(aisles_per_order(slots, random_assign, inst.test_orders).mean())}


def day_index(dates) -> tuple[np.ndarray, int]:
    days = sorted(set(dates))
    pos = {d: k for k, d in enumerate(days)}
    return np.array([pos[d] for d in dates]), len(days)


def boot_days(n_days: int, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> np.ndarray:
    """(n_boot, n_days) day indices drawn with replacement; shared by every comparison."""
    return np.random.default_rng(seed).integers(0, n_days, size=(n_boot, n_days))


def reduction_ci(base_day, other_day, idx) -> dict:
    """% travel reduction of ``other`` vs ``base`` (ratio of totals) with a paired percentile
    bootstrap CI: both methods are summed over the SAME resampled days."""
    base_day, other_day = np.asarray(base_day, float), np.asarray(other_day, float)
    point = 100.0 * (1.0 - other_day.sum() / base_day.sum())
    rb = 100.0 * (1.0 - other_day[idx].sum(axis=1) / base_day[idx].sum(axis=1))
    lo, hi = np.percentile(rb, [2.5, 97.5])
    return {"mean": float(point), "lo": float(lo), "hi": float(hi), "n_boot": int(len(idx))}


def mean_travel_ci(travel_day, count_day, idx) -> dict:
    point = float(travel_day.sum() / count_day.sum())
    rb = travel_day[idx].sum(axis=1) / count_day[idx].sum(axis=1)
    lo, hi = np.percentile(rb, [2.5, 97.5])
    return {"mean": point, "lo": float(lo), "hi": float(hi), "n_boot": int(len(idx))}


def _round(c, nd=2):
    return {k: (round(v, nd) if isinstance(v, float) else v) for k, v in c.items()}


def size_bucket(n_lines: np.ndarray) -> np.ndarray:
    b = np.full(len(n_lines), -1)
    for k, (lo, hi) in enumerate(SIZE_BUCKETS):
        b[(n_lines >= lo) & (n_lines <= hi)] = k
    return b


# ------------------------------------------------------------------ experiments

def slotting_real(log=print) -> tuple[dict, list]:
    inst, dates = real_instance()
    slots = inst.layout.slots()
    n = len(inst.test_orders)
    travel = {m: {p: np.zeros(n) for p in POLICIES} for m in METHODS}
    rows, rand0 = [], None
    for k in SEEDS:
        assigns = run_methods(inst, slots, np.random.default_rng(SEED_BASE + k))
        rand0 = assigns["random"] if rand0 is None else rand0
        for m in METHODS:
            a = assigns[m]
            assert sl.feasible(inst, slots, a), (k, m)
            row = {"seed": k, "method": m, "proxy": sl.proxy_objective(inst, slots, a, LAMBDA)}
            for p in POLICIES:
                t = sl.tour_lengths(inst.layout, slots, a, inst.test_orders, p)
                travel[m][p] += t / len(SEEDS)
                row[f"travel_{p}_m"] = float(t.mean())
            rows.append(row)
    day, n_days = day_index(dates)
    idx = boot_days(n_days)
    per_day = {m: {p: np.bincount(day, weights=travel[m][p], minlength=n_days) for p in POLICIES}
               for m in METHODS}
    count_day = np.bincount(day, minlength=n_days).astype(float)
    out = {"travel": {m: {p: _round(mean_travel_ci(per_day[m][p], count_day, idx)) for p in POLICIES}
                      for m in METHODS},
           "red_vs_random": {}, "red_vs_abc": {}, "red_vs_turnover": {}}
    for base, key in (("random", "red_vs_random"), ("abc", "red_vs_abc"), ("turnover", "red_vs_turnover")):
        for m in METHODS:
            if m == base or (base == "turnover" and m != "heuristic") or (base == "abc" and m == "random"):
                continue
            out[key][m] = {p: _round(reduction_ci(per_day[base][p], per_day[m][p], idx)) for p in POLICIES}
    # spread over the method seeds (each seed scored on its own)
    tab = {(r["seed"], r["method"]): r for r in rows}
    spread = [100 * (1 - tab[k, "heuristic"]["travel_s_shape_m"] / tab[k, "random"]["travel_s_shape_m"])
              for k in SEEDS]
    out["seed_range_heur_vs_random"] = [round(min(spread), 2), round(max(spread), 2)]
    # robustness: resample whole ISO weeks instead of days
    week, n_weeks = day_index([dt.date.fromisoformat(d).isocalendar()[:2] for d in dates])
    widx = boot_days(n_weeks)
    out["week_blocks_heur_vs_random"] = {p: _round(reduction_ci(
        np.bincount(week, weights=travel["random"][p], minlength=n_weeks),
        np.bincount(week, weights=travel["heuristic"][p], minlength=n_weeks), widx)) for p in POLICIES}
    out["n_test_weeks"] = n_weeks
    # by order size (held-out lines on the 1,000 SKUs)
    lines = np.array([len(o) for o in inst.test_orders])
    bucket = size_bucket(lines)
    by_size = []
    for b, (lo, hi) in enumerate(SIZE_BUCKETS):
        mask = bucket == b
        rd = np.bincount(day[mask], weights=travel["random"]["s_shape"][mask], minlength=n_days)
        hd = np.bincount(day[mask], weights=travel["heuristic"]["s_shape"][mask], minlength=n_days)
        by_size.append({"lines": [lo, hi if hi < 10 ** 6 else None], "orders": int(mask.sum()),
                        "share_orders": float(mask.mean()),
                        "share_travel_random": float(travel["random"]["s_shape"][mask].sum()
                                                     / travel["random"]["s_shape"].sum()),
                        "travel_random_m": float(travel["random"]["s_shape"][mask].mean()),
                        "red_heur_vs_random": _round(reduction_ci(rd, hd, idx))})
    out["by_size"] = by_size
    out["n_test_orders"], out["n_test_days"], out["n_train_orders"] = n, n_days, len(inst.train_orders)
    out["n_sku"], out["n_slots"] = inst.n_sku, inst.layout.n_slots
    out["structure"] = structure(inst, slots, rand0)
    log(f"  real orders: swap vs random {out['red_vs_random']['heuristic']['s_shape']['mean']:.1f}% "
        f"(S-shape, {n:,} held-out orders, {n_days} days)")
    return out, rows


def slotting_synthetic_relaxed(log=print) -> tuple[dict, list]:
    """Part 1's synthetic instances (same seeds and method rng) with the level rules relaxed."""
    rows, by_size, structs = [], [], []
    for seed in SYN_SEEDS:
        inst = sl.generate(seed)
        inst.max_level[:] = ANY_LEVEL
        slots = inst.layout.slots()
        assigns = run_methods(inst, slots, np.random.default_rng(10_000 + seed))
        lines = np.array([len(o) for o in inst.test_orders])
        bucket = size_bucket(lines)
        tt = {}
        for m in METHODS:
            assert sl.feasible(inst, slots, assigns[m]), (seed, m)
            row = {"seed": seed, "method": m}
            for p in POLICIES:
                t = sl.tour_lengths(inst.layout, slots, assigns[m], inst.test_orders, p)
                row[f"travel_{p}_m"] = float(t.mean())
                tt[m, p] = t
            rows.append(row)
        for b in range(len(SIZE_BUCKETS)):
            mask = bucket == b
            by_size.append((seed, b, int(mask.sum()), float(tt["random", "s_shape"][mask].sum()),
                            float(tt["heuristic", "s_shape"][mask].sum())))
        structs.append(structure(inst, slots, assigns["random"]))
    tab = {(r["seed"], r["method"]): r for r in rows}
    out = {"travel": {}, "red_vs_random": {}, "red_vs_abc": {}, "red_vs_turnover": {}}
    for m in METHODS:
        out["travel"][m] = {p: _round(mean_ci([tab[s, m][f"travel_{p}_m"] for s in SYN_SEEDS]))
                            for p in POLICIES}
    for base, key in (("random", "red_vs_random"), ("abc", "red_vs_abc"), ("turnover", "red_vs_turnover")):
        for m in METHODS:
            if m == base or (base == "turnover" and m != "heuristic") or (base == "abc" and m == "random"):
                continue
            out[key][m] = {p: _round(mean_ci([100 * (tab[s, base][f"travel_{p}_m"] - tab[s, m][f"travel_{p}_m"])
                                              / tab[s, base][f"travel_{p}_m"] for s in SYN_SEEDS]))
                           for p in POLICIES}
    sizes = []
    for b, (lo, hi) in enumerate(SIZE_BUCKETS):
        r = [x for x in by_size if x[1] == b]
        n_b = sum(x[2] for x in r)
        tr, th = sum(x[3] for x in r), sum(x[4] for x in r)
        sizes.append({"lines": [lo, hi if hi < 10 ** 6 else None], "orders": n_b,
                      "share_orders": n_b / sum(x[2] for x in by_size),
                      "red_heur_vs_random_pooled": 100 * (1 - th / tr) if n_b else None})
    out["by_size"] = sizes
    out["structure"] = {k: float(np.mean([s[k] for s in structs])) for k in structs[0]}
    out["seeds"] = len(SYN_SEEDS)
    log(f"  synthetic, level rules relaxed: swap vs random "
        f"{out['red_vs_random']['heuristic']['s_shape']['mean']:.1f}%")
    return out, rows


# ------------------------------------------------------------------ Part B: inventory replay
#
# For each selected SKU (online_retail.select_inventory_skus, training statistics only):
#   * Q covers the same number of days of mean demand as the Part 2 item's EOQ for that mode
#     (real order costs and unit values are unknown, so the cycle length is carried over);
#   * r is the smallest r = mu_LT + z sigma_LT (z on Z_GRID) whose simulated fill rate reaches the
#     95% target, with the Part 2 lead times (ocean / air) and the Part 2 simulation settings, under
#       "normal":    i.i.d. normal daily demand with the training mean and sd (calendar days, closed
#                    days count as zero; negative draws allowed so the model is exactly normal, as in
#                    the analytic check), or
#       "bootstrap": the training year resampled in whole weeks (keeps the closed-Saturday pattern
#                    and the real lumpiness; ignores seasonality, like the normal model);
#   * the policy is then replayed on the REAL held-out daily demand (warm-up: the last WARMUP_DAYS
#     training days), with REPLAY_REPS lead-time draws, and the achieved fill rate is reported.
#     The same replay over the training year (in-sample) separates model error from drift.

FILL_TARGET = 0.95
Z_COARSE = np.round(np.arange(-1.0, 12.0001, 0.25), 3)   # then refined in steps of Z_STEP
Z_STEP = 0.05
WARMUP_DAYS = 120
REPLAY_REPS = 200
REPLAY_SEED = 20261022
BLOCK_DAYS = 7
INV_METHODS = ("normal", "bootstrap")


def q_days(item: inv.Item, modes: dict) -> dict:
    """Order quantity in days of mean demand: the Part 2 item's EOQ / mean demand, per mode."""
    return {n: inv.eoq(item, m) / item.demand_mean for n, m in modes.items()}


def week_block_paths(train_units: np.ndarray, T: int, reps: int, rng) -> np.ndarray:
    """Demand paths built from whole training weeks drawn with replacement."""
    nb = len(train_units) // BLOCK_DAYS
    blocks = np.asarray(train_units[: nb * BLOCK_DAYS], dtype=float).reshape(nb, BLOCK_DAYS)
    k = -(-T // BLOCK_DAYS)
    return blocks[rng.integers(0, nb, size=(reps, k))].reshape(reps, k * BLOCK_DAYS)[:, :T]


def tune_r(item_c: inv.Item, mode: inv.Mode, Q: float, sim: dict, method: str, train_units=None) -> dict:
    """Smallest r = mu_LT + z sigma_LT meeting FILL_TARGET in simulation under the demand model:
    first on Z_COARSE, then in Z_STEP steps inside the bracket. Every candidate sees the same
    demand and lead-time paths, so this equals a full Z_STEP grid whenever fill rises with r."""
    mu, sig = inv.protection_sigma(item_c, mode)
    T = sim["n_days"] + sim["warmup"]
    if method == "normal":
        kw = {"allow_negative": True}
    elif method == "bootstrap":
        kw = {"demand": week_block_paths(train_units, T, sim["reps"], np.random.default_rng(sim["seed"]))}
    else:
        raise ValueError(method)

    def fill_at(z):
        r = mu + z * sig
        res = inv.simulate("rQ", {"r": r, "Q": np.full_like(r, Q)}, item_c, mode, sim["n_days"],
                           sim["warmup"], sim["reps"], sim["seed"], **kw)
        return res["fill_rate"].mean(axis=1)

    fill = fill_at(Z_COARSE)
    ok = np.flatnonzero(fill >= FILL_TARGET)
    if not len(ok):
        raise RuntimeError(f"no r on the grid reaches {FILL_TARGET:.0%} ({mode.name}, {method})")
    z, f = Z_COARSE[ok[0]], fill[ok[0]]
    if ok[0] > 0:
        fine = np.round(Z_COARSE[ok[0] - 1] + Z_STEP * np.arange(1, round(0.25 / Z_STEP) + 1), 3)
        ff = fill_at(fine)
        j = int(np.flatnonzero(ff >= FILL_TARGET)[0])
        z, f = fine[j], ff[j]
    return {"z": float(z), "r": float(mu + z * sig), "model_fill": float(f)}


def replay(item_c: inv.Item, mode: inv.Mode, r: float, Q: float, path: np.ndarray, warmup: int,
           seed: int = REPLAY_SEED, reps: int = REPLAY_REPS) -> np.ndarray:
    """Achieved fill rate on a given demand path, one value per lead-time draw."""
    res = inv.simulate("rQ", {"r": np.array([r]), "Q": np.array([Q])}, item_c, mode,
                       len(path) - warmup, warmup, reps, seed, demand=path)
    return res["fill_rate"][0]


def season_share(units: np.ndarray, dates) -> float:
    """Share of the units sold in September-December (4 of 12 months; 0.33 if flat)."""
    peak = np.array([int(d[5:7]) >= 9 for d in dates])
    return float(units[peak].sum() / units.sum())


def inventory_real(item: inv.Item, modes: dict, sim: dict, log=print) -> dict:
    daily = orr.load_daily()
    meta = orr.load_meta()
    sel = meta["inventory_selection"]
    stats = {s["stock_code"]: s for s in orr.load_demand_stats()}
    descr = {s["stock_code"]: s["description"] for s in orr.load_skus()}
    dates = np.array(daily["dates"])
    train = dates < orr.SPLIT_DAY
    n_train = int(train.sum())
    qd = q_days(item, modes)
    rows = []
    for group in ("steady", "intermittent"):
        for code in sel[group]:
            u = daily["units"][code]
            ut, uh = u[train], u[~train]
            item_c = inv.Item(**{**item.__dict__, "demand_mean": float(ut.mean()),
                                 "demand_sd": float(ut.std(ddof=1))})
            held = np.concatenate([ut[-WARMUP_DAYS:], uh])
            row = {"stock_code": code, "group": group, "description": descr[code],
                   "season_share": season_share(ut, dates[train]),
                   "adi": stats[code]["adi"], "cv2": stats[code]["cv2"],
                   "mean_train": item_c.demand_mean, "sd_train": item_c.demand_sd,
                   "mean_test": float(uh.mean()), "units_train": float(ut.sum()), "units_test": float(uh.sum())}
            for mname, mode in modes.items():
                Q = qd[mname] * item_c.demand_mean
                for method in INV_METHODS:
                    t = tune_r(item_c, mode, Q, sim, method, ut)
                    f_in = replay(item_c, mode, t["r"], Q, ut, WARMUP_DAYS)
                    f_out = replay(item_c, mode, t["r"], Q, held, WARMUP_DAYS)
                    pre = f"{mname}_{method}"
                    row.update({f"{pre}_z": t["z"], f"{pre}_r": t["r"], f"{pre}_Q": Q,
                                f"{pre}_model_fill": t["model_fill"],
                                f"{pre}_insample_fill": float(f_in.mean()),
                                f"{pre}_heldout_fill": float(f_out.mean()),
                                f"{pre}_heldout_p05": float(np.percentile(f_out, 5)),
                                f"{pre}_heldout_p95": float(np.percentile(f_out, 95))})
            rows.append(row)
    groups = {}
    for group in ("steady", "intermittent"):
        gr = [r for r in rows if r["group"] == group]
        groups[group] = {}
        for mname in modes:
            for method in INV_METHODS:
                pre = f"{mname}_{method}"
                out = [r[f"{pre}_heldout_fill"] for r in gr]
                groups[group][pre] = {"heldout_mean": float(np.mean(out)), "heldout_min": float(np.min(out)),
                                      "n_meet": int(sum(o >= FILL_TARGET for o in out)), "n": len(out),
                                      "insample_mean": float(np.mean([r[f"{pre}_insample_fill"] for r in gr])),
                                      "z_mean": float(np.mean([r[f"{pre}_z"] for r in gr]))}
        groups[group]["demand_change_pct"] = [100 * (r["mean_test"] / r["mean_train"] - 1) for r in gr]
        groups[group]["median_demand_change_pct"] = float(np.median(groups[group]["demand_change_pct"]))
        groups[group]["mean_season_share"] = float(np.mean([r["season_share"] for r in gr]))
        groups[group]["mean_adi"] = float(np.mean([r["adi"] for r in gr]))
    res = {"target": FILL_TARGET, "q_days": qd, "warmup_days": WARMUP_DAYS, "replay_reps": REPLAY_REPS,
           "adi_steady_max": orr.ADI_STEADY_MAX, "adi_intermittent_min": orr.ADI_INTERMITTENT_MIN,
           "train_days": n_train, "test_days": int((~train).sum()), "groups": groups, "skus": rows,
           "selection": sel}
    g = groups
    log(f"  real demand replay (ocean, normal-tuned): steady {100 * g['steady']['ocean_normal']['heldout_mean']:.1f}%, "
        f"intermittent {100 * g['intermittent']['ocean_normal']['heldout_mean']:.1f}% fill vs 95% target")
    return res
