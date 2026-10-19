"""Part 2 - single-echelon inventory policy simulation (ocean vs air replenishment).

Daily periodic review, full backordering. Event order within a day t:
  1. demand D_t is served from on-hand stock; the unmet part is backordered;
  2. replenishments due on day t are received (they clear backorders first);
  3. the inventory position IP = net inventory + on order is reviewed and the
     policy may place an order; it arrives at the END of day t + L (step 2 of
     that day), where L is the lead time drawn for that order. For L = 0,
     receipt occurs immediately after the order, before end-of-day accounting;
     it clears backorders but does not retroactively fill today's demand.
With this convention the protection interval of a base-stock policy is L + 1 days,
which is what the analytic check in ``analytic_base_stock`` uses.

Policies (vectorised: every row of the state arrays is one (parameter, replication)):
  base_stock  S        : order S - IP every day (if positive)
  sS          (s, S)   : if IP <= s, order S - IP
  rQ          (r, Q)   : if IP <= r, order the smallest multiple of Q lifting IP above r

All monetary inputs are ILLUSTRATIVE (see config/inventory_scenario.json).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class Mode:
    name: str
    lt_mean: float          # days
    lt_sd: float            # days (0 => deterministic)
    freight_per_unit: float
    order_fixed_cost: float


@dataclass
class Item:
    demand_mean: float      # units/day
    demand_sd: float        # units/day
    unit_value: float       # USD, used for holding cost
    holding_rate: float     # per year, fraction of value
    backorder_cost: float   # USD per unit backordered per day
    transit_holding: bool = True   # buyer owns goods in transit (e.g. FCA/FOB purchase)

    @property
    def h(self) -> float:
        return self.unit_value * self.holding_rate / 365.0


def draw_lead_times(rng, mode: Mode, shape) -> np.ndarray:
    if (not np.isfinite(mode.lt_mean) or not np.isfinite(mode.lt_sd)
            or mode.lt_mean < 0 or mode.lt_sd < 0
            or (mode.lt_mean == 0 and mode.lt_sd > 0)):
        raise ValueError("lead time mean and sd must be finite and nonnegative; "
                         "a random lead time requires a positive mean")
    if mode.lt_sd <= 0:
        return np.full(shape, int(round(mode.lt_mean)), dtype=int)
    # lognormal with the requested mean / sd, rounded to whole days, >= 1
    s2 = np.log(1 + (mode.lt_sd / mode.lt_mean) ** 2)
    mu = np.log(mode.lt_mean) - s2 / 2
    return np.maximum(1, np.rint(rng.lognormal(mu, np.sqrt(s2), shape))).astype(int)


def draw_demand(rng, item: Item, shape, allow_negative: bool = False, season=None) -> np.ndarray:
    d = rng.normal(item.demand_mean, item.demand_sd, shape)
    if season is not None:
        d = d * season[None, : shape[1]]
    return d if allow_negative else np.maximum(d, 0.0)


def simulate(policy: str, params: dict, item: Item, mode: Mode, n_days: int, warmup: int,
             reps: int, seed: int, allow_negative: bool = False, season=None,
             continuous: bool = True, bucket=None) -> dict:
    """Simulate len(param arrays) configurations x reps replications.

    params: {"S": array} | {"s": array, "S": array} | {"r": array, "Q": array}
    Returns per-configuration arrays of shape (n_cfg, reps) for each metric.
    bucket: optional int array (length warmup + n_days) mapping each day to a group
    (e.g. calendar month); adds "bucket_fill_rate" of shape (n_cfg, n_buckets).
    """
    key = next(iter(params))
    n_cfg = len(np.atleast_1d(params[key]))
    rows = n_cfg * reps
    rng = np.random.default_rng(seed)
    T = n_days + warmup
    # common random numbers: every configuration sees the same demand / lead-time paths
    dem = np.tile(draw_demand(rng, item, (reps, T), allow_negative, season), (n_cfg, 1))
    lts = np.tile(draw_lead_times(rng, mode, (reps, T)), (n_cfg, 1))
    if not continuous:
        dem = np.rint(dem)
    P = {k: np.repeat(np.asarray(v, dtype=float), reps) for k, v in params.items()}

    max_lt = int(lts.max()) + 2
    arrivals = np.zeros((rows, T + max_lt))
    start_level = P["S"] if "S" in P else P["r"] + P["Q"]
    ni = start_level.copy()          # net inventory, start "full"
    on_order = np.zeros(rows)
    rix = np.arange(rows)
    if bucket is not None:
        nb = int(np.max(bucket)) + 1
        b_dem, b_short = np.zeros((rows, nb)), np.zeros((rows, nb))

    acc = {k: np.zeros(rows) for k in ("demand", "short", "on_hand", "backorder", "pipeline",
                                       "orders", "units_ordered", "stockout_days")}
    for t in range(T):
        d = dem[:, t]
        short = np.maximum(0.0, d - np.maximum(ni, 0.0))
        ni = ni - d
        rec = arrivals[:, t]
        ni = ni + rec
        on_order = on_order - rec
        ip = ni + on_order
        if policy == "base_stock":
            q = np.maximum(0.0, P["S"] - ip)
        elif policy == "sS":
            q = np.where(ip <= P["s"], P["S"] - ip, 0.0)
        elif policy == "rQ":
            n = np.where(ip <= P["r"], np.floor((P["r"] - ip) / P["Q"]) + 1, 0.0)
            q = n * P["Q"]
        else:
            raise ValueError(policy)
        placed = q > 1e-12
        if placed.any():
            idx = rix[placed]
            np.add.at(arrivals, (idx, t + lts[idx, t]), q[placed])
            on_order = on_order + q
            # Today's scheduled-receipt step has already run. Keep immediate
            # receipts in arrivals for the balance check and apply them now.
            immediate = placed & (lts[:, t] == 0)
            ni[immediate] += q[immediate]
            on_order[immediate] -= q[immediate]
        if t >= warmup:
            acc["demand"] += d
            acc["short"] += short
            acc["on_hand"] += np.maximum(ni, 0.0)
            acc["backorder"] += np.maximum(-ni, 0.0)
            acc["pipeline"] += on_order
            acc["orders"] += placed
            acc["units_ordered"] += q
            acc["stockout_days"] += short > 0
            if bucket is not None:
                b_dem[:, bucket[t]] += d
                b_short[:, bucket[t]] += short

    nd = float(n_days)
    out = {
        "fill_rate": 1.0 - acc["short"] / np.maximum(acc["demand"], 1e-12),
        "csl": 1.0 - acc["stockout_days"] / nd,
        "on_hand": acc["on_hand"] / nd,
        "backorder": acc["backorder"] / nd,
        "pipeline": acc["pipeline"] / nd,
        "orders_per_day": acc["orders"] / nd,
        "units_ordered_per_day": acc["units_ordered"] / nd,
        "demand_per_day": acc["demand"] / nd,
        "final_balance_error": _balance_error(start_level, dem, arrivals, ni, T),
        "pipeline_error": np.abs(on_order - arrivals[:, T:].sum(axis=1)),
    }
    cost = {
        "holding": out["on_hand"] * item.h,
        "pipeline": out["pipeline"] * item.h if item.transit_holding else 0 * out["pipeline"],
        "backorder": out["backorder"] * item.backorder_cost,
        "ordering": out["orders_per_day"] * mode.order_fixed_cost,
        "freight": out["units_ordered_per_day"] * mode.freight_per_unit,
    }
    cost["total"] = sum(cost.values())
    out.update({f"cost_{k}": v for k, v in cost.items()})
    res = {k: np.asarray(v).reshape(n_cfg, reps) for k, v in out.items()}
    if bucket is not None:
        bd = b_dem.reshape(n_cfg, reps, -1).sum(axis=1)
        bs = b_short.reshape(n_cfg, reps, -1).sum(axis=1)
        res["bucket_fill_rate"] = 1.0 - bs / np.maximum(bd, 1e-12)
    return res


def _balance_error(start, dem, arrivals, ni_end, T):
    """Inventory balance: NI_end = NI_0 - total demand + total receipts."""
    received = arrivals[:, :T].sum(axis=1)
    return np.abs(start - dem.sum(axis=1) + received - ni_end)


# --------------------------------------------------------------------------- analytics

def normal_loss(z):
    """Standard normal first-order loss G(z) = E[(Z - z)^+]."""
    z = np.asarray(z, dtype=float)
    return stats.norm.pdf(z) - z * (1.0 - stats.norm.cdf(z))


def analytic_base_stock(S: float, item: Item, L: int) -> dict:
    """Exact results for i.i.d. normal daily demand, deterministic integer lead time L,
    daily review with the event order of ``simulate`` (protection interval L + 1)."""
    mu, sd = item.demand_mean, item.demand_sd

    def exp_pos(n):  # E[(D(n) - S)^+], D(n) ~ N(n mu, n sd^2)
        if n == 0:
            return max(0.0, -S)
        s = sd * np.sqrt(n)
        return float(s * normal_loss((S - n * mu) / s))

    short = exp_pos(L + 1) - exp_pos(L)
    s_l = sd * np.sqrt(L)
    eb = exp_pos(L)
    return {
        "fill_rate": 1.0 - short / mu,
        "csl": float(stats.norm.cdf((S - (L + 1) * mu) / (sd * np.sqrt(L + 1)))),
        "backorder": eb,
        "on_hand": S - L * mu + eb,        # E[(S - D_L)^+] = S - E[D_L] + E[(D_L - S)^+]
        "pipeline": L * mu,
        "z": (S - (L + 1) * mu) / (sd * np.sqrt(L + 1)),
        "s_l": s_l,
    }


def protection_sigma(item: Item, mode: Mode, review: int = 1) -> tuple[float, float]:
    """Mean and sd of demand over L + review with random L (classic approximation,
    ignores order crossing)."""
    m = item.demand_mean * (mode.lt_mean + review)
    var = (mode.lt_mean + review) * item.demand_sd ** 2 + item.demand_mean ** 2 * mode.lt_sd ** 2
    return m, float(np.sqrt(var))


def eoq(item: Item, mode: Mode) -> float:
    return float(np.sqrt(2 * mode.order_fixed_cost * item.demand_mean / item.h))
