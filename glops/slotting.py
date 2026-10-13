"""Part 1 - warehouse slotting.

Everything here is SYNTHETIC. The generator mimics a spare-parts / network-equipment
distribution centre (optical transceivers, patch cables, PSUs, line cards, router
chassis, radio units, antennas, small cells). Parameters are documented in
``GeneratorParams``; every random draw comes from ``numpy.random.default_rng(seed)``
streams so a seed fully determines the warehouse, the order history used for
optimisation, and the held-out orders used for evaluation.

Layout: single-block, parallel-aisle warehouse. Racks on both sides of each aisle,
``bays`` bays per side, ``levels`` vertical levels. Depot (pick/drop point) at the
front-left corner. The picker walks the aisle centreline; lateral and vertical
reach are not counted as travel.

Slot feasibility (one SKU per slot):
  * size L (chassis, radio units) or weight > HEAVY_KG must be on level 0;
  * size M may use level 0 or 1;
  * size S may use any level.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

HEAVY_KG = 15.0
SIZE_MAX_LEVEL = {"S": 99, "M": 1, "L": 0}

# category: (relative popularity multiplier, size class, weight mean kg, weight sd kg)
CATEGORIES = {
    "optical_transceiver": (3.0, "S", 0.1, 0.03),
    "patch_cable":         (2.5, "S", 0.3, 0.1),
    "power_supply_unit":   (1.2, "M", 2.5, 0.6),
    "line_card":           (1.0, "M", 4.0, 1.0),
    "small_cell":          (0.7, "M", 6.0, 1.5),
    "antenna":             (0.6, "M", 9.0, 4.0),
    "router_chassis":      (0.4, "L", 18.0, 5.0),
    "radio_unit":          (0.5, "L", 22.0, 5.0),
}
# kits: anchor category -> partner categories (co-ordered items)
KIT_PARTNERS = {
    "router_chassis": ["power_supply_unit", "line_card", "optical_transceiver"],
    "radio_unit": ["antenna", "patch_cable", "power_supply_unit"],
    "small_cell": ["patch_cable", "power_supply_unit"],
    "line_card": ["optical_transceiver", "patch_cable"],
}


@dataclass
class Layout:
    aisles: int = 10
    bays: int = 20            # bays per side of an aisle
    levels: int = 3
    bay_width: float = 1.4    # m along the aisle
    aisle_pitch: float = 5.4  # m between aisle centrelines (aisle + 2 rack depths)
    depot_x: float = 0.0      # depot at front cross-aisle, left of aisle 0

    @property
    def n_slots(self) -> int:
        return self.aisles * self.bays * 2 * self.levels

    @property
    def depth(self) -> float:
        return self.bays * self.bay_width

    def slots(self) -> dict:
        a, b, side, lev = np.meshgrid(np.arange(self.aisles), np.arange(self.bays),
                                      np.arange(2), np.arange(self.levels), indexing="ij")
        a, b, side, lev = (v.ravel() for v in (a, b, side, lev))
        x = self.aisle_pitch * (a + 0.5)
        y = self.bay_width * (b + 0.5)
        return {"aisle": a, "bay": b, "side": side, "level": lev, "x": x, "y": y,
                "d_depot": (x - self.depot_x) + y}


@dataclass
class GeneratorParams:
    n_sku: int = 1000
    pop_sigma: float = 1.3          # lognormal spread of base popularity
    n_kits: int = 60
    p_kit_order: float = 0.30       # share of orders that contain a kit
    extra_lines_mean: float = 1.2   # Poisson mean of lines beyond the first / beyond the kit
    n_train_orders: int = 6000      # history used to estimate velocity and affinity
    n_test_orders: int = 3000       # held-out orders for the pick-path evaluation
    max_pairs: int = 0              # keep only the top-k co-picked pairs (0 = all with count >= 2)


@dataclass
class Instance:
    layout: Layout
    params: GeneratorParams
    seed: int
    category: np.ndarray
    size: np.ndarray
    weight: np.ndarray
    max_level: np.ndarray           # highest level each SKU may occupy
    popularity: np.ndarray          # true (unobserved) line probability weights
    kits: list
    train_orders: list
    test_orders: list
    velocity: np.ndarray = field(default=None)   # picks per SKU in train_orders
    pairs: np.ndarray = field(default=None)      # (k,2) co-picked SKU pairs
    pair_w: np.ndarray = field(default=None)     # co-pick counts

    @property
    def n_sku(self) -> int:
        return len(self.size)


def generate(seed: int, layout: Layout | None = None,
             params: GeneratorParams | None = None) -> Instance:
    layout = layout or Layout()
    params = params or GeneratorParams()
    if params.n_sku > layout.n_slots:
        raise ValueError("more SKUs than slots")
    rng_sku, rng_train, rng_test = (np.random.default_rng(s) for s in
                                    np.random.SeedSequence(seed).spawn(3))
    cats = list(CATEGORIES)
    cat_mult = np.array([CATEGORIES[c][0] for c in cats])
    cat_idx = rng_sku.choice(len(cats), size=params.n_sku, p=np.full(len(cats), 1 / len(cats)))
    category = np.array([cats[i] for i in cat_idx])
    size = np.array([CATEGORIES[c][1] for c in category])
    w_mu = np.array([CATEGORIES[c][2] for c in category])
    w_sd = np.array([CATEGORIES[c][3] for c in category])
    weight = np.maximum(0.05, rng_sku.normal(w_mu, w_sd))
    max_level = np.array([SIZE_MAX_LEVEL[s] for s in size])
    max_level = np.where(weight > HEAVY_KG, 0, max_level)
    popularity = rng_sku.lognormal(0.0, params.pop_sigma, params.n_sku) * cat_mult[cat_idx]
    popularity = popularity / popularity.sum()

    # kits: anchor + 1..3 partners drawn from compatible categories
    kits = []
    by_cat = {c: np.flatnonzero(category == c) for c in cats}
    anchors = [c for c in KIT_PARTNERS if len(by_cat[c])]
    for _ in range(params.n_kits if anchors else 0):
        ac = anchors[rng_sku.integers(len(anchors))]
        anchor = int(rng_sku.choice(by_cat[ac]))
        partners = []
        for pc in KIT_PARTNERS[ac]:
            if len(by_cat[pc]) and rng_sku.random() < 0.8:
                partners.append(int(rng_sku.choice(by_cat[pc])))
        kit = sorted(set([anchor] + partners))
        if len(kit) >= 2:
            kits.append(kit)
    kit_pop = rng_sku.lognormal(0.0, 0.8, len(kits))
    kit_pop = kit_pop / kit_pop.sum()

    def draw_orders(rng, n):
        out = []
        for _ in range(n):
            lines = set()
            if kits and rng.random() < params.p_kit_order:
                lines.update(kits[rng.choice(len(kits), p=kit_pop)])
                k = rng.poisson(params.extra_lines_mean * 0.5)
            else:
                k = 1 + rng.poisson(params.extra_lines_mean)
            if k:
                lines.update(rng.choice(params.n_sku, size=min(k, params.n_sku),
                                        replace=False, p=popularity).tolist())
            out.append(np.array(sorted(lines), dtype=int))
        return out

    inst = Instance(layout, params, seed, category, size, weight, max_level, popularity,
                    kits, draw_orders(rng_train, params.n_train_orders),
                    draw_orders(rng_test, params.n_test_orders))
    estimate_stats(inst, inst.train_orders)
    return inst


def estimate_stats(inst: Instance, orders) -> None:
    """Velocity (picks) and pairwise co-pick counts from an order history."""
    vel = np.zeros(inst.n_sku)
    co: dict = {}
    for o in orders:
        vel[o] += 1
        if 1 < len(o) <= 8:
            for i, j in itertools.combinations(o.tolist(), 2):
                co[(i, j)] = co.get((i, j), 0) + 1
    inst.velocity = vel
    if co:
        keys = np.array(list(co.keys()), dtype=int)
        w = np.array(list(co.values()), dtype=float)
        keep = w >= 2  # drop one-off coincidences
        keys, w = keys[keep], w[keep]
        k = inst.params.max_pairs
        if k and len(w) > k:
            top = np.argsort(-w, kind="stable")[:k]
            keys, w = keys[top], w[top]
        inst.pairs, inst.pair_w = keys, w
    else:
        inst.pairs, inst.pair_w = np.zeros((0, 2), int), np.zeros(0)


def abc_classes(velocity: np.ndarray, a_share: float = 0.8, b_share: float = 0.15) -> np.ndarray:
    """0=A, 1=B, 2=C by cumulative share of picks (classic 80/15/5 split)."""
    order = np.argsort(-velocity, kind="stable")
    cum = np.cumsum(velocity[order]) / max(velocity.sum(), 1e-12)
    cls_sorted = np.where(cum <= a_share, 0, np.where(cum <= a_share + b_share, 1, 2))
    # the SKU that crosses the threshold belongs to the lower class boundary
    cls = np.empty_like(cls_sorted)
    cls[order] = cls_sorted
    return cls


# --------------------------------------------------------------------------- feasibility

def feasible(inst: Instance, slots: dict, assign: np.ndarray) -> bool:
    """assign[i] = slot index of SKU i. One SKU per slot, level rules respected."""
    if len(assign) != inst.n_sku or len(set(assign.tolist())) != inst.n_sku:
        return False
    if assign.min() < 0 or assign.max() >= len(slots["x"]):
        return False
    return bool(np.all(slots["level"][assign] <= inst.max_level))


def greedy_assign(inst: Instance, slots: dict, sku_order, slot_order) -> np.ndarray:
    """Give SKUs (in priority order) the best-ranked free slot they fit, while
    reserving enough low-level slots for the constrained SKUs still waiting."""
    n_lev = int(slots["level"].max()) + 1
    level = slots["level"]
    free = np.ones(len(level), dtype=bool)
    # remaining demand of SKUs whose max_level <= l, for each l
    need = np.array([(inst.max_level <= l).sum() for l in range(n_lev)])
    avail = np.array([(level <= l).sum() for l in range(n_lev)])
    assign = np.full(inst.n_sku, -1)
    ordered_slots = np.asarray(slot_order)
    start = 0
    for i in sku_order:
        ml = inst.max_level[i]
        while start < len(ordered_slots) and not free[ordered_slots[start]]:
            start += 1
        for s in ordered_slots[start:]:
            if not free[s] or level[s] > ml:
                continue
            # taking slot s (level ls) uses capacity of every bucket l >= ls;
            # it must leave room for others in buckets l < ml that i does not belong to
            ls = level[s]
            ok = True
            for l in range(ls, n_lev):
                own = 1 if ml <= l else 0
                if avail[l] - 1 < need[l] - own:
                    ok = False
                    break
            if ok:
                break
        else:
            raise RuntimeError("greedy_assign: no feasible slot (instance infeasible?)")
        assign[i] = s
        free[s] = False
        for l in range(n_lev):
            if level[s] <= l:
                avail[l] -= 1
            if ml <= l:
                need[l] -= 1
    return assign


def assign_random(inst, slots, rng) -> np.ndarray:
    return greedy_assign(inst, slots, rng.permutation(inst.n_sku), rng.permutation(len(slots["x"])))


def assign_abc(inst, slots, rng) -> np.ndarray:
    """Class-based storage: A items in the zone nearest the depot, then B, then C;
    random placement within a class zone (the textbook ABC baseline)."""
    cls = abc_classes(inst.velocity)
    sku_order = np.lexsort((rng.random(inst.n_sku), cls))
    n_a, n_b = (cls == 0).sum(), (cls == 1).sum()
    by_dist = np.argsort(slots["d_depot"], kind="stable")
    zone = np.empty(len(by_dist), dtype=int)
    zone[by_dist[:n_a]] = 0
    zone[by_dist[n_a:n_a + n_b]] = 1
    zone[by_dist[n_a + n_b:]] = 2
    slot_order = np.lexsort((rng.random(len(zone)), zone))
    return greedy_assign(inst, slots, sku_order, slot_order)


def assign_turnover(inst, slots, rng) -> np.ndarray:
    """Full-turnover (one slot per SKU => COI equals velocity ordering)."""
    sku_order = np.lexsort((rng.random(inst.n_sku), -inst.velocity))
    slot_order = np.lexsort((rng.random(len(slots["x"])), slots["d_depot"]))
    return greedy_assign(inst, slots, sku_order, slot_order)


# --------------------------------------------------------------------------- proxy objective

def proxy_objective(inst, slots, assign, lam: float) -> float:
    """sum_i v_i * d(depot, slot_i) + lam * sum_pairs w_ij * rect(slot_i, slot_j)."""
    d = float(np.dot(inst.velocity, slots["d_depot"][assign]))
    if len(inst.pairs) and lam:
        si, sj = assign[inst.pairs[:, 0]], assign[inst.pairs[:, 1]]
        rect = np.abs(slots["x"][si] - slots["x"][sj]) + np.abs(slots["y"][si] - slots["y"][sj])
        d += lam * float(np.dot(inst.pair_w, rect))
    return d


def local_search(inst, slots, assign, lam: float, rng, iters: int = 40000) -> np.ndarray:
    """Pairwise-swap descent on the proxy objective. Moves: (a) random SKU pair,
    (b) pull an affinity partner into the aisle of its partner. Only feasible,
    strictly improving swaps are accepted."""
    assign = assign.copy()
    n = inst.n_sku
    x, y, dd, lev = slots["x"], slots["y"], slots["d_depot"], slots["level"]
    v = inst.velocity
    nbr = [[] for _ in range(n)]
    for (i, j), w in zip(inst.pairs.tolist(), inst.pair_w.tolist()):
        nbr[i].append((j, w))
        nbr[j].append((i, w))
    nbr = [(np.array([p for p, _ in l], dtype=int), np.array([w for _, w in l]))
           for l in nbr]
    occupant = np.full(len(x), -1)
    occupant[assign] = np.arange(n)
    has_pairs = np.flatnonzero([len(a[0]) for a in nbr])
    slots_by_aisle = [np.flatnonzero(slots["aisle"] == a) for a in range(int(slots["aisle"].max()) + 1)]

    def aff(i, s, exclude):
        p, w = nbr[i]
        if not len(p):
            return 0.0
        mask = p != exclude
        ps = assign[p[mask]]
        return float(np.dot(w[mask], np.abs(x[s] - x[ps]) + np.abs(y[s] - y[ps])))

    for it in range(iters):
        if lam and len(has_pairs) and it % 2:
            i = int(rng.choice(has_pairs))
            k = int(nbr[i][0][rng.integers(len(nbr[i][0]))])
            cand = slots_by_aisle[int(slots["aisle"][assign[k]])]
            s_new = int(cand[rng.integers(len(cand))])
            j = int(occupant[s_new])
        else:
            i = int(rng.integers(n))
            s_new = int(rng.integers(len(x)))
            j = int(occupant[s_new])
        si = assign[i]
        if s_new == si or lev[s_new] > inst.max_level[i]:
            continue
        if j >= 0 and lev[si] > inst.max_level[j]:
            continue
        delta = v[i] * (dd[s_new] - dd[si])
        if j >= 0:
            delta += v[j] * (dd[si] - dd[s_new])
        if lam:
            delta += lam * (aff(i, s_new, j) - aff(i, si, j))
            if j >= 0:
                delta += lam * (aff(j, si, i) - aff(j, s_new, i))
        if delta < -1e-9:
            assign[i] = s_new
            occupant[s_new] = i
            occupant[si] = j
            if j >= 0:
                assign[j] = si
    return assign


def assign_heuristic(inst, slots, rng, lam: float = 0.5, iters: int = 40000) -> np.ndarray:
    return local_search(inst, slots, assign_turnover(inst, slots, rng), lam, rng, iters)


# --------------------------------------------------------------------------- MILP (small)

def assign_milp(inst, slots, lam: float = 0.5, time_limit: float = 60.0, threads: int = 1):
    """Exact MILP on the proxy objective (HiGHS). Rectilinear pair distance is
    linearised with u_ij >= +/-(X_i - X_j), v_ij >= +/-(Y_i - Y_j), where
    X_i = sum_s x_s z_is. Returns (assign, info)."""
    import highspy

    n, m = inst.n_sku, len(slots["x"])
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(time_limit))
    h.setOptionValue("threads", int(threads))
    h.setOptionValue("mip_rel_gap", 1e-6)
    inf = highspy.kHighsInf
    feas = [(i, s) for i in range(n) for s in range(m) if slots["level"][s] <= inst.max_level[i]]
    col = {}
    lp_cost, lo, hi, integ = [], [], [], []
    for i, s in feas:
        col[(i, s)] = len(lp_cost)
        lp_cost.append(inst.velocity[i] * slots["d_depot"][s])
        lo.append(0.0)
        hi.append(1.0)
        integ.append(True)
    pairs = inst.pairs if lam else np.zeros((0, 2), int)
    uv0 = len(lp_cost)
    for w in (inst.pair_w if lam else []):
        for _ in range(2):  # u, v
            lp_cost.append(lam * w)
            lo.append(0.0)
            hi.append(inf)
            integ.append(False)
    ncol = len(lp_cost)
    h.addVars(ncol, np.array(lo), np.array(hi))
    h.changeColsCost(ncol, np.arange(ncol, dtype=np.int32), np.array(lp_cost))
    idx_int = np.array([k for k in range(ncol) if integ[k]], dtype=np.int32)
    h.changeColsIntegrality(len(idx_int), idx_int,
                            np.full(len(idx_int), highspy.HighsVarType.kInteger))

    def add_row(lb, ub, cols, vals):
        h.addRow(lb, ub, len(cols), np.array(cols, dtype=np.int32), np.array(vals, dtype=float))

    by_sku = [[] for _ in range(n)]
    by_slot = [[] for _ in range(m)]
    for (i, s), c in col.items():
        by_sku[i].append((s, c))
        by_slot[s].append((i, c))
    for i in range(n):
        add_row(1.0, 1.0, [c for _, c in by_sku[i]], [1.0] * len(by_sku[i]))
    for s in range(m):
        if by_slot[s]:
            add_row(-inf, 1.0, [c for _, c in by_slot[s]], [1.0] * len(by_slot[s]))
    for k, (i, j) in enumerate(pairs.tolist()):
        for coord, off in (("x", 0), ("y", 1)):
            cvar = uv0 + 2 * k + off
            ci = [c for _, c in by_sku[i]]
            vi = [slots[coord][s] for s, _ in by_sku[i]]
            cj = [c for _, c in by_sku[j]]
            vj = [slots[coord][s] for s, _ in by_sku[j]]
            # u - X_i + X_j >= 0  and  u + X_i - X_j >= 0
            add_row(0.0, inf, [cvar] + ci + cj, [1.0] + [-a for a in vi] + vj)
            add_row(0.0, inf, [cvar] + ci + cj, [1.0] + vi + [-a for a in vj])
    h.run()
    status = h.modelStatusToString(h.getModelStatus())
    sol = np.array(h.getSolution().col_value)
    assign = np.full(n, -1)
    for (i, s), c in col.items():
        if sol[c] > 0.5:
            assign[i] = s
    info = h.getInfo()
    return assign, {"status": status, "objective": float(info.objective_function_value),
                    "bound": float(info.mip_dual_bound), "gap": float(info.mip_gap),
                    "n_binaries": len(feas), "n_pairs": int(len(pairs))}


# --------------------------------------------------------------------------- pick-path simulation

def tour_length(layout: Layout, slots: dict, picks: np.ndarray, policy: str = "s_shape") -> float:
    """Travel of one picking tour starting and ending at the depot.

    s_shape: every aisle containing a pick is traversed entirely; with an odd number
             of pick aisles the last one is entered and left from the front.
    return:  each pick aisle is entered from the front up to its farthest pick and back.
    Cross-aisle travel = 2 * (x of farthest pick aisle - depot x) for both policies.
    """
    if len(picks) == 0:
        return 0.0
    a = slots["aisle"][picks]
    yv = slots["y"][picks]
    aisles = np.unique(a)
    far_x = layout.aisle_pitch * (aisles.max() + 0.5)
    horiz = 2.0 * (far_x - layout.depot_x)
    ymax = np.array([yv[a == k].max() for k in aisles])
    if policy == "return":
        return float(horiz + 2.0 * ymax.sum())
    if policy != "s_shape":
        raise ValueError(policy)
    m = len(aisles)
    if m % 2 == 0:
        vert = m * layout.depth
    else:
        vert = (m - 1) * layout.depth + 2.0 * ymax[-1]
    return float(horiz + vert)


def evaluate(inst, slots, assign, orders=None, policy: str = "s_shape") -> float:
    """Mean travel (m) per order over held-out orders."""
    orders = inst.test_orders if orders is None else orders
    return float(np.mean([tour_length(inst.layout, slots, assign[o], policy) for o in orders]))
