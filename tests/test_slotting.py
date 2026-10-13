import itertools

import numpy as np
import pytest

from glops import slotting as sl

TINY_LAYOUT = sl.Layout(aisles=2, bays=1, levels=2)          # 8 slots
TINY_PARAMS = sl.GeneratorParams(n_sku=5, n_kits=3, n_train_orders=400, n_test_orders=200)
SMALL_LAYOUT = sl.Layout(aisles=3, bays=3, levels=2)          # 36 slots
SMALL_PARAMS = sl.GeneratorParams(n_sku=30, n_kits=5, n_train_orders=800, n_test_orders=300, max_pairs=20)


def brute_force(inst, slots, lam):
    best, arg = np.inf, None
    for perm in itertools.permutations(range(len(slots["x"])), inst.n_sku):
        a = np.array(perm)
        if np.any(slots["level"][a] > inst.max_level):
            continue
        v = sl.proxy_objective(inst, slots, a, lam)
        if v < best:
            best, arg = v, a
    return best, arg


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_milp_matches_brute_force(seed):
    inst = sl.generate(seed, TINY_LAYOUT, TINY_PARAMS)
    slots = TINY_LAYOUT.slots()
    best, _ = brute_force(inst, slots, 0.5)
    a, info = sl.assign_milp(inst, slots, lam=0.5, time_limit=30)
    assert info["status"] == "Optimal"
    assert sl.feasible(inst, slots, a)
    assert sl.proxy_objective(inst, slots, a, 0.5) == pytest.approx(best, rel=1e-9)
    assert info["objective"] == pytest.approx(best, rel=1e-6)


def test_all_methods_feasible_and_heuristic_improves_proxy():
    inst = sl.generate(4, SMALL_LAYOUT, SMALL_PARAMS)
    slots = SMALL_LAYOUT.slots()
    rng = np.random.default_rng(0)
    start = sl.assign_turnover(inst, slots, rng)
    for a in (sl.assign_random(inst, slots, rng), sl.assign_abc(inst, slots, rng), start):
        assert sl.feasible(inst, slots, a)
    h = sl.local_search(inst, slots, start, 0.5, rng, 5000)
    assert sl.feasible(inst, slots, h)
    assert sl.proxy_objective(inst, slots, h, 0.5) <= sl.proxy_objective(inst, slots, start, 0.5)
    m, info = sl.assign_milp(inst, slots, lam=0.5, time_limit=60)
    assert sl.feasible(inst, slots, m)
    if info["status"] == "Optimal":
        assert sl.proxy_objective(inst, slots, m, 0.5) <= sl.proxy_objective(inst, slots, h, 0.5) + 1e-6


def test_level_constraints_enforced_on_tight_instance():
    """Exactly as many floor slots as floor-only SKUs: the greedy must still find room."""
    lay = sl.Layout(aisles=1, bays=2, levels=2)                # 4 floor + 4 upper slots
    par = sl.GeneratorParams(n_sku=8, n_kits=1, n_train_orders=50, n_test_orders=10)
    inst = sl.generate(0, lay, par)
    inst.max_level[:] = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    slots = lay.slots()
    rng = np.random.default_rng(1)
    for f in (sl.assign_random, sl.assign_abc, sl.assign_turnover):
        for _ in range(20):
            a = f(inst, slots, rng)
            assert sl.feasible(inst, slots, a)
    inst.max_level[:] = 0                                      # 8 floor-only SKUs, 4 floor slots
    with pytest.raises(RuntimeError):
        sl.assign_random(inst, slots, rng)


def test_feasible_rejects_violations():
    inst = sl.generate(0, SMALL_LAYOUT, SMALL_PARAMS)
    slots = SMALL_LAYOUT.slots()
    a = sl.assign_turnover(inst, slots, np.random.default_rng(0))
    dup = a.copy()
    dup[1] = dup[0]
    assert not sl.feasible(inst, slots, dup)
    heavy = np.flatnonzero(inst.max_level == 0)[0]
    high = np.flatnonzero(slots["level"] == 1)
    free_high = [s for s in high if s not in set(a.tolist())]
    bad = a.copy()
    bad[heavy] = free_high[0]
    assert not sl.feasible(inst, slots, bad)


def test_tour_length_geometry():
    lay = sl.Layout(aisles=4, bays=10, levels=1, bay_width=1.0, aisle_pitch=4.0)
    slots = lay.slots()

    def slot(a, b):
        return int(np.flatnonzero((slots["aisle"] == a) & (slots["bay"] == b) & (slots["side"] == 0))[0])

    # one pick in aisle 0, bay 3 (y = 3.5): out and back
    p = np.array([slot(0, 3)])
    assert sl.tour_length(lay, slots, p, "s_shape") == pytest.approx(2 * 2.0 + 2 * 3.5)
    assert sl.tour_length(lay, slots, p, "return") == pytest.approx(2 * 2.0 + 2 * 3.5)
    # two pick aisles (0 and 2): S-shape traverses both fully
    p = np.array([slot(0, 3), slot(2, 1)])
    assert sl.tour_length(lay, slots, p, "s_shape") == pytest.approx(2 * 10.0 + 2 * 10.0)
    assert sl.tour_length(lay, slots, p, "return") == pytest.approx(2 * 10.0 + 2 * 3.5 + 2 * 1.5)
    # three pick aisles: two traversals + return in the last one
    p = np.array([slot(0, 3), slot(1, 9), slot(3, 4)])
    assert sl.tour_length(lay, slots, p, "s_shape") == pytest.approx(2 * 14.0 + 2 * 10.0 + 2 * 4.5)
    assert sl.tour_length(lay, slots, np.array([], dtype=int), "s_shape") == 0.0


def test_tour_lower_bound_holds():
    inst = sl.generate(2, SMALL_LAYOUT, SMALL_PARAMS)
    slots = SMALL_LAYOUT.slots()
    a = sl.assign_random(inst, slots, np.random.default_rng(3))
    for o in inst.test_orders[:200]:
        s = a[o]
        lb = 2 * slots["d_depot"][s].max()
        for pol in ("s_shape", "return"):
            assert sl.tour_length(inst.layout, slots, s, pol) >= lb - 1e-9


def test_abc_classes_shares():
    v = np.array([50, 30, 10, 5, 3, 2], dtype=float)
    cls = sl.abc_classes(v)
    assert cls.tolist() == [0, 0, 1, 1, 2, 2]


def test_generator_is_seed_deterministic_and_train_test_differ():
    a = sl.generate(7, SMALL_LAYOUT, SMALL_PARAMS)
    b = sl.generate(7, SMALL_LAYOUT, SMALL_PARAMS)
    assert np.array_equal(a.velocity, b.velocity)
    assert all(np.array_equal(x, y) for x, y in zip(a.test_orders, b.test_orders))
    assert not all(np.array_equal(x, y) for x, y in zip(a.train_orders, a.test_orders))
    c = sl.generate(8, SMALL_LAYOUT, SMALL_PARAMS)
    assert not np.array_equal(a.velocity, c.velocity)


def test_generator_level_rules():
    inst = sl.generate(3)
    big_or_heavy = (inst.size == "L") | (inst.weight > sl.HEAVY_KG)
    assert np.all(inst.max_level[big_or_heavy] == 0)
    assert np.all(inst.max_level[(inst.size == "M") & ~big_or_heavy] == 1)
    assert big_or_heavy.sum() > 0 and (inst.size == "S").sum() > 0
