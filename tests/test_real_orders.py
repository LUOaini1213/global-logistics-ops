"""Part 1b/2b: real-order data pipeline (UCI Online Retail II) and the real-data experiments."""
import datetime as dt
import hashlib
import importlib.util
import os

import numpy as np
import pytest

from glops import experiments as E
from glops import inventory as inv
from glops import online_retail as orr
from glops import real_orders as ro
from glops import slotting as sl
from glops.common import RESULTS, ROOT, read_json

H = orr.RAW_HEADER


def _row(sheet, invoice, code, qty, day, price="1.25", desc="ITEM"):
    return dict(zip(H, [sheet, invoice, code, desc, str(qty), f"{day} 10:00:00", price, "12345",
                        "United Kingdom"]))


# one row per rule, in the order the rules are applied, plus normalisation and merging
ROWS = [
    _row("0", "100001", "85123A", 6, "2010-01-04"),
    _row("0", "100001", "85123a", 2, "2010-01-04"),              # same product once upper-cased: merged
    _row("0", "100001", " 22423 ", 1, "2010-01-04"),             # spaces trimmed
    _row("0", "C100002", "85123A", -6, "2010-01-05"),            # cancellation
    _row("0", "100003", "POST", 1, "2010-01-05", "18.0"),        # postage
    _row("0", "100003", "gift_0001_20", 1, "2010-01-05", "17.0"),  # gift voucher (prefix, any case)
    _row("0", "100003", "23444", 1, "2010-01-05", "10.0"),       # 5-digit service code
    _row("0", "100004", "21232", 0, "2010-01-06", "0.0", "damaged"),   # quantity rule comes first
    _row("0", "100004", "21232", -5, "2010-01-06", "0.0", "damaged"),
    _row("0", "100005", "21212", 3, "2010-01-06", "0.0"),        # zero price
    _row("0", "100006", "", 3, "2010-01-07"),                    # missing stock code
    _row("0", "100007", "21212", 4, "2010-12-02"),               # in both sheets: first copy kept
    _row("1", "100007", "21212", 4, "2010-12-02"),               # ... second copy dropped
    _row("1", "100008", "21212", 5, "2010-12-03"),
    _row("1", "100009", "85123A", 1, "2011-02-01"),
]


def test_cleaning_rules_count_each_row_once():
    c = orr.clean(lambda: iter(ROWS))
    assert c["removed"] == {"duplicate_sheet_overlap": 1, "cancellation": 1, "missing_stock_code": 1,
                            "non_product_code": 3, "non_positive_quantity": 2, "non_positive_price": 1}
    assert c["rows_in"] == len(ROWS) and c["rows_kept"] == len(ROWS) - 9
    assert c["orders"] == {"100001": {"date": "2010-01-04", "lines": {"85123A": 8, "22423": 1}},
                           "100007": {"date": "2010-12-02", "lines": {"21212": 4}},
                           "100008": {"date": "2010-12-03", "lines": {"21212": 5}},
                           "100009": {"date": "2011-02-01", "lines": {"85123A": 1}}}
    assert c["repeat_lines_merged"] == 1 and c["codes_normalised"] == 2
    assert c["non_product_rows_by_code"] == {"23444": 1, "GIFT_0001_20": 1, "POST": 1}
    assert c["trading_days"] == ["2010-01-04", "2010-01-05", "2010-01-06", "2010-01-07",
                                 "2010-12-02", "2010-12-03", "2011-02-01"]
    assert c["overlap_invoices"] == 1


def test_sheet_overlap_copies_must_be_identical():
    rows = [dict(r) for r in ROWS]
    rows[12]["Quantity"] = "5"
    with pytest.raises(ValueError, match="disagree"):
        orr.clean(lambda: iter(rows))


@pytest.mark.parametrize("code,product", [("POST", False), ("DOT", False), ("M", False), ("BANK CHARGES", False),
                                          ("AMAZONFEE", False), ("GIFT_0001_80", False), ("23574", False),
                                          ("TEST001", False), ("85123A", True), ("DCGS0058", True),
                                          ("PADS", True), ("22423", True)])
def test_non_product_codes(code, product):
    assert orr.is_product(orr.norm_code(code)) is product


def test_top_skus_use_training_lines_only_and_break_ties_by_code():
    tr, te = "2010-03-01", "2011-03-01"
    orders = {"1": {"date": tr, "lines": {"A": 1, "B": 1, "C": 1}}, "2": {"date": tr, "lines": {"A": 1, "C": 1}},
              "3": {"date": tr, "lines": {"A": 1, "B": 1, "D": 1}}}
    orders.update({f"t{k}": {"date": te, "lines": {"D": 1}} for k in range(10)})
    top, lines = orr.top_skus(orders, n=3)
    assert top == ["A", "B", "C"] and lines["D"] == 1      # D is popular only in the held-out period


def test_demand_stats_use_trading_days_for_the_interval():
    units = np.array([5, 0, 5, 0, 0, 5, 0, 5, 0, 5, 0, 0, 5, 0], dtype=float)
    trading = np.ones(14, bool)
    trading[[4, 11]] = False                              # shop closed
    s = orr.demand_stats(units, trading, np.ones(14, bool))
    assert s["adi"] == pytest.approx(12 / 6) and s["cv2"] == pytest.approx(0.0)
    assert s["demand_days"] == 6 and s["train_units"] == 30
    assert s["mean_daily"] == pytest.approx(30 / 14) and s["sd_daily"] == pytest.approx(units.std(ddof=1))


def test_inventory_selection_rule():
    stats = [{"stock_code": c, "adi": a, "train_lines": n} for c, a, n in
             [("A", 1.0, 50), ("B", 1.2, 90), ("C", 1.32, 99), ("D", 2.5, 10), ("E", 2.0, 40), ("F", 1.9, 99),
              ("G", 3.0, 40)]]
    assert orr.select_inventory_skus(stats, n=2) == {"steady": ["B", "A"], "intermittent": ["E", "G"]}


def test_fetch_tool_cell_text_is_deterministic():
    spec = importlib.util.spec_from_file_location("fetch_or", ROOT / "tools" / "fetch_online_retail.py")
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    assert [f.cell(v) for v in (None, 13085, 2.55, 1.0, "85123A", dt.datetime(2010, 12, 1, 8, 26))] == \
        ["", "13085", "2.55", "1.0", "85123A", "2010-12-01 08:26:00"]
    assert len(f.SHA256) == 3 and all(len(h) == 64 for h in f.SHA256.values())


# ------------------------------------------------------------------ committed derived tables

def test_derived_tables_are_consistent_and_leak_free():
    meta = orr.load_meta()
    skus = orr.load_skus()
    assert [int(s["sku"]) for s in skus] == list(range(orr.N_SKU))
    key = [(-int(s["train_lines"]), s["stock_code"]) for s in skus]
    assert key == sorted(key)
    inv_tr, d_tr, tr = orr.load_orders("train")
    inv_te, d_te, te = orr.load_orders("test")
    assert max(d_tr) < orr.SPLIT_DAY <= min(d_te) and min(d_tr) >= orr.FIRST_DAY and max(d_te) <= orr.LAST_DAY
    assert len(set(inv_tr)) == len(inv_tr) and not set(inv_tr) & set(inv_te)
    for o in tr[:2000] + te[:2000]:
        assert len(o) and np.all(np.diff(o) > 0) and o[0] >= 0 and o[-1] < orr.N_SKU
    # velocity in skus.csv is exactly the training order count, never the held-out one
    v_tr = np.bincount(np.concatenate(tr), minlength=orr.N_SKU)
    v_te = np.bincount(np.concatenate(te), minlength=orr.N_SKU)
    assert v_tr.tolist() == [int(s["train_lines"]) for s in skus]
    assert v_te.tolist() == [int(s["test_lines"]) for s in skus]
    od, cov, cl = meta["orders"], meta["coverage"], meta["cleaning"]
    assert od["train_orders_kept"] == len(tr) and od["test_orders_kept"] == len(te)
    assert cov["test"]["lines_on_top_skus"] == v_te.sum() and cov["train"]["lines_on_top_skus"] == v_tr.sum()
    assert cov["test"]["share"] == pytest.approx(v_te.sum() / cov["test"]["lines"])
    assert cl["rows_in"] - sum(cl["removed"].values()) == cl["rows_kept"]
    assert sum(cl["non_product_rows_by_code"].values()) == cl["removed"]["non_product_code"]
    assert int(skus[-1]["train_lines"]) == od["top_sku_min_train_lines"] >= od["first_excluded_train_lines"]
    size = sum(p.stat().st_size for p in orr.DERIVED.iterdir())
    assert size < 5e6


def test_inventory_tables_match_selection_rule():
    meta = orr.load_meta()
    stats = orr.load_demand_stats()
    sel = orr.select_inventory_skus(stats)
    assert sel == meta["inventory_selection"]
    daily = orr.load_daily()
    assert list(daily["units"]) == sel["steady"] + sel["intermittent"]
    assert daily["dates"] == orr.calendar()
    assert daily["trading"].sum() == meta["calendar"]["trading_days"]
    closed = [d for d, t in zip(daily["dates"], daily["trading"]) if not t]
    assert set(meta["calendar"]["closed_days"]) <= set(closed)
    train = np.array([d < orr.SPLIT_DAY for d in daily["dates"]])
    by_code = {s["stock_code"]: s for s in stats}
    for code, u in daily["units"].items():
        s = orr.demand_stats(u, daily["trading"], train)
        for k in ("adi", "cv2", "mean_daily", "sd_daily"):
            assert s[k] == pytest.approx(by_code[code][k], abs=1e-6), (code, k)
        assert s["train_units"] == by_code[code]["train_units"]


@pytest.mark.skipif(not orr.RAW_CSV.exists() or bool(os.environ.get("GLOPS_NO_RAW")),
                    reason="raw Online Retail II file not downloaded (tools/fetch_online_retail.py)")
def test_derived_tables_regenerate_exactly_from_raw(tmp_path):
    spec = importlib.util.spec_from_file_location("fetch_or", ROOT / "tools" / "fetch_online_retail.py")
    f = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(f)
    assert hashlib.sha256(orr.RAW_CSV.read_bytes()).hexdigest() == f.SHA256[orr.RAW_CSV.name]
    orr.derive(orr.RAW_CSV, tmp_path)
    committed = sorted(p.name for p in orr.DERIVED.iterdir())
    assert sorted(p.name for p in tmp_path.iterdir()) == committed
    for name in committed:
        assert (tmp_path / name).read_bytes() == (orr.DERIVED / name).read_bytes(), name


# ------------------------------------------------------------------ slotting on real orders

def test_gini_and_top_share():
    assert ro.gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert ro.gini([0, 0, 0, 10]) == pytest.approx(0.75)
    assert ro.gini([1, 2, 3, 4]) == pytest.approx(0.25)
    assert ro.top_share([10, 1, 1, 1, 1], 0.2) == pytest.approx(10 / 14)


def test_reduction_ci_is_paired():
    rng = np.random.default_rng(0)
    base = rng.uniform(50, 500, 300)                      # days differ a lot ...
    idx = ro.boot_days(300, 500, 1)
    c = ro.reduction_ci(base, 0.8 * base, idx)            # ... but the ratio is the same every day
    assert c["mean"] == pytest.approx(20.0) and c["lo"] == pytest.approx(20.0) and c["hi"] == pytest.approx(20.0)
    same = ro.reduction_ci(base, base, idx)
    assert same["mean"] == same["lo"] == same["hi"] == 0.0
    noisy = ro.reduction_ci(base, base * rng.uniform(0.7, 0.9, 300), idx)
    assert noisy["lo"] < noisy["mean"] < noisy["hi"] and noisy["hi"] - noisy["lo"] < 2.0


def test_boot_days_and_size_buckets():
    a, b = ro.boot_days(30, 7, 5), ro.boot_days(30, 7, 5)
    assert a.shape == (7, 30) and np.array_equal(a, b) and a.min() >= 0 and a.max() < 30
    assert ro.size_bucket(np.array([1, 2, 3, 4, 8, 9, 20, 21, 500])).tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4]
    day, n = ro.day_index(["2011-01-05", "2011-01-04", "2011-01-05"])
    assert day.tolist() == [1, 0, 1] and n == 2
    t = ro.mean_travel_ci(np.array([10.0, 30.0]), np.array([1.0, 3.0]), ro.boot_days(2, 50, 0))
    assert t["mean"] == pytest.approx(10.0) and t["lo"] == pytest.approx(10.0)


def test_structure_statistics_on_a_toy_history():
    orders = [np.array(o) for o in ([0, 1], [0, 1], [0, 2], [3])]
    n = 4
    inst = sl.Instance(sl.Layout(aisles=2, bays=2, levels=1), sl.GeneratorParams(n_sku=n), 0,
                       np.full(n, "x"), np.full(n, "S"), np.ones(n), np.full(n, 99), np.ones(n), [],
                       orders, orders)
    sl.estimate_stats(inst, orders)
    slots = inst.layout.slots()
    st = ro.structure(inst, slots, np.arange(n))
    assert st["affinity_pairs"] == 1                       # (0,1) twice; (0,2) once is dropped
    assert st["affinity_weight_per_pick"] == pytest.approx(2 / 7)
    assert st["best_partner_rate"] == pytest.approx(4 / 7)
    assert st["affinity_order_share"] == pytest.approx(3 / 4)
    assert st["single_line_share"] == pytest.approx(1 / 4) and st["lines_per_order_mean"] == pytest.approx(7 / 4)


def test_real_instance_is_estimated_on_training_orders_only():
    inst, dates = ro.real_instance()
    assert inst.n_sku == orr.N_SKU and inst.layout.n_slots == sl.Layout().n_slots
    assert np.all(inst.max_level == ro.ANY_LEVEL)
    v_tr = np.bincount(np.concatenate(inst.train_orders), minlength=inst.n_sku)
    assert np.array_equal(inst.velocity, v_tr)
    assert len(dates) == len(inst.test_orders) and min(dates) >= orr.SPLIT_DAY
    ref = sl.Instance(inst.layout, inst.params, -1, inst.category, inst.size, inst.weight, inst.max_level,
                      inst.popularity, [], inst.train_orders, inst.test_orders)
    sl.estimate_stats(ref, inst.train_orders)
    assert np.array_equal(ref.pairs, inst.pairs) and np.array_equal(ref.pair_w, inst.pair_w)


def test_settings_are_the_part1_settings():
    summary = read_json(RESULTS / "summary.json")
    assert ro.LAMBDA == summary["slotting"]["lambda_star"]           # not retuned on real data
    assert ro.LS_ITERS == E.LS_ITERS and ro.SYN_SEEDS == E.SLOT_LARGE_SEEDS


def test_committed_real_results_are_coherent():
    R = read_json(RESULTS / "summary.json")["real_orders"]
    s = R["slotting"]
    for key in ("red_vs_random", "red_vs_abc", "red_vs_turnover"):
        for m, d in s[key].items():
            for c in d.values():
                assert c["lo"] <= c["mean"] <= c["hi"]
    t = s["travel"]
    assert t["heuristic"]["s_shape"]["mean"] < t["turnover"]["s_shape"]["mean"] < t["abc"]["s_shape"]["mean"] \
        < t["random"]["s_shape"]["mean"]
    assert sum(b["orders"] for b in s["by_size"]) == s["n_test_orders"]
    for r in R["inventory"]["skus"]:
        for k in (k for k in r if k.endswith("_model_fill")):
            assert r[k] >= R["inventory"]["target"]


# ------------------------------------------------------------------ inventory replay

def test_given_demand_equals_internal_draw_with_fixed_lead_time():
    item = inv.Item(40.0, 12.0, 400.0, 0.25, 6.0)
    mode = inv.Mode("det", 5.0, 0.0, 1.0, 100.0)
    p = {"r": np.array([220.0, 260.0]), "Q": np.array([300.0, 300.0])}
    a = inv.simulate("rQ", p, item, mode, 600, 50, 3, 7)
    path = inv.draw_demand(np.random.default_rng(7), item, (3, 650))
    b = inv.simulate("rQ", p, item, mode, 600, 50, 3, 7, demand=path)
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def test_given_demand_is_used_and_validated():
    item = inv.Item(999.0, 50.0, 100.0, 0.25, 1.0)         # its own mean must be ignored
    mode = inv.Mode("d", 2.0, 0.0, 0.0, 0.0)
    S = np.array([40.0 * 3, 40.0 * 2])
    r = inv.simulate("base_stock", {"S": S}, item, mode, 300, 30, 2, 0, demand=np.full(330, 40.0))
    assert r["fill_rate"][0] == pytest.approx(1.0) and r["fill_rate"][1] == pytest.approx(0.0, abs=1e-12)
    assert np.all(r["final_balance_error"] < 1e-9) and np.all(r["pipeline_error"] < 1e-9)
    with pytest.raises(ValueError):
        inv.simulate("base_stock", {"S": S}, item, mode, 300, 30, 2, 0, demand=np.full(330, -1.0))
    with pytest.raises(ValueError):
        inv.simulate("base_stock", {"S": S}, item, mode, 300, 30, 2, 0, demand=np.full(329, 40.0))


def test_replay_measures_only_after_the_warmup():
    """Hand-worked: r = 0, Q = 10, lead time 1 day, start full (10 units). Warm-up days demand
    5 + 5 (both filled); then 10 a day: every one of the four measured days is short 10."""
    item = inv.Item(1.0, 1.0, 100.0, 0.25, 1.0)
    mode = inv.Mode("d", 1.0, 0.0, 0.0, 0.0)
    path = np.array([5.0, 5.0, 10.0, 10.0, 10.0, 10.0])
    f = ro.replay(item, mode, 0.0, 10.0, path, warmup=2, reps=3)
    assert f.shape == (3,) and np.allclose(f, 0.0)
    assert ro.replay(item, mode, 0.0, 10.0, path, warmup=0, reps=1)[0] == pytest.approx(0.2)


def test_week_block_paths_keep_whole_weeks():
    units = np.tile(np.array([3.0, 4.0, 5.0, 6.0, 0.0, 8.0, 9.0]), 52) + np.repeat(np.arange(52) * 10.0, 7)
    units[4::7] = 0.0                                     # the closed day of every week
    p = ro.week_block_paths(units, 100, 6, np.random.default_rng(0))
    assert p.shape == (6, 100)
    assert np.all(p[:, 4::7] == 0) and np.all(np.delete(p, np.s_[4::7], axis=1) > 0)
    blocks = {tuple(b) for b in units.reshape(52, 7)}
    assert all(tuple(row[k:k + 7]) in blocks for row in p for k in range(0, 98, 7))


def test_q_days_follow_part2_eoq():
    sc, item, modes = E._scenario()
    q = ro.q_days(item, modes)
    for name, m in modes.items():
        assert q[name] * item.demand_mean == pytest.approx(inv.eoq(item, m))
    assert q["ocean"] > q["air"] > 0


def test_tune_r_returns_the_smallest_r_meeting_the_target():
    item = inv.Item(10.0, 4.0, 100.0, 0.25, 1.0)
    mode = inv.Mode("d", 4.0, 0.0, 0.0, 0.0)
    sim = {"n_days": 1500, "warmup": 100, "reps": 4, "seed": 5}
    Q = 30.0
    t = ro.tune_r(item, mode, Q, sim, "normal")
    mu, sig = inv.protection_sigma(item, mode)
    assert t["model_fill"] >= ro.FILL_TARGET and t["r"] == pytest.approx(mu + t["z"] * sig)
    below = inv.simulate("rQ", {"r": np.array([mu + (t["z"] - ro.Z_STEP) * sig]), "Q": np.array([Q])}, item, mode,
                         sim["n_days"], sim["warmup"], sim["reps"], sim["seed"], allow_negative=True)
    assert below["fill_rate"].mean() < ro.FILL_TARGET
    assert -1.0 < t["z"] < 3.0
    with pytest.raises(ValueError):
        ro.tune_r(item, mode, Q, sim, "poisson")


def test_season_share():
    dates = ["2010-01-15", "2010-09-15", "2010-12-15", "2010-06-15"]
    assert ro.season_share(np.array([1.0, 2.0, 3.0, 4.0]), dates) == pytest.approx(0.5)
