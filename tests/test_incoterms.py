"""Incoterms 2020 cost allocation - table driven.

EXPECTED below was typed by hand from the public sources cited in
config/incoterms_2020.json (Wikipedia's Incoterms 2020 cost table, cross-checked with
ICC's 2020 change notes and the Gebrueder Weiss summary), independently of the config
file, so a wrong cell in the config fails here.
Columns: load, exp_clr, pre, unl_orig, load_main, main, unl_dest, imp_clr, duty, tax, on, unl_final
Insurance and packaging are tested separately.
"""
import pytest

from glops import landed as lc

COLS = ["loading_at_seller", "export_clearance", "pre_carriage", "unloading_export_terminal",
        "loading_main_carriage", "main_carriage", "unloading_import_terminal", "import_clearance",
        "import_duty", "import_tax", "on_carriage", "unloading_at_destination"]
EXPECTED = {
    #               load exp pre unlO ldM main unlD impC duty tax  on  unlF
    "EXW":          "B    B   B   B    B   B    B    B    B    B    B   B",
    "FCA":          "S    S   B   B    B   B    B    B    B    B    B   B",
    "FCA_TERMINAL": "S    S   S   B    B   B    B    B    B    B    B   B",
    "FAS":          "S    S   S   S    B   B    B    B    B    B    B   B",
    "FOB":          "S    S   S   S    S   B    B    B    B    B    B   B",
    "CFR":          "S    S   S   S    S   S    B    B    B    B    B   B",
    "CIF":          "S    S   S   S    S   S    B    B    B    B    B   B",
    "CPT":          "S    S   S   S    S   S    B    B    B    B    B   B",
    "CIP":          "S    S   S   S    S   S    B    B    B    B    B   B",
    "DAP":          "S    S   S   S    S   S    S    B    B    B    S   B",
    "DPU":          "S    S   S   S    S   S    S    B    B    B    S   S",
    "DDP":          "S    S   S   S    S   S    S    S    S    S    S   B",
}


@pytest.mark.parametrize("rule", list(EXPECTED))
@pytest.mark.parametrize("col", range(len(COLS)))
def test_payer_cell(rule, col):
    payer = lc.resolve_payers(rule, lc.Options())
    assert payer[COLS[col]] == EXPECTED[rule].split()[col], (rule, COLS[col])


@pytest.mark.parametrize("rule", list(EXPECTED))
def test_packaging_always_seller(rule):
    assert lc.resolve_payers(rule, lc.Options())["export_packaging"] == "S"


INSURANCE = {"CIF": ("S", "C"), "CIP": ("S", "A")}


@pytest.mark.parametrize("rule", list(EXPECTED))
def test_insurance_obligation_only_cif_cip(rule):
    spec = lc.load_rules()[rule]
    if rule in INSURANCE:
        assert spec["payer"]["cargo_insurance"] == INSURANCE[rule][0]
        assert spec["insurance_clause"] == INSURANCE[rule][1]
    else:
        assert spec["payer"]["cargo_insurance"] == "-"      # no Incoterms obligation
        expected_risk = "S" if rule.startswith("D") else "B"
        assert spec["risk"] == expected_risk


def test_sea_only_rules_and_modes():
    rules = lc.load_rules()
    assert {r for r, v in rules.items() if v["modes"] == "sea"} == {"FAS", "FOB", "CFR", "CIF"}
    air = lc.example_shipments()[2]
    assert air.mode == "air"
    for r in ("FAS", "FOB", "CFR", "CIF"):
        with pytest.raises(ValueError):
            lc.landed_cost(air, r)
    assert "FOB" not in lc.rules_for(air) and "FCA" in lc.rules_for(air)


def test_contract_dependent_switches():
    o = lc.Options(c_rules_carriage_includes_dest_thc=True)
    for r in ("CFR", "CIF", "CPT", "CIP"):
        assert lc.resolve_payers(r, o)["unloading_import_terminal"] == "S"
    assert lc.resolve_payers("FOB", o)["unloading_import_terminal"] == "B"
    o = lc.Options(cpt_cip_named_place_is_buyer_premises=True)
    assert lc.resolve_payers("CPT", o)["on_carriage"] == "S"
    assert lc.resolve_payers("CFR", o)["on_carriage"] == "B"


# ----------------------------------------------------------------- landed cost arithmetic

SHIPS = lc.example_shipments()


@pytest.mark.parametrize("sh", SHIPS, ids=lambda s: s.name)
def test_conservation_without_markup(sh):
    """Pass-through pricing: the rule only moves cost between invoices (except DDP tax)."""
    o = lc.Options(seller_markup=0.0, insurance_rates={"A": 0.0015, "C": 0.0015})
    vals = [lc.landed_cost(sh, r, o)["landed_net"] for r in lc.rules_for(sh) if r != "DDP"]
    assert max(vals) - min(vals) < 1e-6


@pytest.mark.parametrize("sh", SHIPS, ids=lambda s: s.name)
def test_exw_and_ddp_extremes(sh):
    o = lc.Options(seller_markup=0.1)
    exw = lc.landed_cost(sh, "EXW", o)
    assert exw["price"] == pytest.approx(sh.goods_value)
    assert exw["seller_logistics_cost"] == 0
    ddp = lc.landed_cost(sh, "DDP", o)
    # buyer pays only unloading at the final destination separately
    assert ddp["buyer_paid_separately"] == pytest.approx(ddp["cost"]["unloading_at_destination"])
    assert ddp["landed_gross"] == pytest.approx(ddp["landed_net"])   # no tax for the buyer to reclaim


def test_markup_lands_on_seller_elements_only():
    sh = SHIPS[0]
    a = lc.landed_cost(sh, "CIF", lc.Options(seller_markup=0.0))
    b = lc.landed_cost(sh, "CIF", lc.Options(seller_markup=0.10))
    assert b["seller_margin_on_logistics"] == pytest.approx(0.10 * b["seller_logistics_cost"], rel=1e-9)
    assert a["seller_margin_on_logistics"] == pytest.approx(0.0, abs=1e-9)
    assert b["landed_net"] > a["landed_net"]


def test_singapore_gst_by_hand():
    """FCA FCL into SG: customs value = goods + pre-border logistics + main freight + insurance;
    duty 0; GST 9% on (CV + duty)."""
    sh = SHIPS[0]
    r = lc.landed_cost(sh, "FCA", lc.Options())
    c = r["cost"]
    cv = sh.goods_value + sum(c[e] for e in ("loading_at_seller", "export_clearance", "pre_carriage",
                                             "unloading_export_terminal", "loading_main_carriage",
                                             "main_carriage", "cargo_insurance"))
    assert r["customs_value"] == pytest.approx(cv)
    assert r["duty"] == 0.0
    assert r["tax"] == pytest.approx(0.09 * cv)
    assert r["landed_net"] == pytest.approx(r["landed_gross"] - r["tax"])


def test_insurance_premium_110pct_including_premium():
    sh = SHIPS[0]
    r = lc.landed_cost(sh, "CIP", lc.Options())
    rate = lc.load_rates()["insurance"]["rate_icc_a"]
    insured = 1.10 * (r["customs_value"])          # CIF value incl. the premium itself
    assert r["cost"]["cargo_insurance"] == pytest.approx(rate * insured, rel=1e-9)


def test_fob_valuation_excludes_international_freight():
    sh = SHIPS[4]
    assert sh.regime == "US_STYLE_EXAMPLE"
    r = lc.landed_cost(sh, "FCA", lc.Options())
    c = r["cost"]
    cv = sh.goods_value + sum(c[e] for e in ("loading_at_seller", "export_clearance", "pre_carriage",
                                             "unloading_export_terminal", "loading_main_carriage"))
    assert r["customs_value"] == pytest.approx(cv)
    assert r["duty"] == pytest.approx(0.025 * cv)
    assert r["tax"] == 0


def test_ddp_tax_recovery_flag():
    sh = SHIPS[0]
    no = lc.landed_cost(sh, "DDP", lc.Options(seller_recovers_import_tax_ddp=False))
    yes = lc.landed_cost(sh, "DDP", lc.Options(seller_recovers_import_tax_ddp=True))
    assert no["landed_net"] - yes["landed_net"] == pytest.approx(no["tax"])


def test_chargeable_weight_conventions():
    rates = lc.load_rates()
    s = lc.Shipment("t", "air", 1000, 1, gross_kg=100, cbm=1.2, regime="SG")
    assert lc.chargeable(s, rates)["air_kg"] == pytest.approx(200.0)    # 1.2 m3 / 6000 cm3/kg
    s = lc.Shipment("t", "sea_lcl", 1000, 1, gross_kg=2500, cbm=1.2, regime="SG")
    assert lc.chargeable(s, rates)["wm"] == pytest.approx(2.5)          # weight governs


def test_vat_base_includes_duty_eu_style():
    sh = SHIPS[3]
    assert sh.regime == "EU_STYLE_EXAMPLE"
    r = lc.landed_cost(sh, "CPT", lc.Options())
    assert r["duty"] == pytest.approx(0.02 * r["customs_value"])
    assert r["tax"] == pytest.approx(0.21 * (r["customs_value"] + r["duty"]))
