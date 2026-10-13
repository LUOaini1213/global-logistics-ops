"""Part 3 - landed cost by Incoterms 2020 rule and transport mode.

Inputs live in config/:
  incoterms_2020.json        who pays which cost element under each rule (+ sources)
  rates_illustrative.json    ILLUSTRATIVE logistics rates (not quotes)
  duty_regimes_example.json  example import-tax regimes (SG checked; others placeholders)

Pricing model (documented assumption, not part of Incoterms):
  * the seller's EXW price already contains export packaging (A8 of every rule);
  * every logistics element the seller pays under a rule is re-invoiced to the buyer
    inside the goods price with a markup ``seller_markup`` (0 = pass-through);
  * under DDP the seller is importer of record and passes duty and import tax at cost.
    Whether import tax is a real cost depends on who can recover it (flags below).
Not tax or legal advice.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .common import CONFIG, read_json

ELEMENT_IDS = [e["id"] for e in read_json(CONFIG / "incoterms_2020.json")["elements"]]
LOGISTICS = [e for e in ELEMENT_IDS if e not in
             ("export_packaging", "cargo_insurance", "import_duty", "import_tax")]
PRE_BORDER_FOB = ["loading_at_seller", "export_clearance", "pre_carriage",
                  "unloading_export_terminal", "loading_main_carriage"]
PRE_BORDER_CIF = PRE_BORDER_FOB + ["main_carriage", "cargo_insurance"]
SEA_ONLY_MODES = ("sea_fcl_40", "sea_lcl")


@dataclass
class Shipment:
    name: str
    mode: str                 # sea_fcl_40 | sea_lcl | air
    goods_value: float        # seller's EXW price incl. export packaging, USD
    units: int
    gross_kg: float
    cbm: float
    regime: str
    containers: int = 1       # FCL only
    lane: str = ""


@dataclass
class Options:
    seller_markup: float = 0.0
    default_insurance_clause: str = "A"   # cover chosen by the risk bearer when not obliged
    buyer_recovers_import_tax: bool = True
    seller_recovers_import_tax_ddp: bool = False
    c_rules_carriage_includes_dest_thc: bool = False
    cpt_cip_named_place_is_buyer_premises: bool = False
    insurance_rates: dict = field(default_factory=dict)  # override clause rates (tests)


def load_rules():
    return read_json(CONFIG / "incoterms_2020.json")["rules"]


def load_rates():
    return read_json(CONFIG / "rates_illustrative.json")


def load_regimes():
    return read_json(CONFIG / "duty_regimes_example.json")["regimes"]


def chargeable(sh: Shipment, rates: dict) -> dict:
    conv = rates["conventions"]
    wm = max(sh.cbm, sh.gross_kg / conv["lcl_wm_kg_per_cbm"])
    vol_kg = sh.cbm * 1e6 / conv["air_volumetric_divisor_cm3_per_kg"]
    return {"wm": wm, "air_kg": max(sh.gross_kg, vol_kg)}


def element_costs(sh: Shipment, rates: dict) -> dict:
    """Cost of every logistics element for the shipment (who pays is decided later)."""
    r = rates["modes"][sh.mode]
    ch = chargeable(sh, rates)
    if sh.mode == "sea_fcl_40":
        n = sh.containers
        c = {"loading_at_seller": r["loading_at_seller"] * n,
             "export_clearance": r["export_clearance"],
             "pre_carriage": r["pre_carriage"] * n,
             "unloading_export_terminal": r["unloading_export_terminal"] * n,
             "loading_main_carriage": r["loading_main_carriage"] * n,
             "main_carriage": r["main_carriage_per_container"] * n,
             "unloading_import_terminal": r["unloading_import_terminal"] * n,
             "import_clearance": r["import_clearance"],
             "on_carriage": r["on_carriage"] * n,
             "unloading_at_destination": r["unloading_at_destination"] * n}
    elif sh.mode == "sea_lcl":
        w = ch["wm"]
        c = {"loading_at_seller": r["loading_at_seller"],
             "export_clearance": r["export_clearance"],
             "pre_carriage": r["pre_carriage_base"] + r["pre_carriage_per_wm"] * w,
             "unloading_export_terminal": r["unloading_export_terminal_per_wm"] * w,
             "loading_main_carriage": r["loading_main_carriage_per_wm"] * w,
             "main_carriage": max(r["main_carriage_min"], r["main_carriage_per_wm"] * w),
             "unloading_import_terminal": r["unloading_import_terminal_per_wm"] * w,
             "import_clearance": r["import_clearance"],
             "on_carriage": r["on_carriage_base"] + r["on_carriage_per_wm"] * w,
             "unloading_at_destination": r["unloading_at_destination"]}
    elif sh.mode == "air":
        k = ch["air_kg"]
        c = {"loading_at_seller": r["loading_at_seller"],
             "export_clearance": r["export_clearance"],
             "pre_carriage": r["pre_carriage_base"] + r["pre_carriage_per_kg"] * k,
             "unloading_export_terminal": r["unloading_export_terminal_per_kg"] * k,
             "loading_main_carriage": r["loading_main_carriage_per_kg"] * k,
             "main_carriage": max(r["main_carriage_min"], r["main_carriage_per_kg"] * k),
             "unloading_import_terminal": r["unloading_import_terminal_per_kg"] * k,
             "import_clearance": r["import_clearance"],
             "on_carriage": r["on_carriage_base"] + r["on_carriage_per_kg"] * k,
             "unloading_at_destination": r["unloading_at_destination"]}
    else:
        raise ValueError(f"unknown mode {sh.mode}")
    assert set(c) == set(LOGISTICS), set(c) ^ set(LOGISTICS)
    return {k: float(v) for k, v in c.items()}


def resolve_payers(rule: str, opts: Options, rules=None) -> dict:
    """Element -> 'S' | 'B' after applying the contract-dependent switches."""
    rules = rules or load_rules()
    spec = rules[rule]
    payer = dict(spec["payer"])
    if payer["cargo_insurance"] == "-":           # no obligation: risk bearer insures
        payer["cargo_insurance"] = spec["risk"]
    if rule in ("CFR", "CIF", "CPT", "CIP") and opts.c_rules_carriage_includes_dest_thc:
        payer["unloading_import_terminal"] = "S"
    if rule in ("CPT", "CIP") and opts.cpt_cip_named_place_is_buyer_premises:
        payer["unloading_import_terminal"] = "S"
        payer["on_carriage"] = "S"
    return payer


def validate_rule_for_mode(rule: str, sh: Shipment, rules=None) -> list:
    """Raise for invalid combinations, return warnings for discouraged ones."""
    rules = rules or load_rules()
    if rules[rule]["modes"] == "sea" and sh.mode not in SEA_ONLY_MODES:
        raise ValueError(f"{rule} is a sea/inland-waterway rule; not valid for {sh.mode}")
    warn = []
    if rules[rule]["modes"] == "sea" and sh.mode in SEA_ONLY_MODES:
        warn.append(f"{rule} with containerised cargo: goods are usually handed over at a "
                    "terminal before loading; FCA/CPT/CIP are generally more appropriate.")
    return warn


def landed_cost(sh: Shipment, rule: str, opts: Options | None = None,
                rates=None, regimes=None, rules=None) -> dict:
    opts = opts or Options()
    rates = rates or load_rates()
    regimes = regimes or load_regimes()
    rules = rules or load_rules()
    warnings = validate_rule_for_mode(rule, sh, rules)
    payer = resolve_payers(rule, opts, rules)
    cost = element_costs(sh, rates)
    m = opts.seller_markup
    # amount billed for each logistics element (seller-paid ones carry the markup)
    billed = {e: cost[e] * (1 + m if payer[e] == "S" else 1.0) for e in LOGISTICS}

    clause = rules[rule].get("insurance_clause", opts.default_insurance_clause)
    ins_rate = opts.insurance_rates.get(clause, rates["insurance"][f"rate_icc_{clause.lower()}"])
    share = rates["insurance"]["insured_share"]
    pre_ins = sh.goods_value + sum(billed[e] for e in PRE_BORDER_CIF if e != "cargo_insurance")
    premium = ins_rate * share * pre_ins / (1 - ins_rate * share)   # insured value includes premium
    cost["cargo_insurance"] = premium
    billed["cargo_insurance"] = premium * (1 + m if payer["cargo_insurance"] == "S" else 1.0)

    reg = regimes[sh.regime]
    basis = PRE_BORDER_CIF if reg["valuation_basis"] == "CIF" else PRE_BORDER_FOB
    customs_value = sh.goods_value + sum(billed[e] for e in basis)
    duty = reg["duty_rate"] * customs_value
    tax = reg["tax_rate"] * (customs_value + duty)
    cost["import_duty"], cost["import_tax"] = duty, tax
    billed["import_duty"], billed["import_tax"] = duty, tax   # never marked up

    seller_elems = [e for e in billed if payer.get(e) == "S" and e != "export_packaging"]
    buyer_elems = [e for e in billed if payer.get(e) == "B"]
    price = sh.goods_value + sum(billed[e] for e in seller_elems)
    if rule == "DDP" and opts.seller_recovers_import_tax_ddp:
        price -= tax                       # seller reclaims it, so it is not priced in
    buyer_paid = sum(billed[e] for e in buyer_elems)
    gross = price + buyer_paid
    recoverable = tax if (payer["import_tax"] == "B" and opts.buyer_recovers_import_tax) else 0.0
    net = gross - recoverable
    seller_cost = sum(cost[e] for e in seller_elems)
    if rule == "DDP" and opts.seller_recovers_import_tax_ddp:
        seller_cost -= tax
    return {
        "rule": rule, "shipment": sh.name, "mode": sh.mode, "regime": sh.regime,
        "insurance_clause": clause, "payer": payer, "cost": cost, "billed": billed,
        "price": price, "buyer_paid_separately": buyer_paid, "customs_value": customs_value,
        "duty": duty, "tax": tax, "landed_gross": gross, "landed_net": net,
        "landed_net_per_unit": net / sh.units, "seller_logistics_cost": seller_cost,
        "seller_margin_on_logistics": price - sh.goods_value - seller_cost,
        "warnings": warnings,
    }


def example_shipments() -> list:
    """Example shipments of network equipment (synthetic, no real parties)."""
    return [
        Shipment("FCL radio units", "sea_fcl_40", goods_value=980_000, units=280,
                 gross_kg=7_000, cbm=52, regime="SG", containers=1,
                 lane="South China port -> Singapore"),
        Shipment("LCL optics + PSUs", "sea_lcl", goods_value=140_000, units=1_400,
                 gross_kg=900, cbm=5.5, regime="SG", lane="South China port -> Singapore"),
        Shipment("Air optics + PSUs", "air", goods_value=140_000, units=1_400,
                 gross_kg=900, cbm=5.5, regime="SG", lane="South China airport -> Singapore"),
        Shipment("Air transceivers to EU", "air", goods_value=250_000, units=2_500,
                 gross_kg=160, cbm=1.1, regime="EU_STYLE_EXAMPLE",
                 lane="Singapore -> North-West Europe"),
        Shipment("FCL small cells to US", "sea_fcl_40", goods_value=600_000, units=1_500,
                 gross_kg=9_500, cbm=58, regime="US_STYLE_EXAMPLE", containers=1,
                 lane="Singapore -> US West Coast"),
    ]


ALL_RULES = ["EXW", "FCA", "FCA_TERMINAL", "FAS", "FOB", "CFR", "CIF", "CPT", "CIP",
             "DAP", "DPU", "DDP"]


def rules_for(sh: Shipment, rules=None) -> list:
    rules = rules or load_rules()
    return [r for r in ALL_RULES if not (rules[r]["modes"] == "sea" and sh.mode not in SEA_ONLY_MODES)]
