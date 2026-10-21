"""Hand-written mutation test: break the code/config on purpose, check the tests notice.

    python tools/mutation_test.py            # writes results/mutation_report.json

Each mutant is an exact, single-occurrence text replacement in one file. The file is
restored in a finally block and a SHA-256 check confirms the tree is unchanged at the end.
A mutant is CAUGHT if pytest fails (exit code != 0) within the time limit.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MUTANTS = [
    # ---- Incoterms configuration
    ("config/incoterms_2020.json", "FOB: origin loading charged to buyer",
     '"unloading_export_terminal": "S", "loading_main_carriage": "S", "main_carriage": "B", "cargo_insurance": "-", "unloading_import_terminal": "B", "import_clearance": "B", "import_duty": "B", "import_tax": "B", "on_carriage": "B", "unloading_at_destination": "B"},\n            "risk": "B"},\n    "CFR"',
     '"unloading_export_terminal": "S", "loading_main_carriage": "B", "main_carriage": "B", "cargo_insurance": "-", "unloading_import_terminal": "B", "import_clearance": "B", "import_duty": "B", "import_tax": "B", "on_carriage": "B", "unloading_at_destination": "B"},\n            "risk": "B"},\n    "CFR"'),
    ("config/incoterms_2020.json", "DAP: seller unloads at destination",
     '"on_carriage": "S", "unloading_at_destination": "B"},\n            "risk": "S"},\n    "DPU"',
     '"on_carriage": "S", "unloading_at_destination": "S"},\n            "risk": "S"},\n    "DPU"'),
    ("config/incoterms_2020.json", "CIF insured at ICC (A) instead of (C)",
     '"risk": "B", "insurance_clause": "C"}', '"risk": "B", "insurance_clause": "A"}'),
    # ---- landed cost arithmetic
    ("glops/landed.py", "customs value omits insurance",
     'PRE_BORDER_CIF = PRE_BORDER_FOB + ["main_carriage", "cargo_insurance"]',
     'PRE_BORDER_CIF = PRE_BORDER_FOB + ["main_carriage"]'),
    ("glops/landed.py", "VAT/GST base forgets duty",
     'tax = reg["tax_rate"] * (customs_value + duty)', 'tax = reg["tax_rate"] * customs_value'),
    ("glops/landed.py", "insurance premium not grossed up",
     'premium = ins_rate * share * pre_ins / (1 - ins_rate * share)', 'premium = ins_rate * share * pre_ins'),
    ("glops/landed.py", "markup also applied to buyer-paid elements",
     'billed = {e: cost[e] * (1 + m if payer[e] == "S" else 1.0) for e in LOGISTICS}',
     'billed = {e: cost[e] * (1 + m) for e in LOGISTICS}'),
    ("glops/landed.py", "air volumetric divisor ignored",
     'vol_kg = sh.cbm * 1e6 / conv["air_volumetric_divisor_cm3_per_kg"]', 'vol_kg = sh.cbm * 1e6 / 5000'),
    # ---- inventory simulator
    # (round 1 used a re-ordering of two additions here: an EQUIVALENT mutant, replaced)
    ("glops/inventory.py", "receipts usable before the day's demand",
     "        short = np.maximum(0.0, d - np.maximum(ni, 0.0))\n        ni = ni - d\n        rec = arrivals[:, t]\n        ni = ni + rec\n",
     "        rec = arrivals[:, t]\n        ni = ni + rec\n        short = np.maximum(0.0, d - np.maximum(ni, 0.0))\n        ni = ni - d\n"),
    ("glops/inventory.py", "shortage computed from net (not on-hand) inventory",
     "short = np.maximum(0.0, d - np.maximum(ni, 0.0))", "short = np.maximum(0.0, d - ni)"),
    ("glops/inventory.py", "(r,Q) misses the +1 batch",
     'n = np.where(ip <= P["r"], np.floor((P["r"] - ip) / P["Q"]) + 1, 0.0)',
     'n = np.where(ip <= P["r"], np.floor((P["r"] - ip) / P["Q"]), 0.0)'),
    ("glops/inventory.py", "on-order not reduced by receipts",
     "        on_order = on_order - rec\n", "        on_order = on_order - 0 * rec\n"),
    ("glops/inventory.py", "analytic fill rate drops the E[(D_L-S)+] term",
     "short = exp_pos(L + 1) - exp_pos(L)", "short = exp_pos(L + 1)"),
    ("glops/inventory.py", "holding cost on a 360-day year",
     "return self.unit_value * self.holding_rate / 365.0", "return self.unit_value * self.holding_rate / 360.0"),
    ("glops/inventory.py", "lead time off by one day",
     "np.add.at(arrivals, (idx, t + lts[idx, t]), q[placed])",
     "np.add.at(arrivals, (idx, t + lts[idx, t] + 1), q[placed])"),
    # ---- slotting
    ("glops/slotting.py", "S-shape odd case uses the first aisle's depth",
     "vert = (m - 1) * layout.depth + 2.0 * ymax[-1]", "vert = (m - 1) * layout.depth + 2.0 * ymax[0]"),
    ("glops/slotting.py", "cross-aisle travel to nearest instead of farthest aisle",
     "far_x = layout.aisle_pitch * (aisles.max() + 0.5)", "far_x = layout.aisle_pitch * (aisles.min() + 0.5)"),
    ("glops/slotting.py", "greedy ignores low-level reservation",
     "                if avail[l] - 1 < need[l] - own:", "                if False:"),
    ("glops/slotting.py", "MILP pair distance linearised on one side only",
     "            add_row(0.0, inf, [cvar] + ci + cj, [1.0] + vi + [-a for a in vj])\n", ""),
    ("glops/slotting.py", "local search accepts worsening swaps",
     "        if delta < -1e-9:", "        if delta > 1e-9:"),
    ("glops/slotting.py", "level rule lets L-size items go to level 1",
     'SIZE_MAX_LEVEL = {"S": 99, "M": 1, "L": 0}', 'SIZE_MAX_LEVEL = {"S": 99, "M": 1, "L": 1}'),
    # ---- reverse logistics
    ("glops/reverse.py", "failures use yield instead of 1 - yield",
     '[1.0] * len(fs) + [-(1 - rv["yield"][k[2]]) for k in xs], 0.0, 0.0)',
     '[1.0] * len(fs) + [-rv["yield"][k[2]] for k in xs], 0.0, 0.0)'),
    ("glops/reverse.py", "cross-border waste cost dropped from the LP",
     '            border = cb["waste_for_recycling_per_unit"] if cross else 0.0\n            for g in ("A", "B", "C"):',
     '            border = 0.0\n            for g in ("A", "B", "C"):'),
    ("glops/reverse.py", "market cap row dropped",
     '    lp.row("market", list(S.values()), [1.0] * len(S), None, float(mk["resale_demand_cap"]))\n', ""),
    ("glops/reverse.py", "cross-border recycling flag ignored",
     "            if cross and not allow_x:\n                continue\n            border = cb[\"waste_for_recycling_per_unit\"] if cross else 0.0\n            for g",
     "            border = cb[\"waste_for_recycling_per_unit\"] if cross else 0.0\n            for g"),
    # ---- real orders (Online Retail II): cleaning and derived tables
    ("glops/online_retail.py", "cancellations kept",
     '    if row["Invoice"].startswith("C"):\n        return "cancellation"',
     '    if row["Invoice"].startswith("c"):\n        return "cancellation"'),
    ("glops/online_retail.py", "duplicated sheet-overlap invoices kept",
     '    if row["sheet"] == "1" and row["Invoice"] in first_sheet_invoices:', "    if False:"),
    ("glops/online_retail.py", "stock codes not upper-cased",
     "    return code.strip().upper()", "    return code.strip()"),
    ("glops/online_retail.py", "zero-price lines kept",
     '    if float(row["Price"]) <= 0:', '    if float(row["Price"]) < 0:'),
    ("glops/online_retail.py", "'Next Day Carriage' counted as a product",
     '    "23444": "next day carriage", "23574": "packing charge",', '    "23574": "packing charge",'),
    ("glops/online_retail.py", "repeat lines overwrite instead of adding quantities",
     '        o["lines"][code] = o["lines"].get(code, 0) + int(r["Quantity"])',
     '        o["lines"][code] = int(r["Quantity"])'),
    ("glops/online_retail.py", "top SKUs ranked on training + held-out lines (leakage)",
     '    lines = Counter(c for o in orders.values() if o["date"] < SPLIT_DAY for c in o["lines"])',
     '    lines = Counter(c for o in orders.values() for c in o["lines"])'),
    ("glops/online_retail.py", "demand interval counts closed days",
     "    x = units[train & trading]", "    x = units[train]"),
    ("glops/online_retail.py", "inventory SKUs: least-ordered instead of most-ordered",
     '    key = lambda s: (-s["train_lines"], s["stock_code"])', '    key = lambda s: (s["train_lines"], s["stock_code"])'),
    # ---- real orders: experiments
    ("glops/real_orders.py", "velocity / affinity estimated on held-out orders (leakage)",
     "    sl.estimate_stats(inst, inst.train_orders)          # training orders only",
     "    sl.estimate_stats(inst, inst.test_orders)          # training orders only"),
    ("glops/real_orders.py", "bootstrap not paired (methods resampled independently)",
     "    rb = 100.0 * (1.0 - other_day[idx].sum(axis=1) / base_day[idx].sum(axis=1))",
     "    rb = 100.0 * (1.0 - other_day[idx].sum(axis=1) / base_day[idx[::-1]].sum(axis=1))"),
    ("glops/real_orders.py", "Gini off by one rank",
     "    return float(((2 * i - n - 1) * v).sum() / (n * v.sum()))",
     "    return float(((2 * i - n) * v).sum() / (n * v.sum()))"),
    ("glops/real_orders.py", "replay also scores the warm-up days",
     "                       len(path) - warmup, warmup, reps, seed, demand=path)",
     "                       len(path), 0, reps, seed, demand=path)"),
    ("glops/real_orders.py", "bootstrap resamples single days, not weeks",
     "BLOCK_DAYS = 7", "BLOCK_DAYS = 1"),
    ("glops/real_orders.py", "r search skips the fine step (not the smallest r)",
     "    if ok[0] > 0:", "    if ok[0] < 0:"),
    ("glops/slotting.py", "vectorised router uses the nearest pick aisle as the farthest",
     "    last = np.cumsum(m) - 1                          # farthest pick aisle of each order",
     "    last = np.cumsum(m) - m                          # farthest pick aisle of each order"),
    ("glops/inventory.py", "given (real) demand ignored",
     "    if demand is None:\n        base = draw_demand(", "    if True:\n        base = draw_demand("),
]


def tree_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(list((ROOT / "glops").glob("*.py")) + list((ROOT / "config").glob("*.json"))):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def run_tests(timeout=600) -> tuple[bool, str]:
    # GLOPS_NO_RAW: behave like CI, which has no raw Online Retail II download, so a mutant
    # only counts as caught if a check that also runs in CI catches it
    env = {**os.environ, "GLOPS_NO_RAW": "1"}
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"],
                           cwd=ROOT, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return True, "timeout"
    tail = [l for l in r.stdout.splitlines() if l.startswith("FAILED") or l.startswith("ERROR")]
    return r.returncode != 0, (tail[0] if tail else r.stdout.strip().splitlines()[-1])


def main() -> int:
    before = tree_hash()
    ok, msg = run_tests()
    if ok:
        print("baseline test run FAILS - fix the suite first:", msg)
        return 1
    results = []
    for i, (rel, name, old, new) in enumerate(MUTANTS, 1):
        path = ROOT / rel
        orig = path.read_bytes()
        # match on LF text: a Windows checkout (core.autocrlf) has CRLF files, and multi-line
        # patterns would silently never match; the original bytes are restored afterwards
        text = orig.decode("utf-8").replace("\r\n", "\n")
        if text.count(old) != 1:
            print(f"[{i:2d}] SKIP pattern not unique/missing: {name}")
            results.append({"id": i, "file": rel, "mutant": name, "status": "invalid"})
            continue
        t = time.time()
        try:
            path.write_bytes(text.replace(old, new).encode("utf-8"))
            caught, detail = run_tests()
        finally:
            path.write_bytes(orig)
        status = "caught" if caught else "ESCAPED"
        print(f"[{i:2d}] {status:7s} {rel}: {name}  ({time.time() - t:.0f}s)  {detail}", flush=True)
        results.append({"id": i, "file": rel, "mutant": name, "status": status.lower(), "detail": detail})
    assert tree_hash() == before, "source tree changed - restore failed!"
    valid = [r for r in results if r["status"] != "invalid"]
    caught = sum(r["status"] == "caught" for r in valid)
    report = {"caught": caught, "total": len(valid), "mutants": results}
    out = ROOT / "results" / "mutation_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"caught {caught}/{len(valid)}")
    # README carries the mutation score: re-render it from the new report
    subprocess.run([sys.executable, "run_all.py", "--readme-only"], cwd=ROOT, check=True)
    invalid = len(results) - len(valid)
    if invalid:
        print(f"{invalid} mutant pattern(s) did not match - the score above leaves them out")
    return 1 if invalid else 0


if __name__ == "__main__":
    sys.exit(main())
