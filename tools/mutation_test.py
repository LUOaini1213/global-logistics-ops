"""Hand-written mutation test: break the code/config on purpose, check the tests notice.

    python tools/mutation_test.py            # writes results/mutation_report.json

Each mutant is an exact, single-occurrence text replacement in one file. The file is
restored in a finally block and a SHA-256 check confirms the tree is unchanged at the end.
A mutant is CAUGHT if pytest fails (exit code != 0) within the time limit.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LF, CRLF = chr(10), chr(13) + chr(10)

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
    # ---- overseas rollout (Part 5)
    ("glops/rollout.py", "quartile fit: sigma from one z instead of two",
     'return Dist("lognorm", mu, math.log(q3 / q1) / (2 * Z75))', 'return Dist("lognorm", mu, math.log(q3 / q1) / Z75)'),
    ("glops/rollout.py", "median-mean fit drops the factor 2",
     "math.sqrt(2 * math.log(mean / md))", "math.sqrt(math.log(mean / md))"),
    ("glops/rollout.py", "occurrence mixture variance loses the Bernoulli term",
     "return self.p * (self.cond_var() + m * m) - (self.p * m) ** 2", "return self.p * self.cond_var()"),
    ("glops/rollout.py", "ocean leg drops the port-call allowance",
     'ocean_mean = nm / (24.0 * v) + sea["port_call_days_per_1000nm"] * nm / 1000.0', "ocean_mean = nm / (24.0 * v)"),
    ("glops/rollout.py", "transshipment time not conditioned on occurrence",
     "        cond = t_ts / p_ts\n", "        cond = t_ts\n"),
    ("glops/rollout.py", "planning quantile rounds down",
     'np.quantile(x, p, method="higher")', 'np.quantile(x, p, method="lower")'),
    ("glops/rollout.py", "rollout shares one inland leg across a week's sites",
     'inland = comps["inland"].sample(rng, (reps, weeks, sites))', 'inland = comps["inland"].sample(rng, (reps, weeks, 1))'),
    ("glops/rollout.py", "checker ignores overlapping stacks",
     "if min(r1[2], r2[2]) - max(r1[0], r2[0]) > 1e-9", "if min(r1[2], r2[2]) - max(r1[0], r2[0]) > 1e9"),
    ("glops/rollout.py", "stacking ignores the top-load limit",
     "return (same_fp and upper.top_ok and upper.kg <= lower.max_top_kg\n", "return (same_fp and upper.top_ok\n"),
    ("glops/rollout.py", "packer ignores the payload",
     "    if box.kg + kg > ct.payload + 1e-9:\n        return False\n", ""),
    ("glops/rollout.py", "floor bound ignores what carriers can hold",
     "floor = c_fp + max(0.0, t_fp - carriers) / max_tiers", "floor = c_fp + max(0.0, t_fp) / max_tiers"),
    ("glops/rollout.py", "split policy plans each leg at p instead of sqrt(p)",
     "        q = math.sqrt(p)\n", "        q = p\n"),
    ("glops/rollout.py", "expedited air kit never arrives first",
     "arrive = np.where(exp, np.minimum(t, decide + t2), t)", "arrive = t"),
    ("glops/rollout.py", "holding on a 360-day year",
     'rate = kit.value * c["holding_rate_per_year"] / 365.0', 'rate = kit.value * c["holding_rate_per_year"] / 360.0'),
    ("glops/rollout.py", "air chargeable weight ignores volume",
     "return max(kit.kg, kit.cbm * 1e6 / divisor_cm3_per_kg)", "return kit.kg"),
    ("glops/rollout.py", "market rule picks the smallest eligible partner",
     'picked.append(max(cands, key=lambda r: (r[share_key], r["iso3"]))["iso3"])',
     'picked.append(min(cands, key=lambda r: (r[share_key], r["iso3"]))["iso3"])'),
    ("glops/rollout.py", "consolidation forgets the later weeks' slack",
     "slack = L + 7.0 * np.arange(k)[None, None, :, None]", "slack = L + 0.0 * np.arange(k)[None, None, :, None]"),
    ("config/rollout.json", "container dwell fitted to median + mean instead of quartiles",
     '"container": "quartiles",', '"container": "median_mean",'),
]


def tree_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(list((ROOT / "glops").glob("*.py")) + list((ROOT / "config").glob("*.json"))):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def run_tests(timeout=600) -> tuple[bool, str]:
    try:
        r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider"],
                           cwd=ROOT, capture_output=True, text=True, timeout=timeout)
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
        text = orig.decode("utf-8")
        if CRLF in text:         # Windows checkout with core.autocrlf: multi-line patterns must match too
            old, new = old.replace(LF, CRLF), new.replace(LF, CRLF)
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
    report = {"caught": caught, "total": len(valid), "invalid": len(results) - len(valid), "mutants": results}
    out = ROOT / "results" / "mutation_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"caught {caught}/{len(valid)}" + (f"  WARNING: {len(results) - len(valid)} mutants invalid (not run)"
                                              if len(results) > len(valid) else ""))
    # README carries the mutation score: re-render it from the new report
    subprocess.run([sys.executable, "run_all.py", "--readme-only"], cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
