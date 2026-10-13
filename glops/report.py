"""Static HTML report (results/report.html) with charts embedded as PNG data URIs."""
from __future__ import annotations

import base64
import csv
import html
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .common import RESULTS, read_json  # noqa: E402

# reference palette (validated categorical order; light surface)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 5,
})


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _table(headers, rows) -> str:
    th = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def _ci(c, nd=1, unit=""):
    return f"{c['mean']:.{nd}f}{unit} [{c['lo']:.{nd}f}, {c['hi']:.{nd}f}]"


# ------------------------------------------------------------------ charts

def chart_slotting(s) -> str:
    names = {"random": "Random", "abc": "ABC class-based", "turnover": "Full turnover",
             "heuristic": "Turnover + affinity swaps"}
    m = list(names)
    tr = s["large"]["travel"]
    means = [tr[k]["s_shape"]["mean"] for k in m]
    err = np.array([[tr[k]["s_shape"]["mean"] - tr[k]["s_shape"]["lo"] for k in m],
                    [tr[k]["s_shape"]["hi"] - tr[k]["s_shape"]["mean"] for k in m]])
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    y = np.arange(len(m))[::-1]
    ax.barh(y, means, height=0.55, color=C1, xerr=err, error_kw={"ecolor": INK2, "elinewidth": 1})
    for yi, v, e in zip(y, means, err[1]):
        ax.text(v + e + 2, yi, f"{v:.1f} m", va="center", color=INK, fontsize=8)
    ax.set_yticks(y, [names[k] for k in m])
    ax.set_xlabel("mean S-shape travel per order (m), 95% CI over 20 seeds")
    ax.set_title("Pick travel by slotting method (1,000 SKUs, 1,200 slots)")
    ax.grid(axis="y", visible=False)
    return _png(fig)


def chart_tradeoff() -> str:
    rows = _csv(RESULTS / "inventory" / "tradeoff.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    names = {"base_stock": "base-stock (daily)", "sS": "(s,S)", "rQ": "(r,Q), Q = EOQ"}
    for ax, mode in zip(axes, ("ocean", "air")):
        for col, pol in zip((C1, C2, C3), names):
            rr = [r for r in rows if r["mode"] == mode and r["policy"] == pol]
            x = [100 * float(r["fill_rate"]) for r in rr]
            yv = [float(r["cost_total"]) for r in rr]
            ax.plot(x, yv, marker="o", color=col, label=names[pol])
        ax.axvline(95, color=MUTED, lw=1, ls="--")
        ax.set_title(f"{mode.capitalize()} replenishment")
        ax.set_xlabel("fill rate (%)")
        ax.set_ylabel("total cost per day (USD, illustrative)")
    axes[1].legend(loc="upper center", fontsize=8)
    fig.suptitle("Service vs cost trade-off (each point = one safety factor z; 10 reps x 10 years)",
                 x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout()
    return _png(fig)


def chart_validation() -> str:
    rows = [r for r in _csv(RESULTS / "inventory" / "validation_analytic_vs_sim.csv")
            if r["metric"] == "fill_rate"]
    a = np.array([100 * float(r["analytic"]) for r in rows])
    s = np.array([100 * float(r["sim_mean"]) for r in rows])
    lo = np.array([100 * float(r["sim_lo"]) for r in rows])
    hi = np.array([100 * float(r["sim_hi"]) for r in rows])
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.plot([60, 100], [60, 100], color=AXIS, lw=1)
    ax.errorbar(a, s, yerr=[s - lo, hi - s], fmt="o", color=C1, ecolor=INK2, elinewidth=1, capsize=2)
    ax.set_xlabel("analytic fill rate (%)")
    ax.set_ylabel("simulated fill rate (%), 95% CI")
    ax.set_title("Simulator vs exact normal-demand formula")
    return _png(fig)


def chart_breakeven(inv) -> str:
    rows = inv["breakeven_table"]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    for col, mode in ((C1, "ocean"), (C2, "air")):
        rr = [r for r in rows if r["mode"] == mode]
        ax.plot([r["unit_value"] for r in rr], [r["cost_at_95"] for r in rr], marker="o",
                color=col, label=mode)
        ax.text(rr[-1]["unit_value"] + 8, rr[-1]["cost_at_95"], mode, color=INK2, va="center")
    if inv["breakeven_unit_value"]:
        ax.axvline(inv["breakeven_unit_value"], color=MUTED, ls="--", lw=1)
        ax.text(inv["breakeven_unit_value"] + 8, ax.get_ylim()[1] * 0.92,
                f"break-even ~ USD {inv['breakeven_unit_value']:.0f}/unit", color=INK2, fontsize=8)
    ax.set_xlabel("unit value (USD)")
    ax.set_ylabel("min cost/day at >= 95% fill (USD)")
    ax.set_title("Ocean vs air: who wins depends on unit value")
    ax.legend(loc="upper left")
    return _png(fig)


def chart_landed() -> str:
    rows = [r for r in _csv(RESULTS / "landed" / "landed_cost_matrix.csv")
            if r["shipment"] == "FCL radio units"]
    base = next(float(r["landed_net"]) for r in rows if r["rule"] == "FCA")
    rules = [r["rule"].replace("FCA_TERMINAL", "FCA@term") for r in rows]
    d = [float(r["landed_net"]) - base for r in rows]
    top = 450.0
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    shown = [min(v, top) for v in d]
    cols = [C2 if v > top else C1 for v in d]
    ax.bar(rules, shown, color=cols, width=0.6)
    for i, v in enumerate(d):
        if v > top:
            ax.text(i - 0.4, top * 0.82, f"+{v:,.0f} (off scale:\nunrecoverable GST)", ha="right",
                    va="center", color=INK, fontsize=7)
        else:
            ax.text(i, v + (15 if v >= 0 else -60), f"{v:,.0f}", ha="center", color=INK, fontsize=7)
    ax.set_ylim(-800, top)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_ylabel("net landed cost minus FCA (USD)")
    ax.set_title("FCL 40' into Singapore: landed cost by Incoterm (5% seller markup)")
    ax.tick_params(axis="x", rotation=45)
    return _png(fig)


def chart_reverse() -> str:
    rows = _csv(RESULTS / "reverse" / "sensitivity.csv")
    pr = [r for r in rows if r["sweep"] == "resale_price"]
    cap = [r for r in rows if r["sweep"] == "penang_capacity"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    ax = axes[0]
    x = [float(r["value"]) for r in pr]
    ax.plot(x, [100 * float(r["refurb_share_A"]) for r in pr], marker="o", color=C1, label="grade A refurbished")
    ax.plot(x, [100 * float(r["refurb_share_B"]) for r in pr], marker="o", color=C2, label="grade B refurbished")
    ax.set_xlabel("resale price of a refurbished unit (USD)")
    ax.set_ylabel("share of returns refurbished (%)")
    ax.set_title("Routing vs resale price")
    ax.legend(loc="center right", fontsize=8)
    ax = axes[1]
    ax.plot([float(r["value"]) for r in cap], [float(r["net_value"]) / 1000 for r in cap], marker="o", color=C1)
    ax.set_xlabel("Penang refurb capacity (units/month)")
    ax.set_ylabel("network net value (USD thousand / month)")
    ax.set_title("Value of refurb capacity")
    fig.tight_layout()
    return _png(fig)


def chart_season(pub) -> str:
    idx = pub["seasonal_index"]
    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.bar(range(1, 13), [100 * (v - 1) for v in idx], color=C1, width=0.6)
    ax.axhline(0, color=AXIS, lw=1)
    ax.set_xticks(range(1, 13), ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"])
    ax.set_ylabel("deviation from mean month (%)")
    ax.set_title("Singapore container throughput seasonality, 2010-2025")
    ax.grid(axis="x", visible=False)
    return _png(fig)


# ------------------------------------------------------------------ page

CSS = """
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--accent:#2a78d6;--warn:#8a5a00;--warnbg:#fff4dc}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--line:#2c2c2a;--accent:#3987e5;--warn:#fab219;--warnbg:#2a2310}}
:root[data-theme="dark"]{--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;--line:#2c2c2a;--accent:#3987e5;--warn:#fab219;--warnbg:#2a2310}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 8px;border-top:1px solid var(--line);padding-top:20px}
h3{font-size:15px;margin:18px 0 6px}p,li{color:var(--ink2)}.lead{color:var(--ink2);margin:0 0 16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:16px 0}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:12px 14px}.tile b{display:block;font-size:22px;color:var(--ink)}.tile span{font-size:13px;color:var(--muted)}
.fig{background:#fcfcfb;border:1px solid var(--line);border-radius:10px;padding:8px;margin:12px 0}.fig img{max-width:100%;height:auto;display:block}
.tw{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums;width:100%}th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{color:var(--ink2);font-weight:600}
.note{background:var(--warnbg);border-left:3px solid var(--warn);padding:8px 12px;border-radius:6px;color:var(--ink2);font-size:14px}code{font-size:13px}
"""


def build(summary: dict) -> str:
    s, inv, val, land, rev, pub = (summary[k] for k in ("slotting", "inventory", "inventory_validation",
                                                         "landed", "reverse", "public"))
    L = s["large"]
    sm = s["small"]
    tiles = [
        (f"{L['red_vs_random']['heuristic']['s_shape']['mean']:.1f}%", "pick travel cut vs random slotting (20 seeds)"),
        (f"{L['red_vs_abc']['heuristic']['s_shape']['mean']:.1f}%", "pick travel cut vs ABC class-based"),
        (f"{val['n_pass_3se']}/{val['n_checks']}", "simulator checks within 3 SE of exact formula"),
        (f"USD {inv['breakeven_unit_value']:.0f}", "unit value where air beats ocean (illustrative)"),
        (f"{land['fcl_sg']['ddp_premium_pct']:.1f}%", "DDP vs FCA when import GST is unrecoverable"),
        (f"{rev['lp_vs_greedy_gain_pct']:.1f}%", "returns-network value: LP vs nearest-first rule"),
    ]
    tiles_html = "".join(f'<div class="tile"><b>{a}</b><span>{html.escape(b)}</span></div>' for a, b in tiles)

    names = {"random": "Random", "abc": "ABC class-based", "turnover": "Full turnover",
             "heuristic": "Turnover + affinity swaps", "milp": "MILP (HiGHS)"}
    slot_rows = []
    for k in ("random", "abc", "turnover", "heuristic"):
        slot_rows.append([names[k], _ci(L["travel"][k]["s_shape"]), _ci(L["travel"][k]["return"]),
                          "-" if k == "random" else _ci(L["red_vs_random"][k]["s_shape"], 1, "%"),
                          "-" if k in ("random", "abc") else _ci(L["red_vs_abc"][k]["s_shape"], 1, "%")])
    small_rows = []
    for k in ("random", "abc", "turnover", "heuristic", "milp"):
        small_rows.append([names[k], _ci(sm["travel"][k]["s_shape"], 2),
                           "-" if k == "random" else _ci(sm["red_vs_random"][k]["s_shape"], 1, "%")])

    val_rows = [[r["z"], r["metric"], f"{float(r['analytic']):.4f}", f"{float(r['sim_mean']):.4f}",
                 f"[{float(r['sim_lo']):.4f}, {float(r['sim_hi']):.4f}]", f"{float(r['z_dev']):+.2f}"]
                for r in _csv(RESULTS / "inventory" / "validation_analytic_vs_sim.csv")]
    lt_rows = [[r["mode"], r["z"], f"{100*float(r['target_csl']):.1f}%", f"{100*float(r['sim_csl']):.1f}%",
                f"{100*float(r['sim_fill_rate']):.1f}%"] for r in _csv(RESULTS / "inventory" / "random_leadtime_formula_check.csv")]
    best_rows = []
    for key, b in inv["best"].items():
        mode, pol = key.split("|")
        a = b["at_95"]
        best_rows.append([mode, pol, "-" if a is None else f"{a['z']:.2f}",
                          "-" if a is None else f"{100*a['fill_rate']:.1f}%",
                          "-" if a is None else f"{a['cost_total']:.1f} [{a['cost_lo']:.1f}, {a['cost_hi']:.1f}]",
                          "-" if a is None else f"{a['on_hand']:.0f}"])

    payer = _csv(RESULTS / "landed" / "payer_matrix.csv")
    elems = [k for k in payer[0] if k != "rule"]
    short = {"export_packaging": "pack", "loading_at_seller": "load", "export_clearance": "exp.clr",
             "pre_carriage": "pre-carr", "unloading_export_terminal": "unl.orig", "loading_main_carriage": "load.main",
             "main_carriage": "main", "cargo_insurance": "ins", "unloading_import_terminal": "unl.dest",
             "import_clearance": "imp.clr", "import_duty": "duty", "import_tax": "VAT/GST",
             "on_carriage": "on-carr", "unloading_at_destination": "unl.final"}
    payer_rows = [[p["rule"]] + [p[e] for e in elems] for p in payer]
    lm = _csv(RESULTS / "landed" / "landed_cost_matrix.csv")
    land_rows = [[r["shipment"], r["rule"], r["insurance_clause"], f"{float(r['price']):,.0f}",
                  f"{float(r['buyer_paid_separately']):,.0f}", f"{float(r['duty']):,.0f}", f"{float(r['tax']):,.0f}",
                  f"{float(r['landed_net']):,.0f}", f"{float(r['landed_net_per_unit']):,.2f}"]
                 for r in lm if r["rule"] in ("EXW", "FCA", "FOB", "CIF", "CIP", "DAP", "DDP")]

    rs = rev["summary"]
    sens = _csv(RESULTS / "reverse" / "sensitivity.csv")
    sens_rows = [[r["sweep"], r["value"], f"{float(r['net_value']):,.0f}", f"{100*float(r['refurb_share_A']):.0f}%",
                  f"{100*float(r['refurb_share_B']):.0f}%", f"{float(r['overflow']):,.0f}"]
                 for r in sens if r["sweep"] in ("volume_mult", "transport_mult", "no_cross_border_recycling")]
    seas = inv["seasonality"]
    meta = pub["meta"]

    body = f"""
<main>
<h1>Global logistics operations analytics</h1>
<p class="lead">Personal portfolio project, October 2026, built with AI assistance. Four models for a
network-equipment supply chain: warehouse slotting, inventory policy, landed cost by Incoterms 2020, and a
returns network. <b>All operational data is synthetic and all rates are illustrative</b>; the only real data is
a public port-throughput series used as a seasonality proxy. Not tax, customs or legal advice.</p>
<div class="tiles">{tiles_html}</div>

<h2>1. Warehouse slotting</h2>
<p>Single-block warehouse, {s['large_instance']['skus']:,} SKUs in {s['large_instance']['slots']:,} slots (10 aisles x 20 bays x 2 sides x 3 levels),
depot front-left. Velocity and co-pick affinity are estimated on a training order history; every method is then scored
on {s['large_instance']['test_orders_per_seed']:,} <i>held-out</i> orders per seed by simulating S-shape and return routing.
The affinity weight lambda = {s['lambda_star']} was tuned on three development seeds that are not among the 20 evaluation seeds.
Heavy (&gt;15 kg) and large items are forced to floor level; medium items to the lower two levels.</p>
<div class="fig"><img alt="Bar chart of mean travel per order by slotting method" src="{chart_slotting(s)}"></div>
{_table(["method", "S-shape m/order [95% CI]", "return m/order [95% CI]", "reduction vs random", "reduction vs ABC"], slot_rows)}
<h3>Exact MILP on a small instance</h3>
<p>{s['small_instance']['skus']} SKUs, {s['small_instance']['slots']} slots, top {sm['milp']['n_pairs']} affinity pairs, {sm['milp']['n_binaries']} binaries.
HiGHS proved optimality in {sm['milp']['n_optimal']}/{sm['milp']['n']} seeds (max MIP gap {sm['milp']['max_mip_gap']:.1e}).
On the proxy objective the swap heuristic is {sm['milp']['heuristic_gap_pct']['mean']:.2f}% above the optimum on average
(worst {sm['milp']['max_heuristic_gap_pct']:.2f}%). Note the MILP optimises a proxy (velocity x depot distance + lambda x
rectilinear pair distance), not the routed tour length, so "optimal" here means optimal for the proxy.</p>
{_table(["method", "S-shape m/order [95% CI]", "reduction vs random"], small_rows)}

<h2>2. Inventory policy: ocean vs air</h2>
<p>Daily-review simulation with full backordering, stochastic demand and lognormal lead times (ocean 32 +/- 6 d,
air 6 +/- 1.5 d). Costs: holding (25%/yr of value) on stock on hand <i>and in transit</i>, backorders, per-order cost,
per-unit freight. All cost inputs are illustrative (<code>config/inventory_scenario.json</code>).</p>
<div class="fig"><img alt="Fill rate versus cost per day for three policies, ocean and air" src="{chart_tradeoff()}"></div>
{_table(["mode", "policy", "z", "fill rate", "cost/day [95% CI]", "mean on hand"], best_rows)}
<p>Daily base-stock pays the fixed order cost every day, so with ocean's order cost it is
{inv['base_stock_vs_rQ_ocean_95_pct']:.0f}% dearer than (r,Q) at the same >=95% fill target. With a USD 400 unit, air is cheaper
overall because ocean ties up ~4x more stock in transit and on the shelf; the break-even below is the more useful output.</p>
<div class="fig"><img alt="Cost per day versus unit value for ocean and air" src="{chart_breakeven(inv)}"></div>
<h3>Validation against an exact formula</h3>
<p>For i.i.d. normal daily demand and a fixed lead time L = {val['lead_time']} d, a base-stock level S gives exact
fill rate 1 - (E[(D<sub>L+1</sub>-S)<sup>+</sup>] - E[(D<sub>L</sub>-S)<sup>+</sup>])/mu, on-hand S - L mu + E[(D<sub>L</sub>-S)<sup>+</sup>]
and no-stockout probability Phi(z). {val['reps']} replications x {val['days']:,} days: <b>{val['n_pass_3se']}/{val['n_checks']}</b>
comparisons within 3 standard errors, {val['n_within_95ci']}/{val['n_checks']} inside the 95% CI;
largest fill-rate gap {val['max_abs_fill_rate_diff_pp']:.2f} pp. Inventory balance error &lt; {max(val['max_balance_error'], 1e-12):.0e} units.</p>
<div class="fig"><img alt="Analytic versus simulated fill rate scatter" src="{chart_validation()}"></div>
{_table(["z", "metric", "analytic", "simulated", "95% CI", "deviation (SE)"], val_rows)}
<p>The textbook safety-stock formula with random lead time, sigma<sup>2</sup> = (L+1) sigma<sub>D</sub><sup>2</sup> + mu<sup>2</sup> sigma<sub>L</sub><sup>2</sup>,
is <i>not</i> exact here: with daily ordering, orders cross and lead-time noise averages out, so the formula over-protects
(it is an approximation, reported without pass/fail):</p>
{_table(["mode", "z", "target CSL", "simulated CSL", "simulated fill"], lt_rows)}
<h3>Seasonality stress test (public data)</h3>
<p>Monthly demand scaled by the Singapore port-throughput seasonal index (a <i>proxy</i>: port boxes are not SKU demand).
The ocean (r,Q) policy tuned for flat demand delivers {100*seas['flat_fill_rate']:.1f}% on flat demand and
{100*seas['overall_seasonal_fill']:.1f}% on seasonal demand; the weakest month is month {seas['worst_month']} at
{100*seas['worst_month_fill']:.1f}%. The index ranges from {100*(min(pub['seasonal_index'])-1):+.0f}% to {100*(max(pub['seasonal_index'])-1):+.0f}% around the mean month, so the effect is small.</p>
<div class="fig"><img alt="Monthly seasonal index bar chart" src="{chart_season(pub)}"></div>

<h2>3. Landed cost by Incoterms 2020</h2>
<p class="note">Rates are ILLUSTRATIVE round numbers, not quotes. Singapore GST 9% and its valuation basis were checked
against IRAS / Singapore Customs pages (2026-10-13); the EU-style and US-style duty and VAT rates are placeholders.
The responsibility split is a model of the rules checked against ICC, trade.gov and two secondary summaries; the binding
text is the ICC publication. Not tax advice.</p>
<p>S = seller pays (and re-invoices inside the price with a {100*land['markup']:.0f}% markup), B = buyer pays.
Insurance is an obligation only under CIF (ICC C) and CIP (ICC A); elsewhere the model charges it to the risk bearer.</p>
{_table(["rule"] + [short[e] for e in elems], payer_rows)}
<div class="fig"><img alt="Landed cost difference versus FCA by Incoterm" src="{chart_landed()}"></div>
<p>With pass-through pricing (0% markup, equal insurance rates) the buyer's net landed cost is identical under every
rule except DDP (max spread {land['conservation_max_spread_usd']:.1e} USD) - the Incoterm moves cost between invoices,
it does not create it. The DDP gap of {land['fcl_sg']['ddp_premium_pct']:.1f}% is entirely the assumption that a foreign
seller acting as importer cannot reclaim Singapore GST; flip <code>seller_recovers_import_tax_ddp</code> and it disappears.
CIF looks cheaper only because its minimum cover (ICC C) is narrower than ICC A.</p>
{_table(["shipment", "rule", "ins.", "price", "buyer pays separately", "duty", "GST/VAT", "net landed", "per unit"], land_rows)}

<h2>4. Reverse logistics network</h2>
<p>Five return hubs (SG, KL, BKK, JKT, MNL), two refurb centres, five recyclers plus uncapacitated spot recycling.
LP (HiGHS) decides refurbish vs recycle per hub and grade. All numbers illustrative (<code>config/reverse_network.json</code>).</p>
<div class="fig"><img alt="Refurbishment share versus resale price and network value versus refurb capacity" src="{chart_reverse()}"></div>
<p>Base case: net value USD {rev['net_value']:,.0f}/month vs USD {rev['greedy_net_value']:,.0f} for a nearest-facility rule
(+{rev['lp_vs_greedy_gain_pct']:.1f}%). All grade A and {100*rs['refurb_share_B']:.0f}% of grade B are refurbished; the
secondary-market cap binds (shadow price USD {rev['shadow_price_market_cap']:.2f}/unit) and Penang capacity binds (shadow price
USD {rev['shadow_price_penang_capacity']:.2f}/unit-month; finite-difference check {rev['finite_difference_penang']:.2f}).
Forbidding cross-border recycling costs {rev['no_cross_border']['loss_pct']:.1f}% of value.</p>
{_table(["sweep", "value", "net value USD/month", "A refurbished", "B refurbished", "overflow units"], sens_rows)}

<h2>Data and licences</h2>
<p>{html.escape(meta['attribution'])}</p>
<p>Unit: {html.escape(meta['unit_note'])}</p>
<p>Everything else is synthetic (generators and seeds in the source) or an illustrative assumption in <code>config/</code>.</p>
</main>"""
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' "
            "content='width=device-width,initial-scale=1'><title>Global Logistics Ops</title>"
            f"<style>{CSS}</style></head><body>{body}</body></html>")


def write(summary: dict) -> None:
    (RESULTS / "report.html").write_text(build(summary), encoding="utf-8", newline="\n")
