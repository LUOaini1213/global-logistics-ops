# global-logistics-ops

Logistics and warehouse operations analytics for a global network-equipment supply chain
(transceivers, cables, PSUs, line cards, radio units, small cells): **warehouse slotting**,
**inventory policy (ocean vs air)**, **landed cost by Incoterms 2020**, a
**reverse-logistics (returns / refurbish / recycle) network**, and **overseas rollout logistics**
(transport options, ship-date planning, container loading and transport cost for equipment shipped from Shenzhen
to seven overseas markets).

> **What this is.** A personal portfolio project, October 2026, built with AI assistance
> (code, tests and write-up were drafted with an AI coding assistant and checked by running
> them). It is not work for, or data from, any company.
>
> **Data.** All operational data is **synthetic** (documented generators, fixed seeds). All
> freight, handling, insurance, duty and VAT inputs except Singapore GST are **ILLUSTRATIVE**
> round numbers, not quotes. Real data: one public dataset in Parts 1-4 (Singapore monthly
> container throughput, Singapore Open Data Licence v1.0), used as a seasonality proxy; in Part 5, public
> trade shares (UN Comtrade), World Bank port and customs dwell times (LPI 2.0, CC BY 4.0) and airport coordinates
> (OurAirports); its sea distances are computed with an open-source routing library. Part 5 labels every input
> real, SYNTHETIC or ILLUSTRATIVE.
> **Not tax, customs or legal advice.**

Every number below between the markers is written by `run_all.py` from `results/summary.json`
and checked by `tests/test_readme_and_data.py`; none is typed by hand.

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Windows; use .venv/bin/python elsewhere
.venv/Scripts/python run_all.py          # ~5 min, writes results/ and results/report.html
.venv/Scripts/python -m pytest           # ~10 s
.venv/Scripts/python tools/mutation_test.py   # optional, ~8 min, writes results/mutation_report.json
```

HiGHS (via `highspy`) solves the MILP and LPs. Everything runs single-threaded.
`results/report.html` is a self-contained static report with all charts and tables.

## 1. Warehouse slotting

* **Instance.** <!--n:slot_skus-->1,000<!--/n--> SKUs, <!--n:slot_slots-->1,200<!--/n--> slots
  (10 aisles x 20 bays x 2 sides x 3 levels), depot front-left. SKU categories, sizes and weights,
  skewed lognormal popularity (a minority of SKUs carries most picks), and "kits" of co-ordered
  items (chassis + PSU + line card + optics ...) come from `glops/slotting.py::generate(seed)`.
  Heavy (>15 kg) and large items must sit on the floor level; medium items on the lower two.
* **Methods.** random; ABC class-based (A zone nearest the depot, random inside a zone);
  full-turnover (fastest movers nearest); and a swap local search on
  `velocity x depot distance + lambda x co-pick count x rectilinear pair distance`
  (lambda = <!--n:slot_lambda-->0.5<!--/n-->, tuned on three development seeds that are not used for evaluation).
* **Evaluation.** Velocity and affinity are estimated on a training order history; travel is
  then measured on *held-out* orders by simulating S-shape and return routing. 95% CIs are over
  <!--n:slot_seeds-->20<!--/n--> independent seeds (paired per seed).

| comparison (S-shape routing) | travel reduction |
|---|---|
| swap heuristic vs random | <!--n:slot_heur_vs_random-->49.4% (95% CI 48.7-50.2)<!--/n--> |
| swap heuristic vs ABC class-based | <!--n:slot_heur_vs_abc-->21.2% (95% CI 20.6-21.8)<!--/n--> |
| full turnover vs ABC class-based | <!--n:slot_turn_vs_abc-->13.0% (95% CI 12.2-13.9)<!--/n--> |
| ABC class-based vs random | <!--n:slot_abc_vs_random-->35.8% (95% CI 34.8-36.8)<!--/n--> |
| swap heuristic vs random, *return* routing | <!--n:slot_heur_vs_random_return-->58.0% (95% CI 57.2-58.7)<!--/n--> |

**Exact MILP (small instance).** 30 SKUs / 36 slots / 30 affinity pairs: HiGHS proves optimality on
<!--n:slot_milp_optimal-->10/10<!--/n--> seeds; the swap heuristic is on average
<!--n:slot_heur_gap-->0.63%<!--/n--> above the proxy optimum (worst seed <!--n:slot_heur_gap_max-->2.37%<!--/n-->).
Tests check the MILP against brute-force enumeration on an 8-slot instance. Caveat: the MILP
optimises a *proxy* objective (it cannot see the routing), so "optimal" means optimal for the proxy.

## 2. Inventory policy: ocean vs air

Daily-review simulation with full backordering, normal demand, lognormal lead times
(ocean 32 +/- 6 d, air 6 +/- 1.5 d), holding cost on stock on hand **and in transit**,
per-order and per-unit freight costs (`config/inventory_scenario.json`, illustrative). Policies:
daily base-stock, (s,S) and (r,Q) with Q = EOQ, each swept over the safety factor to give a
fill-rate vs cost curve.

* Cheapest (r,Q) setting with >= 95% fill for a USD 400 optical module:
  ocean <!--n:inv_ocean_cost-->USD 621/day<!--/n-->, air <!--n:inv_air_cost-->USD 415/day<!--/n-->.
  Air wins at this unit value because ocean ties up far more stock in transit.
* Break-even unit value (ocean cheaper below, air above): <!--n:inv_breakeven-->USD 185<!--/n-->.
* Daily base-stock with ocean's per-order cost is <!--n:inv_bs_penalty-->118%<!--/n--> dearer than (r,Q) at the same >= 95% target.

**Validation against an exact formula.** For i.i.d. normal demand and a fixed lead time L,
base-stock S has exact fill rate `1 - (E[(D_{L+1}-S)^+] - E[(D_L-S)^+]) / mu`, on-hand
`S - L mu + E[(D_L-S)^+]` and no-stockout probability `Phi(z)`. Simulation vs formula:
<!--n:inv_val_pass-->20/20<!--/n--> comparisons within 3 standard errors
(<!--n:inv_val_ci-->19/20<!--/n--> inside the 95% CI), largest fill-rate gap <!--n:inv_val_maxdiff-->0.03 pp<!--/n-->.
Inventory balance and pipeline identities are asserted in every test run.
Deterministic zero-day replenishment is received immediately after the daily order,
before end-of-day inventory and holding costs are measured. It can clear a backorder
but cannot retroactively improve that day's immediate fill rate. Negative or non-finite
lead-time parameters, and a zero-mean positive-variance lognormal lead time, are rejected.
The textbook random-lead-time formula is *not* exact when orders cross: with ocean lead times a
z = 1 target (84.1% CSL) actually delivered <!--n:inv_lt_ocean_csl-->99.3%<!--/n--> under daily ordering.
This is reported as an approximation gap, not hidden.

**Seasonality (public data).** The Singapore container-throughput seasonal index (February,
around Lunar New Year, is the weakest month) scales demand in a stress test of the ocean (r,Q) policy. Port boxes are
not SKU demand; this is a proxy only. The effect is small (see report).

## 3. Landed cost by Incoterms 2020

`config/incoterms_2020.json` encodes who pays 14 cost elements under EXW, FCA (two named-place
variants), FAS, FOB, CFR, CIF, CPT, CIP, DAP, DPU and DDP, plus the rules' transport-mode
restrictions (FAS/FOB/CFR/CIF are sea/inland-waterway only; the model raises an error for air).
`glops/landed.py` prices five example shipments (FCL 40', LCL, air; SG / EU-style / US-style regimes).

* With pass-through pricing the buyer's net landed cost is identical under every rule except DDP
  (max spread <!--n:land_conservation-->5.8e-11 USD<!--/n-->): an Incoterm moves cost between invoices, it does not create it.
* With a 5% seller markup on the logistics it arranges, the non-DDP spread for the FCL shipment is only
  <!--n:land_non_ddp_spread-->0.08%<!--/n-->.
* DDP is <!--n:land_ddp_premium-->9.0%<!--/n--> dearer for a GST-registered Singapore buyer, entirely
  because the model assumes a foreign seller acting as importer of record cannot reclaim the 9% GST.
  That is an assumption (switchable), and it is the point: the importer-of-record choice matters more than freight.
* For a USD 100 item, air instead of LCL raises landed cost per unit by only <!--n:land_air_premium-->2.5%<!--/n-->
  (inventory effects are in Part 2).

**Sources for the responsibility split** (accessed 2026-10-13). The binding text is ICC Publication
723E, which is not freely available, so the split was checked against:
1. ICC, *Incoterms 2020* page - CIF default cover Institute Cargo Clauses (C), CIP now (A),
   DPU replaces DAT and the seller unloads, DAP seller does not unload, costs listed in A9/B9.
   <https://iccwbo.org/business-solutions/incoterms-rules/incoterms-2020/>
2. U.S. International Trade Administration, *Know Your Incoterms* - 11 rules, 7 any-mode, 4 sea only.
   <https://www.trade.gov/know-your-incoterms>
3. Wikipedia, *Incoterms* (2020 cost-allocation table; secondary). <https://en.wikipedia.org/wiki/Incoterms>
4. Gebrueder Weiss, *Incoterms 2020* (forwarder summary; secondary) - FCA variants, C-rule unloading
   "if specified in the contract of carriage". <https://www.gw-world.com/incoterms>
5. Trade Finance Global, CIF guide - minimum cover 110% of contract price.
   <https://www.tradefinanceglobal.com/wp-content/uploads/2020/01/Cost__Insurance__Freight_CIF.pdf>

**Marked uncertain / contract-dependent:** destination terminal handling under C rules (buyer unless
the seller's carriage contract includes it; switch provided); FCA pre-carriage (depends on the named
place; both variants modelled); insurance outside CIF/CIP (no obligation; the model charges the risk
bearer, an assumption); origin THC under FOB (follows the rule, though disputed in practice).

**Tax inputs.** Singapore: GST 9% on (customs value + duty), CIF valuation, telecom equipment not in the
dutiable categories (IRAS and Singapore Customs pages, cited in `config/duty_regimes_example.json`).
"EU-style" (CIF valuation per UCC Art. 71) and "US-style" (transaction value excluding international
freight) regimes use **placeholder** duty/VAT rates and are labelled as examples.

## 4. Reverse logistics

Five return hubs (Singapore, Kuala Lumpur, Bangkok, Jakarta, Manila), two refurbishment centres, five
recyclers plus uncapacitated "spot" recycling. Returns are graded A (no fault found), B (repairable),
C (end of life). An LP (HiGHS) routes each hub/grade to refurbish or recycle, with yields, capacities,
cross-border compliance costs and a secondary-market cap (`config/reverse_network.json`, illustrative).

* Net value <!--n:rev_value-->USD 369,472<!--/n--> per month, <!--n:rev_gain-->3.4%<!--/n--> above a nearest-facility rule of thumb.
* Shadow price of one more unit of Penang refurb capacity <!--n:rev_shadow-->USD 8.94<!--/n-->/month,
  confirmed by re-solving with +1 capacity (<!--n:rev_fd-->USD 8.94<!--/n-->).
* All grade-A returns are refurbished once the resale price reaches <!--n:rev_threshold-->USD 50<!--/n-->.
* Forbidding cross-border shipment of e-waste for recycling (cf. Basel Convention controls) costs
  <!--n:rev_nocross-->1.2%<!--/n--> of network value in this instance.

The independent flow checker rejects non-finite or non-numeric quantities, unknown nodes/grades,
malformed flow keys, and prohibited cross-border recycling flows; omitted flow families mean zero flow.
Cross-border returns for repair remain a separate permitted route. Summaries require a feasible plan.
For a grade with no returns or a facility with zero capacity and no throughput, the reported ratio is
`0.0`, meaning **no activity**, not measured operational performance. This convention keeps shutdown
and zero-volume scenarios JSON-compatible for both LP and greedy solutions; positive throughput at a
closed facility is a constraint violation (subject to the checker's numerical tolerance).

## 5. Overseas rollout logistics

Scenario: a network-equipment vendor ships radio-access equipment (synthetic site kit: radio units, baseband,
antennas, cabinet, power, batteries, cables; 7 packed units per site, no brands) from Shenzhen to network-rollout
projects in <!--n:ro_n_markets-->7<!--/n--> markets, by sea from Yantian or by air from Shenzhen Bao'an.
Code: `glops/rollout.py`; assumptions: `config/rollout.json`; results: `results/rollout/`.

**What is real, what is not.**

| input | status | source |
|---|---|---|
| choice of markets | **real** data, documented rule | UN Comtrade: China's exports of HS 851761 "base stations", 2024 + 2025 |
| destination port + customs time (sea) | **real** quartiles and mean, lognormal shape assumed | World Bank LPI 2.0, container import dwell time, 2024 |
| origin port time (Yantian) | **real**, national figure for China | LPI 2.0, container export dwell time, 2024 |
| transshipment probability and time | **real** but for all origins, not only China | LPI 2.0, number of / time in transshipment, 2024 |
| destination airport + customs time | **real** median and mean, lognormal shape assumed | LPI 2.0, aviation import dwell time (Cargo iQ), 2024 |
| sea distances | computed | `searoute` 1.6.0 shortest paths, Yantian to each port |
| airport positions | **real** | OurAirports |
| sea speed, port-call allowance, flight and handling times, inland leg | ASSUMPTION / **ILLUSTRATIVE** | `config/rollout.json` |
| site kit (dimensions, weights, values), rollout schedule | **SYNTHETIC** | `config/rollout.json` |
| freight rates, fees, holding and storage costs | **ILLUSTRATIVE** | round numbers; public indices quoted below only for scale |

**Market choice.** Partners are ranked by their share of China's 2024-2025 HS 851761 exports; a partner is eligible only
if LPI 2.0 publishes both its container and its aviation import dwell time for 2024 (this drops landlocked partners
and those without data, e.g. Uzbekistan, Russia, Uganda, Iran and Ukraine). The largest eligible partner in each of seven
regions gives Indonesia (Tanjung Priok / CGK), Pakistan (Karachi / KHI), Iraq (Umm Qasr / BGW), Algeria (Algiers / ALG),
Nigeria (Apapa / LOS), Peru (Callao / LIM) and Italy (Genoa / MXP), together <!--n:ro_cover-->33.8%<!--/n--> of that
export trade. The rule is fixed in code and a test re-derives it from the committed shares. Using 2025 alone would pick
Lebanon instead of Iraq and Argentina instead of Peru.

**(a) Door-to-site lead time.** Sea = truck to port + origin port dwell + ocean (distance / 16 kn + 0.5 day per 1,000 nm
for intermediate calls, CV 10%) + transshipment + destination port and customs dwell + inland leg (3 +/- 1.5 days).
Air = origin handling + flight (great circle x 1.15 at 750 km/h) + one hub transfer + check-in + airport/customs dwell +
inland leg. 200,000 Monte Carlo draws per market and mode (fixed seeds). Tests check the sample mean and variance against
the analytic sums of the component means and variances (largest deviation in this run: <!--n:ro_check_z-->2.1 / 3.0<!--/n-->
standard errors for mean / variance).

| market | sea mean (d) | sea P95 (d) | buffer P95 - mean (d) | port + customs share of mean | ... of the slowest 5% | air mean (d) | air P95 (d) |
|---|---|---|---|---|---|---|---|
| Indonesia | <!--n:ro_idn_sea_mean-->24<!--/n--> | <!--n:ro_idn_sea_p95-->41<!--/n--> | <!--n:ro_idn_sea_buf-->17<!--/n--> | <!--n:ro_idn_port-->34%<!--/n--> | <!--n:ro_idn_tail-->47%<!--/n--> | <!--n:ro_idn_air_mean-->9<!--/n--> | <!--n:ro_idn_air_p95-->15<!--/n--> |
| Pakistan | <!--n:ro_pak_sea_mean-->37<!--/n--> | <!--n:ro_pak_sea_p95-->56<!--/n--> | <!--n:ro_pak_sea_buf-->19<!--/n--> | <!--n:ro_pak_port-->36%<!--/n--> | <!--n:ro_pak_tail-->57%<!--/n--> | <!--n:ro_pak_air_mean-->12<!--/n--> | <!--n:ro_pak_air_p95-->23<!--/n--> |
| Iraq | <!--n:ro_irq_sea_mean-->44<!--/n--> | <!--n:ro_irq_sea_p95-->65<!--/n--> | <!--n:ro_irq_sea_buf-->21<!--/n--> | <!--n:ro_irq_port-->28%<!--/n--> | <!--n:ro_irq_tail-->57%<!--/n--> | <!--n:ro_irq_air_mean-->9<!--/n--> | <!--n:ro_irq_air_p95-->15<!--/n--> |
| Algeria | <!--n:ro_dza_sea_mean-->68<!--/n--> | <!--n:ro_dza_sea_p95-->98<!--/n--> | <!--n:ro_dza_sea_buf-->29<!--/n--> | <!--n:ro_dza_port-->37%<!--/n--> | <!--n:ro_dza_tail-->57%<!--/n--> | <!--n:ro_dza_air_mean-->19<!--/n--> | <!--n:ro_dza_air_p95-->25<!--/n--> |
| Nigeria | <!--n:ro_nga_sea_mean-->62<!--/n--> | <!--n:ro_nga_sea_p95-->89<!--/n--> | <!--n:ro_nga_sea_buf-->27<!--/n--> | <!--n:ro_nga_port-->33%<!--/n--> | <!--n:ro_nga_tail-->58%<!--/n--> | <!--n:ro_nga_air_mean-->13<!--/n--> | <!--n:ro_nga_air_p95-->25<!--/n--> |
| Peru | <!--n:ro_per_sea_mean-->48<!--/n--> | <!--n:ro_per_sea_p95-->62<!--/n--> | <!--n:ro_per_sea_buf-->13<!--/n--> | <!--n:ro_per_port-->15%<!--/n--> | <!--n:ro_per_tail-->24%<!--/n--> | <!--n:ro_per_air_mean-->10<!--/n--> | <!--n:ro_per_air_p95-->16<!--/n--> |
| Italy | <!--n:ro_ita_sea_mean-->46<!--/n--> | <!--n:ro_ita_sea_p95-->61<!--/n--> | <!--n:ro_ita_sea_buf-->15<!--/n--> | <!--n:ro_ita_port-->22%<!--/n--> | <!--n:ro_ita_tail-->36%<!--/n--> | <!--n:ro_ita_air_mean-->10<!--/n--> | <!--n:ro_ita_air_p95-->15<!--/n--> |

"Port + customs" = origin port dwell + destination port and customs dwell (sea) or airport and customs dwell (air
columns not shown). Port and customs time is <!--n:ro_port_mean-->15-37%<!--/n--> of the average sea lead time but
<!--n:ro_port_tail-->24-58%<!--/n--> of the lead time of the slowest 5% of shipments: it drives the buffer more than
the distance does. Algeria stands out: its median container dwell (16.7 days) and median air-cargo dwell (11.8 days)
are both LPI 2.0 values, not assumptions.

**(b) Ship dates.** For each market a synthetic rollout of 6 sites per week for 12 weeks; each week's kits travel as
one shipment and share the port and customs time, the inland leg is drawn per site. Shipping P95 days before the
need date (a buffer of <!--n:ro_buf_sea-->13-29 days<!--/n--> beyond the average by sea, <!--n:ro_buf_air-->5-11 days<!--/n-->
by air) gives <!--n:ro_oos_p95_sea-->94.7-95.2%<!--/n--> of sites on time by sea in an independent simulation
(<!--n:ro_oos_p95_air-->94.9-95.1%<!--/n--> by air). Planning on the average transit time gives only
<!--n:ro_ontime_mean_sea-->54-61%<!--/n--> on time by sea and <!--n:ro_ontime_mean_air-->54-66%<!--/n--> by air, because
the distributions are right-skewed. Shared shipments matter too: with P95 planning, all six sites of a week are on time
less often than 95% (see `results/rollout/ship_date_plan.csv`).

**The LPI-to-distribution mapping is an assumption.** Only quartiles and a mean are published, so each container dwell
is a lognormal through the log-quartiles; it reproduces the published means within <!--n:ro_fit_err-->15%<!--/n-->
(`results/rollout/dwell_fit_check.csv`). The Data360 file labels the metric code `M` "Median", but it is the mean
(the LPI 2.0 methodology lists mean, median and IQR, and `M` >= `MD` for <!--n:ro_m_ge_md-->162 of 163<!--/n--> economies in 2024).
The aviation P80-P20 range is not used because for <!--n:ro_av_bad-->16 of 121<!--/n--> economies the mean is below
0.2 x (P80 - P20), impossible for one non-negative sample.
Change in the sea P95 across the seven markets:

| assumption changed | P95 change |
|---|---|
| destination dwell x 1.25 / x 0.75 | <!--n:ro_sens_dwell_up-->+1 to +11 days<!--/n--> / <!--n:ro_sens_dwell_down-->-11 to -1 days<!--/n--> |
| lognormal matched to median + mean instead of quartiles | <!--n:ro_sens_fit-->-3 to +13 days<!--/n--> |
| LPI 2.0 data for 2023 instead of 2024 | <!--n:ro_sens_year-->-5 to +5 days<!--/n--> |
| sea speed 14 kn instead of 16 | <!--n:ro_sens_speed14-->+1 to +4 days<!--/n--> |
| no transshipment | <!--n:ro_sens_transship-->-16 to -8 days<!--/n--> |
| Cape of Good Hope instead of Suez (Algeria / Italy) | <!--n:ro_sens_cape_dza-->+14 days<!--/n--> / <!--n:ro_sens_cape_ita-->+16 days<!--/n--> |

Dropping transshipment moves the P95 about as much as or more than any dwell assumption, and it is the weakest input (all-origin averages).
As a plausibility check of the survey-based LPI, the 2023 customs score and the measured 2024 median container dwell
have a Spearman correlation of <!--n:ro_rho-->-0.45<!--/n--> over <!--n:ro_rho_n-->108<!--/n--> economies: related,
but far from interchangeable, which is why the measured dwell times are used rather than a score-to-days mapping.

**(c) Mode policy.** Cost per site (USD, ILLUSTRATIVE) for the cheapest plan of each policy that is on time at least
95% of the time: freight + handling + inland delivery + holding of equipment value in transit and while waiting early
(25%/yr) + storage + expedites. "Split" flies the radio and baseband pallets (67% of kit value, 16% of its weight) and
ships the rest by sea; "sea + expedite" re-ships the whole kit by air (30% premium) when the sea shipment has not left
the destination port by the last date an air shipment would still arrive at P95. Planned shipments, sea or air, go once
a week per market, so a week's sites share the per-shipment fees; an expedite is a one-site shipment.

| market | all sea | split | sea + expedite | all air |
|---|---|---|---|---|
| Indonesia | <!--n:ro_idn_cost_sea-->2,295<!--/n--> | <!--n:ro_idn_cost_split-->3,073<!--/n--> | <!--n:ro_idn_cost_sea_expedite-->3,073<!--/n--> | <!--n:ro_idn_cost_air-->5,777<!--/n--> |
| Pakistan | <!--n:ro_pak_cost_sea-->3,024<!--/n--> | <!--n:ro_pak_cost_split-->3,940<!--/n--> | <!--n:ro_pak_cost_sea_expedite-->4,292<!--/n--> | <!--n:ro_pak_cost_air-->7,966<!--/n--> |
| Iraq | <!--n:ro_irq_cost_sea-->3,535<!--/n--> | <!--n:ro_irq_cost_split-->4,212<!--/n--> | <!--n:ro_irq_cost_sea_expedite-->4,497<!--/n--> | <!--n:ro_irq_cost_air-->8,658<!--/n--> |
| Algeria | <!--n:ro_dza_cost_sea-->5,012<!--/n--> | <!--n:ro_dza_cost_split-->5,128<!--/n--> | <!--n:ro_dza_cost_sea_expedite-->6,388<!--/n--> | <!--n:ro_dza_cost_air-->9,925<!--/n--> |
| Nigeria | <!--n:ro_nga_cost_sea-->4,865<!--/n--> | <!--n:ro_nga_cost_split-->5,374<!--/n--> | <!--n:ro_nga_cost_sea_expedite-->6,269<!--/n--> | <!--n:ro_nga_cost_air-->10,912<!--/n--> |
| Peru | <!--n:ro_per_cost_sea-->3,602<!--/n--> | <!--n:ro_per_cost_split-->4,751<!--/n--> | <!--n:ro_per_cost_sea_expedite-->5,723<!--/n--> | <!--n:ro_per_cost_air-->11,583<!--/n--> |
| Italy | <!--n:ro_ita_cost_sea-->3,770<!--/n--> | <!--n:ro_ita_cost_split-->4,263<!--/n--> | <!--n:ro_ita_cost_sea_expedite-->4,798<!--/n--> | <!--n:ro_ita_cost_air-->8,649<!--/n--> |

All sea with a P95 ship date is cheapest in <!--n:ro_best_sea-->7/7<!--/n--> markets under these inputs. The reason is
arithmetic, not a law: a day of buffer costs about USD 35 per site (holding plus storage), while flying even the small
critical pallets costs more than the holding it saves, and an expedited kit costs several thousand USD. Split comes
closest where the sea leg is longest (Algeria). A cost for late sites (not modelled) or a higher holding rate
(`config/rollout.json`) could change the ranking.

**(d) Container loading.** Pallets and crates have dimensions, weight, "may ride on top" and a top-load limit. Stacks
are at most two high, only identical footprints stack, and a stack must pass under the door with 5 cm to spare.
Container interiors, door apertures and payloads are the typical ISO values tabulated on Wikipedia's *Intermodal
container* page (accessed 2026-10-21; real boxes differ by a few cm). Heuristic: greedy stacking, then first-fit
decreasing into shelves across the width, choosing for each new shelf the orientation that fills the width best.
An independent checker asserts that every unit is loaded exactly once, inside the floor, without overlaps, overweight,
overheight or illegal stacking. Lower bound per container type = max(volume, weight, floor-area bound).
One weekly batch (6 sites, 42 units) loads as <!--n:ro_load_mix-->1 x 40HC + 1 x 20GP<!--/n--> at
<!--n:ro_load_fill-->49%<!--/n--> volume fill: this cargo runs out of floor, never of weight. Two boxes is the minimum:
under the stacking rules the batch needs at least 21 pallet-size floor stacks and 3 crate stacks, 29.0 m2 of floor,
more than the 28.0 m2 of a 40-foot box. The floor bound encodes exactly this argument (units that may not ride on top
need floor; each can carry at most one unit; the rest pair up at best). On 200 random instances (40HC) the heuristic
equals the lower bound in <!--n:ro_load_eq-->105/200<!--/n--> and is never more than <!--n:ro_load_gap-->1<!--/n--> box
above it; whether such a gap is the heuristic's or the bound's is not resolved (the bound ignores the 33 cm of width a
pair of 1.0 m pallets leaves unused). A detail the model surfaced: with the tabulated 11.998 m interior, ten 1.2 m pallet
rows miss a 40-foot box by 2 mm.

**(e) Consolidation frequency (sea, totals over the seven markets, 72 sites each).**

| sailing every | containers | load per sailing | volume fill | mean early arrival (d) | mean cost per site (USD) |
|---|---|---|---|---|---|
| week | <!--n:ro_cons_1_containers-->168<!--/n--> | <!--n:ro_cons_1_mix-->1 x 40HC + 1 x 20GP<!--/n--> | <!--n:ro_cons_1_fill-->49%<!--/n--> | <!--n:ro_cons_1_early-->21<!--/n--> | <!--n:ro_cons_1_cost-->3,727<!--/n--> |
| 2 weeks | <!--n:ro_cons_2_containers-->126<!--/n--> | <!--n:ro_cons_2_mix-->3 x 40GP<!--/n--> | <!--n:ro_cons_2_fill-->52%<!--/n--> | <!--n:ro_cons_2_early-->24<!--/n--> | <!--n:ro_cons_2_cost-->3,650<!--/n--> |
| 4 weeks | <!--n:ro_cons_4_containers-->126<!--/n--> | <!--n:ro_cons_4_mix-->5 x 40HC + 1 x 20GP<!--/n--> | <!--n:ro_cons_4_fill-->51%<!--/n--> | <!--n:ro_cons_4_early-->31<!--/n--> | <!--n:ro_cons_4_cost-->3,849<!--/n--> |

Each sailing is planned to reach the earliest need date of its batch at P95, so later weeks wait at an in-country
warehouse. Fortnightly sailings are cheapest in <!--n:ro_cons_best-->7/7<!--/n--> markets: they save a quarter of the
boxes, and monthly sailings save no further boxes while adding about a week of waiting. Even weekly sailings arrive about
three weeks early on average: that is what a 95% on-time promise costs with these lead-time tails.

**Order-of-magnitude references for the ILLUSTRATIVE rates** (accessed 2026-10-21; not used as inputs): Drewry World
Container Index, 15 Oct 2026, composite USD 4,485 and Shanghai-Genoa USD 3,945 per 40 ft
(<https://www.drewry.co.uk/supply-chain-advisors/supply-chain-expertise/world-container-index-assessed-by-drewry>);
WorldACD via AJOT, Hong Kong-Europe air spot USD 4.96/kg in week 31 of 2026
(<https://www.ajot.com/news/weekly-air-cargo-trends-july-27-to-august-2-2026>). Air chargeable weight uses the IATA
general-cargo volume rule of 6,000 cm3 per kg (express couriers often use 5,000). The 16 kn sea speed is an assumption,
not a cited figure (slow-steaming liner services typically sail in the mid-teens of knots); the table above shows 14 kn.

## Tests and mutation testing

`pytest` covers: the Incoterms split cell by cell against a hand-typed table (12 rules x 12 elements),
insurance obligations, mode restrictions, cost conservation, GST and duty arithmetic by hand,
simulator invariants (inventory balance, pipeline = scheduled arrivals, no negative on-hand, backorders
charged), exact deterministic-demand timing, analytic-vs-simulation, MILP = brute force, routing
geometry, level-constraint feasibility, LP feasibility, dual = finite difference, and this README. Part 5
(`tests/test_rollout.py`): lognormal fits, analytic moments of every component and of the summed lead time vs
Monte Carlo, the market-selection rule re-derived from the committed shares, ship-date quantile coverage out of
sample, container loading (every unit exactly once, no overweight, no overheight, no overlap, lower bound <=
heuristic, plus hand-made infeasible plans the checker must reject), and the cost arithmetic of each mode policy.

`tools/mutation_test.py` breaks the code or configuration on purpose (wrong payer cells, dropped
insurance from customs value, receipts before demand, off-by-one lead time, one-sided MILP
linearisation, LP rows removed, ...) and checks that the suite fails:
**<!--n:mut_caught-->43/43<!--/n--> mutants caught** (details in `results/mutation_report.json`).
The first run caught 20/25 (`results/mutation_report_round1.json`): one mutant was equivalent
(it re-ordered two additions; replaced by a real timing change) and four were genuine test gaps -
VAT base without duty (Singapore duty is 0, so only an EU-style test can see it), the (r,Q) "+1 batch",
the 365-day holding convention, and the size-L floor-level rule. Tests were added for each, then
the full set was re-run.

Part 5 added 18 mutants: wrong lognormal fits, a mixture-variance error, a dropped port-call allowance,
unconditioned transshipment time, a planning quantile that rounds down, one inland leg shared by a week's sites,
a load checker blind to overlaps, ignored top-load and payload limits, a floor bound that forgets what carriers
hold, split legs planned at p instead of sqrt(p), an expedite that never arrives first, a 360-day year, chargeable
weight without volume, a market rule that picks the smallest partner, consolidation without slack, and a config
switch to another dwell fit. All 18 were caught on their first run. That run also exposed a tool bug: on a Windows
checkout with CRLF line endings, 8 of the original 25 mutants (multi-line patterns) were silently skipped as
"invalid", so a score of 35/35 looked clean. The tool now matches CRLF files as well and prints the number of
invalid mutants next to the score; the score above is from a run with none.

## Limitations

* Synthetic warehouse and orders; one-SKU-per-slot; no replenishment moves, congestion, batching
  or zoning; routing is S-shape / return only (no optimal TSP routing).
* Slotting heuristic and MILP optimise a proxy objective; gains are measured by simulation, but
  only for the generator's demand structure.
* Inventory: single echelon, single SKU, normal demand, independent lead times; no capacity limits,
  no multi-sourcing (e.g. ocean + air expedite), no lost sales.
* Landed cost: illustrative rates; simplified valuation (no royalties/assists, no EU VAT incidental
  costs, no US MPF/HMF or trade-remedy duties); Incoterms modelled from public summaries, not the ICC text.
* Reverse network: illustrative costs and yields; single period; no stochastic return volumes.
* Rollout (Part 5): lead-time components are independent (a congested port slows both dwell and transshipment in
  reality); dwell distributions are lognormals through three or four published statistics, national averages for
  all goods, not telecom equipment through one terminal; transshipment uses all-origin averages, which probably
  overstates it for China-origin cargo; the ocean leg is distance / speed with an illustrative allowance, not a
  carrier schedule (blank sailings, rollover and weekly cut-offs are not modelled); air assumes one hub transfer and
  no capacity limit, and ignores dangerous-goods rules for batteries; no cost of a late site, so the policy ranking
  only compares logistics cost at an equal on-time target; container loading is a heuristic on box-shaped units with
  two tiers and no load-securing or axle-load rules; the market rule depends on the 2024-2025 window (reported).

## Layout

```
glops/        slotting.py inventory.py landed.py reverse.py rollout.py public_data.py experiments.py report.py
config/       incoterms_2020.json rates_illustrative.json duty_regimes_example.json
              inventory_scenario.json reverse_network.json rollout.json
data/public/  container_throughput_monthly.csv (+ .meta.json, LICENCE_SODL_v1.0.txt)
data/public/rollout/  comtrade_cn_hs851761_partner_shares.csv lpi_indicators.csv airports.csv sea_routes.csv
              sources.meta.json (access date, SHA-256 of the raw downloads)
tools/        fetch_public_data.py fetch_rollout_data.py mutation_test.py
results/      generated by run_all.py (summary.json is the source of every README number)
```

**Public data attribution.** Contains information from *Container Throughput, Monthly* accessed on
2026-10-13 from data.gov.sg (Maritime and Port Authority of Singapore) which is made available under
the terms of the Singapore Open Data Licence version 1.0 <https://data.gov.sg/open-data-licence>.
Use here does not imply any endorsement by the data provider.

Part 5 (all accessed 2026-10-21):
* The World Bank: *Logistics Performance Indicators (LPI) 2.0* (2023-2024) and *Logistics Performance Index (LPI)*
  2023 survey scores: Data360 bulk files, <https://lpi.worldbank.org/>, licensed under CC BY 4.0
  (<https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets>). Data sources of the dwell times: maritime
  data from a global shipping line and MarineTraffic, aviation data from Cargo iQ. The committed extract
  (`lpi_indicators.csv`) keeps a subset of indicators and drops regional aggregates; values are unchanged.
* Source: UN Comtrade (<https://comtradeplus.un.org/>), China's exports of HS 851761, 2024 and 2025, public preview
  API. Only derived partner shares of the top 40 partners are committed: UN Comtrade's re-dissemination guidance
  allows publication of limited amounts in a few tables and treats derived shares as transformed data
  (<https://uncomtrade.org/docs/re-dissemination/>); raw trade values are not redistributed.
* Airports: OurAirports, <https://ourairports.com/data/> (public domain).
* Sea distances: computed with `searoute` 1.6.0 (Apache-2.0, <https://github.com/genthalili/searoute-py>), whose
  network derives from Eurostat SeaRoute (EUPL-1.2), based on the ORNL Global Shipping Lane Network. `searoute` is
  not in `requirements.txt` (its sdist does not build on Windows with a non-UTF-8 locale); CI reads the committed
  distances.
* Container dimensions: Wikipedia, *Intermodal container* (only factual dimensions are used).

## 中文摘要

面向通信/网络设备全球供应链的物流与仓储运营分析，个人作品集项目（2026 年 10 月，借助 AI 辅助完成）。
四部分：(1) 仓库储位优化——合成仓库与订单，启发式交换搜索在留出订单上相对随机储位减少拣货行走
<!--n:slot_heur_vs_random-->49.4% (95% CI 48.7-50.2)<!--/n-->，相对 ABC 分类减少 <!--n:slot_heur_vs_abc-->21.2% (95% CI 20.6-21.8)<!--/n-->，小实例用 HiGHS MILP 求精确解对照；
(2) 库存策略仿真——海运与空运补货下 base-stock / (s,S) / (r,Q) 的服务水平-成本曲线，仿真与正态需求解析公式对照
<!--n:inv_val_pass-->20/20<!--/n--> 项在 3 个标准误内；(3) 按 Incoterms 2020 计算到岸成本，责任划分对照 ICC 等公开资料核查并注明来源，
新加坡 GST 9% 已核实，其余费率为**示意值**；(4) 退货/翻新/回收网络 LP 及敏感性分析；(5) 海外交付物流：从深圳把无线基站设备
运到 7 个海外市场（按 UN Comtrade 中国基站出口份额选出），用世界银行 LPI 2.0 的真实港口与清关停留时间估计门到站点时效，
按平均时效排发货日只有 <!--n:ro_ontime_mean_sea-->54-61%<!--/n--> 的站点准时（海运），并比较海运、空运、混合方式的成本、装箱方案与拼箱频率。
数据除新加坡港口月度集装箱吞吐量（新加坡开放数据许可证 v1.0）及第 5 部分注明的公开数据外均为合成；不构成税务或法律意见。

## License

Code: MIT (see `LICENSE`). The file in `data/public/` is published by MPA via data.gov.sg under the Singapore Open Data Licence v1.0 (`data/public/LICENCE_SODL_v1.0.txt`); it is not covered by the MIT licence. The files in `data/public/rollout/` keep their sources' terms (World Bank CC BY 4.0, UN Comtrade-derived shares, OurAirports public domain; see the attribution above) and are not covered by the MIT licence.
