# global-logistics-ops

Logistics and warehouse operations analytics for a global network-equipment supply chain
(transceivers, cables, PSUs, line cards, radio units, small cells): **warehouse slotting**,
**inventory policy (ocean vs air)**, **landed cost by Incoterms 2020**, and a
**reverse-logistics (returns / refurbish / recycle) network**.

> **What this is.** A personal portfolio project, October 2026, built with AI assistance
> (code, tests and write-up were drafted with an AI coding assistant and checked by running
> them). It is not work for, or data from, any company.
>
> **Data.** All operational data is **synthetic** (documented generators, fixed seeds). All
> freight, handling, insurance, duty and VAT inputs except Singapore GST are **ILLUSTRATIVE**
> round numbers, not quotes. The only real data is one public dataset (Singapore monthly
> container throughput, Singapore Open Data Licence v1.0), used as a seasonality proxy.
> **Not tax, customs or legal advice.**

Every number below between the markers is written by `run_all.py` from `results/summary.json`
and checked by `tests/test_readme_and_data.py`; none is typed by hand.

## Reproduce

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Windows; use .venv/bin/python elsewhere
.venv/Scripts/python run_all.py          # ~5 min, writes results/ and results/report.html
.venv/Scripts/python -m pytest           # ~10 s
.venv/Scripts/python tools/mutation_test.py   # optional, ~5 min, writes results/mutation_report.json
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

## Tests and mutation testing

`pytest` covers: the Incoterms split cell by cell against a hand-typed table (12 rules x 12 elements),
insurance obligations, mode restrictions, cost conservation, GST and duty arithmetic by hand,
simulator invariants (inventory balance, pipeline = scheduled arrivals, no negative on-hand, backorders
charged), exact deterministic-demand timing, analytic-vs-simulation, MILP = brute force, routing
geometry, level-constraint feasibility, LP feasibility, dual = finite difference, and this README.

`tools/mutation_test.py` breaks the code or configuration on purpose (wrong payer cells, dropped
insurance from customs value, receipts before demand, off-by-one lead time, one-sided MILP
linearisation, LP rows removed, ...) and checks that the suite fails:
**<!--n:mut_caught-->25/25<!--/n--> mutants caught** (details in `results/mutation_report.json`).
The first run caught 20/25 (`results/mutation_report_round1.json`): one mutant was equivalent
(it re-ordered two additions; replaced by a real timing change) and four were genuine test gaps -
VAT base without duty (Singapore duty is 0, so only an EU-style test can see it), the (r,Q) "+1 batch",
the 365-day holding convention, and the size-L floor-level rule. Tests were added for each, then
the full set was re-run.

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

## Layout

```
glops/        slotting.py inventory.py landed.py reverse.py public_data.py experiments.py report.py
config/       incoterms_2020.json rates_illustrative.json duty_regimes_example.json
              inventory_scenario.json reverse_network.json
data/public/  container_throughput_monthly.csv (+ .meta.json, LICENCE_SODL_v1.0.txt)
tools/        fetch_public_data.py mutation_test.py
results/      generated by run_all.py (summary.json is the source of every README number)
```

**Public data attribution.** Contains information from *Container Throughput, Monthly* accessed on
2026-10-13 from data.gov.sg (Maritime and Port Authority of Singapore) which is made available under
the terms of the Singapore Open Data Licence version 1.0 <https://data.gov.sg/open-data-licence>.
Use here does not imply any endorsement by the data provider.

## 中文摘要

面向通信/网络设备全球供应链的物流与仓储运营分析，个人作品集项目（2026 年 10 月，借助 AI 辅助完成）。
四部分：(1) 仓库储位优化——合成仓库与订单，启发式交换搜索在留出订单上相对随机储位减少拣货行走
<!--n:slot_heur_vs_random-->49.4% (95% CI 48.7-50.2)<!--/n-->，相对 ABC 分类减少 <!--n:slot_heur_vs_abc-->21.2% (95% CI 20.6-21.8)<!--/n-->，小实例用 HiGHS MILP 求精确解对照；
(2) 库存策略仿真——海运与空运补货下 base-stock / (s,S) / (r,Q) 的服务水平-成本曲线，仿真与正态需求解析公式对照
<!--n:inv_val_pass-->20/20<!--/n--> 项在 3 个标准误内；(3) 按 Incoterms 2020 计算到岸成本，责任划分对照 ICC 等公开资料核查并注明来源，
新加坡 GST 9% 已核实，其余费率为**示意值**；(4) 退货/翻新/回收网络 LP 及敏感性分析。
数据除新加坡港口月度集装箱吞吐量（新加坡开放数据许可证 v1.0）外均为合成；不构成税务或法律意见。

## License

Code: MIT (see `LICENSE`). The file in `data/public/` is published by MPA via data.gov.sg under the Singapore Open Data Licence v1.0 (`data/public/LICENCE_SODL_v1.0.txt`); it is not covered by the MIT licence.
