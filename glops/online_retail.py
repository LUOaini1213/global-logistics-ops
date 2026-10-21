"""Real order data: UCI "Online Retail II" -> small derived tables (data/public/online_retail_ii/).

Source  : Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
          https://doi.org/10.24432/C5CG6D (page https://archive.ics.uci.edu/dataset/502/online+retail+ii,
          accessed 2026-10-21). Licence CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
What    : every invoice line of a UK online giftware retailer (many customers are wholesalers),
          2009-12-01 to 2011-12-09, 1,067,371 rows in two workbook sheets.

The derived tables are an adaptation of that dataset (cleaned, filtered, aggregated as below).
They are CC BY 4.0, not MIT. ``tools/fetch_online_retail.py`` downloads the raw file and calls
``derive``; run_all.py and CI only read the committed derived tables.

Stock codes are normalised first: surrounding spaces removed, letters upper-cased ("85123a" and
"85123A" are the same product). Then every row is checked against these rules, in this order,
and counted under the first rule it fails:
  1. duplicate_sheet_overlap  the sheets overlap on 2010-12-01..2010-12-09: the second sheet's
                              copies of invoices already in the first sheet are dropped (``derive``
                              verifies the two copies are identical row for row)
  2. cancellation             invoice number starts with "C"
  3. missing_stock_code       empty stock code
  4. non_product_code         a code in NON_PRODUCT or starting with "GIFT_" (postage, carriage,
                              packing, manual entries, discounts, samples, adjustments, bad debt,
                              bank/marketplace fees, commission, test items, gift vouchers)
  5. non_positive_quantity    quantity <= 0 (all are zero-price stock write-offs: "damaged", ...)
  6. non_positive_price       unit price <= 0
Repeat lines of the same product within one invoice are merged into one order line (one pick;
quantities added). An order is an invoice.
"""
from __future__ import annotations

import csv
import datetime as dt
from collections import Counter
from pathlib import Path

import numpy as np

from .common import DATA, read_json, write_json

DERIVED = DATA / "public" / "online_retail_ii"
RAW_CSV = DATA / "raw" / "online_retail_ii" / "online_retail_II.csv"
RAW_HEADER = ["sheet", "Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price",
              "Customer ID", "Country"]

FIRST_DAY, SPLIT_DAY, LAST_DAY = "2009-12-01", "2010-12-01", "2011-12-09"   # train = [FIRST, SPLIT)
N_SKU = 1000

RULES = ("duplicate_sheet_overlap", "cancellation", "missing_stock_code", "non_product_code",
         "non_positive_quantity", "non_positive_price")
# Every non-standard code in the file was inspected (python tools/fetch_online_retail.py prints
# the counts). Codes kept as products although non-standard: DCGS... (items sold via the
# retailer's web shop), SP1002, PADS. Two 5-digit codes are services, not products.
NON_PRODUCT = {
    "POST": "postage", "DOT": "web-shop postage", "C2": "carriage", "C3": "carriage (blank)",
    "23444": "next day carriage", "23574": "packing charge",
    "M": "manual entry", "D": "discount", "S": "samples", "B": "bad-debt adjustment",
    "ADJUST": "manual adjustment", "ADJUST2": "manual adjustment",
    "BANK CHARGES": "bank charges", "AMAZONFEE": "marketplace fee", "CRUK": "charity commission",
    "TEST001": "test product", "TEST002": "test product",
    "GIFT": "gift voucher", "22016": "gift voucher (its only priced row)",
}
NON_PRODUCT_PREFIX = "GIFT_"           # GIFT_0001_10 ... GIFT_0001_90: gift vouchers

# Part B (inventory replay): SKU groups chosen from training-period statistics only
ADI_STEADY_MAX = 1.32     # Syntetos-Boylan cut-off: sold on > ~76% of trading days
ADI_INTERMITTENT_MIN = 2.0  # sold on at most half of the trading days
N_PER_GROUP = 10


def norm_code(code: str) -> str:
    return code.strip().upper()


def is_product(code: str) -> bool:
    return code not in NON_PRODUCT and not code.startswith(NON_PRODUCT_PREFIX)


def rule_for(row: dict, first_sheet_invoices: set) -> str | None:
    """Name of the first cleaning rule this raw row fails, or None if it is kept."""
    code = norm_code(row["StockCode"])
    if row["sheet"] == "1" and row["Invoice"] in first_sheet_invoices:
        return "duplicate_sheet_overlap"
    if row["Invoice"].startswith("C"):
        return "cancellation"
    if not code:
        return "missing_stock_code"
    if not is_product(code):
        return "non_product_code"
    if int(row["Quantity"]) <= 0:
        return "non_positive_quantity"
    if float(row["Price"]) <= 0:
        return "non_positive_price"
    return None


def clean(open_rows) -> dict:
    """Apply the rules to raw rows. ``open_rows()`` must return a fresh iterator of row dicts
    (the file is streamed twice: once for the invoice sets of each sheet, once to filter).

    Returns orders {invoice: {"date": "YYYY-MM-DD", "lines": {code: quantity}}}, the removal
    count per rule, the trading days (any recorded row) and a few audit counts."""
    sheets: dict = {"0": set(), "1": set()}
    for r in open_rows():
        sheets[r["sheet"]].add(r["Invoice"])
    first = sheets["0"]
    overlap = first & sheets["1"]
    copies = {"0": Counter(), "1": Counter()}
    removed = dict.fromkeys(RULES, 0)
    orders: dict = {}
    descriptions: Counter = Counter()      # (code, description) over kept training rows
    trading = set()
    n_rows = n_norm = n_merged = n_multi_date = 0
    by_code: Counter = Counter()           # removed non-product rows per code
    for r in open_rows():
        n_rows += 1
        if r["Invoice"] in overlap:
            copies[r["sheet"]][tuple(r[k] for k in RAW_HEADER[1:])] += 1
        rule = rule_for(r, first)
        if rule != "duplicate_sheet_overlap":
            trading.add(r["InvoiceDate"][:10])
        code = norm_code(r["StockCode"])
        if rule is not None:
            removed[rule] += 1
            if rule == "non_product_code":
                by_code[code] += 1
            continue
        n_norm += code != r["StockCode"]           # kept rows whose code was normalised
        day = r["InvoiceDate"][:10]
        o = orders.setdefault(r["Invoice"], {"date": day, "lines": {}})
        if day != o["date"]:
            n_multi_date += 1
            o["date"] = min(o["date"], day)
        if code in o["lines"]:
            n_merged += 1
        o["lines"][code] = o["lines"].get(code, 0) + int(r["Quantity"])
        if day < SPLIT_DAY:
            descriptions[code, r["Description"].strip()] += 1
    if copies["0"] != copies["1"]:
        raise ValueError("the two sheets disagree on the overlapping invoices")
    kept = n_rows - sum(removed.values())
    return {"orders": orders, "removed": removed, "trading_days": sorted(trading),
            "descriptions": descriptions, "non_product_rows_by_code": dict(sorted(by_code.items())),
            "rows_in": n_rows, "rows_kept": kept, "codes_normalised": n_norm,
            "repeat_lines_merged": n_merged, "invoices_spanning_dates": n_multi_date,
            "overlap_invoices": len(overlap)}


def read_raw(path: Path):
    """open_rows callable for ``clean`` over the raw CSV written by tools/fetch_online_retail.py."""
    def open_rows():
        with open(path, encoding="utf-8", newline="") as f:
            rd = csv.reader(f)
            if next(rd) != RAW_HEADER:
                raise ValueError(f"unexpected header in {path}")
            for row in rd:
                yield dict(zip(RAW_HEADER, row))
    return open_rows


def top_skus(orders: dict, n: int = N_SKU) -> tuple[list, Counter]:
    """The n codes on most training order lines; ties broken by stock code."""
    lines = Counter(c for o in orders.values() if o["date"] < SPLIT_DAY for c in o["lines"])
    return sorted(lines, key=lambda c: (-lines[c], c))[:n], lines


def calendar(first: str = FIRST_DAY, last: str = LAST_DAY) -> list[str]:
    d0, d1 = dt.date.fromisoformat(first), dt.date.fromisoformat(last)
    return [(d0 + dt.timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]


def demand_stats(units: np.ndarray, trading: np.ndarray, train: np.ndarray) -> dict:
    """Training-period statistics of one SKU's daily units (calendar-day arrays).

    adi      trading days / trading days with demand (average demand interval)
    cv2      squared coefficient of variation of the non-zero daily quantities
    mean/sd  over all calendar days (closed days count as zero demand), used by the normal model
    """
    x = units[train & trading]
    pos = x[x > 0]
    cal = units[train]
    return {"adi": float(len(x) / len(pos)) if len(pos) else float("inf"),
            "cv2": float(pos.var() / pos.mean() ** 2) if len(pos) else float("nan"),
            "demand_days": int(len(pos)), "train_units": int(cal.sum()),
            "mean_daily": float(cal.mean()), "sd_daily": float(cal.std(ddof=1))}


def select_inventory_skus(stats: list[dict], n: int = N_PER_GROUP) -> dict:
    """Training-only rule. steady: ADI < 1.32; intermittent: ADI >= 2.0. In each group take the
    n SKUs on the most training order lines (ties: stock code)."""
    key = lambda s: (-s["train_lines"], s["stock_code"])
    steady = sorted((s for s in stats if s["adi"] < ADI_STEADY_MAX), key=key)[:n]
    inter = sorted((s for s in stats if s["adi"] >= ADI_INTERMITTENT_MIN), key=key)[:n]
    return {"steady": [s["stock_code"] for s in steady], "intermittent": [s["stock_code"] for s in inter]}


def _write_csv(path: Path, header: list, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def derive(raw_csv: Path, out_dir: Path) -> dict:
    """Raw CSV -> derived tables. Deterministic: same input bytes, same output bytes."""
    c = clean(read_raw(raw_csv))
    orders = c["orders"]
    top, lines = top_skus(orders)
    idx = {code: k for k, code in enumerate(top)}

    by_code: dict = {}
    for (code, text), n in c["descriptions"].items():
        if text:
            by_code.setdefault(code, []).append((-n, text))

    def desc(code):          # most frequent training description; ties: alphabetical
        return min(by_code[code])[1] if code in by_code else ""

    out_dir.mkdir(parents=True, exist_ok=True)
    split = {"train": [], "test": []}
    test_lines: Counter = Counter()
    cover = {"train": [0, 0], "test": [0, 0]}          # [lines on top SKUs, all lines]
    dropped = {"train": 0, "test": 0}
    for inv in sorted(orders, key=lambda i: (orders[i]["date"], i)):
        o = orders[inv]
        part = "train" if o["date"] < SPLIT_DAY else "test"
        skus = sorted(idx[code] for code in o["lines"] if code in idx)
        cover[part][0] += len(skus)
        cover[part][1] += len(o["lines"])
        if part == "test":
            test_lines.update(code for code in o["lines"] if code in idx)
        if skus:
            split[part].append((inv, o["date"], " ".join(map(str, skus))))
        else:
            dropped[part] += 1
    for part, rows in split.items():
        _write_csv(out_dir / f"orders_{part}.csv", ["invoice", "date", "skus"], rows)
    _write_csv(out_dir / "skus.csv", ["sku", "stock_code", "description", "train_lines", "test_lines"],
               [(k, code, desc(code), lines[code], test_lines[code]) for k, code in enumerate(top)])

    # ---- daily units per SKU on the calendar (Part B)
    days = calendar()
    day_idx = {d: i for i, d in enumerate(days)}
    units = np.zeros((len(top), len(days)), dtype=np.int64)
    for o in orders.values():
        for code, q in o["lines"].items():
            if code in idx:
                units[idx[code], day_idx[o["date"]]] += q
    open_days = set(c["trading_days"])
    trading = np.array([d in open_days for d in days])
    train = np.array([d < SPLIT_DAY for d in days])
    stats = []
    for k, code in enumerate(top):
        s = demand_stats(units[k], trading, train)
        stats.append({"sku": k, "stock_code": code, "train_lines": lines[code], **s})
    _write_csv(out_dir / "demand_stats.csv",
               ["sku", "stock_code", "train_lines", "demand_days", "adi", "cv2", "train_units",
                "mean_daily", "sd_daily"],
               [(s["sku"], s["stock_code"], s["train_lines"], s["demand_days"], f"{s['adi']:.6f}",
                 f"{s['cv2']:.6f}", s["train_units"], f"{s['mean_daily']:.6f}", f"{s['sd_daily']:.6f}")
                for s in stats])
    sel = select_inventory_skus(stats)
    chosen = sel["steady"] + sel["intermittent"]
    _write_csv(out_dir / "daily_demand.csv", ["date", "trading"] + chosen,
               [[d, int(trading[i])] + [int(units[idx[code], i]) for code in chosen]
                for i, d in enumerate(days)])

    n_lines = {p: cover[p][1] for p in cover}
    meta = {
        "dataset": "Online Retail II", "uci_id": 502,
        "page": "https://archive.ics.uci.edu/dataset/502/online+retail+ii",
        "download": "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip",
        "accessed": "2026-10-21",
        "licence": "CC BY 4.0, https://creativecommons.org/licenses/by/4.0/",
        "citation": "Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository. "
                    "https://doi.org/10.24432/C5CG6D",
        "changes": "Cleaned, filtered to the top 1,000 products of the training year, aggregated to "
                   "orders and daily units; see glops/online_retail.py.",
        "periods": {"train": [FIRST_DAY, SPLIT_DAY], "test": [SPLIT_DAY, LAST_DAY],
                    "_note": "train = [first, split), test = [split, last]"},
        "cleaning": {"rows_in": c["rows_in"], "removed": c["removed"], "rows_kept": c["rows_kept"],
                     "codes_normalised": c["codes_normalised"],
                     "repeat_lines_merged": c["repeat_lines_merged"],
                     "invoices_spanning_dates": c["invoices_spanning_dates"],
                     "overlap_invoices": c["overlap_invoices"],
                     "non_product_rows_by_code": c["non_product_rows_by_code"]},
        "orders": {"invoices": len(orders), "products": len({x for o in orders.values() for x in o["lines"]}),
                   "train_invoices": len(split["train"]) + dropped["train"],
                   "test_invoices": len(split["test"]) + dropped["test"],
                   "train_orders_kept": len(split["train"]), "test_orders_kept": len(split["test"]),
                   "train_orders_without_top_sku": dropped["train"],
                   "test_orders_without_top_sku": dropped["test"],
                   "train_products": len(lines),
                   "top_sku_min_train_lines": lines[top[-1]],
                   "first_excluded_train_lines": (sorted(lines.values(), reverse=True)[N_SKU]
                                                  if len(lines) > N_SKU else 0)},
        "coverage": {p: {"lines_on_top_skus": cover[p][0], "lines": n_lines[p],
                         "share": cover[p][0] / n_lines[p] if n_lines[p] else 0.0} for p in cover},
        "calendar": {"days": len(days), "trading_days": int(trading.sum()),
                     "closed_days": [d for d, t in zip(days, trading) if not t and
                                     dt.date.fromisoformat(d).weekday() != 5],
                     "saturdays_traded": [d for d, t in zip(days, trading)
                                          if t and dt.date.fromisoformat(d).weekday() == 5]},
        "inventory_selection": sel,
    }
    write_json(out_dir / "derivation.json", meta)
    return meta


# ------------------------------------------------------------------ loaders (committed tables)

def load_meta(d: Path = DERIVED) -> dict:
    return read_json(d / "derivation.json")


def _read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_skus(d: Path = DERIVED) -> list[dict]:
    return _read(d / "skus.csv")


def load_orders(part: str, d: Path = DERIVED) -> tuple[list, list, list]:
    """(invoices, dates, orders) where orders[i] is a sorted int array of SKU indices."""
    rows = _read(d / f"orders_{part}.csv")
    return ([r["invoice"] for r in rows], [r["date"] for r in rows],
            [np.array(r["skus"].split(), dtype=int) for r in rows])


def load_demand_stats(d: Path = DERIVED) -> list[dict]:
    out = []
    for r in _read(d / "demand_stats.csv"):
        out.append({"sku": int(r["sku"]), "stock_code": r["stock_code"], "train_lines": int(r["train_lines"]),
                    "demand_days": int(r["demand_days"]), "adi": float(r["adi"]), "cv2": float(r["cv2"]),
                    "train_units": int(r["train_units"]), "mean_daily": float(r["mean_daily"]),
                    "sd_daily": float(r["sd_daily"])})
    return out


def load_daily(d: Path = DERIVED) -> dict:
    rows = _read(d / "daily_demand.csv")
    codes = [k for k in rows[0] if k not in ("date", "trading")]
    return {"dates": [r["date"] for r in rows], "trading": np.array([int(r["trading"]) for r in rows], bool),
            "units": {c: np.array([float(r[c]) for r in rows]) for c in codes}}
