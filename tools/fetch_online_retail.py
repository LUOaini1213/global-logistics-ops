"""Download UCI "Online Retail II" and rebuild the small derived tables from it.

Dataset : Online Retail II, UCI Machine Learning Repository, dataset id 502
Page    : https://archive.ics.uci.edu/dataset/502/online+retail+ii
Licence : Creative Commons Attribution 4.0 International (CC BY 4.0)
Cite    : Chen, D. (2012). Online Retail II [Dataset]. UCI Machine Learning Repository.
          https://doi.org/10.24432/C5CG6D

    pip install -r requirements-fetch.txt        # openpyxl, only needed here
    python tools/fetch_online_retail.py          # download (~45 MB) + verify + convert + derive
    python tools/fetch_online_retail.py --zip PATH   # use an archive you already downloaded
    python tools/fetch_online_retail.py --derive-only  # rebuild derived tables from the raw CSV

The raw files go to data/raw/online_retail_ii/ (git-ignored, never committed):
  online+retail+ii.zip   the archive exactly as served by UCI      (SHA-256 checked)
  online_retail_II.xlsx  the workbook inside it                    (SHA-256 checked)
  online_retail_II.csv   both sheets as plain text, one row per    (SHA-256 checked)
                         workbook row, plus a 'sheet' column
The derived tables (data/public/online_retail_ii/) are what run_all.py and CI read, so the
pipeline stays offline. tests/test_real_orders.py rebuilds them from the raw CSV when it is
present and fails if a single byte differs.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "online_retail_ii"
URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ZIP_NAME, XLSX_NAME, CSV_NAME = "online+retail+ii.zip", "online_retail_II.xlsx", "online_retail_II.csv"
SHA256 = {
    ZIP_NAME: "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb",
    XLSX_NAME: "bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980",
    CSV_NAME: "0f0d60f9ddbf1f410eaf06bd518ae337d8ab6bcec02da76300598709dcd89f1a",
}
HEADER = ("Invoice", "StockCode", "Description", "Quantity", "InvoiceDate", "Price",
          "Customer ID", "Country")
SHEETS = ("Year 2009-2010", "Year 2010-2011")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check(path: Path) -> None:
    got = sha256(path)
    if got != SHA256[path.name]:
        raise SystemExit(f"SHA-256 mismatch for {path.name}: got {got}, expected {SHA256[path.name]}")
    print(f"ok  {path.name}  sha256={got}")


def download(dest: Path) -> None:
    req = urllib.request.Request(URL, headers={"User-Agent": "glops-portfolio/1.0"})
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)


def cell(v) -> str:
    """Deterministic text form of one workbook cell (openpyxl value)."""
    if v is None:
        return ""
    if isinstance(v, bool):
        raise TypeError("unexpected boolean cell")
    if isinstance(v, dt.datetime):
        return v.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)          # shortest round-trip form, platform independent
    return str(v)


def xlsx_to_csv(xlsx: Path, out: Path) -> int:
    import openpyxl  # only needed for this step (requirements-fetch.txt)

    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    if tuple(wb.sheetnames) != SHEETS:
        raise SystemExit(f"unexpected sheets {wb.sheetnames}")
    n = 0
    tmp = out.with_suffix(".part")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(("sheet",) + HEADER)
        for k, name in enumerate(SHEETS):
            rows = wb[name].iter_rows(values_only=True)
            if tuple(next(rows)) != HEADER:
                raise SystemExit(f"unexpected header in sheet {name!r}")
            for row in rows:
                if all(v is None for v in row):
                    continue
                w.writerow([str(k)] + [cell(v) for v in row])
                n += 1
    wb.close()
    tmp.replace(out)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--zip", type=Path, help="use this already-downloaded archive")
    ap.add_argument("--derive-only", action="store_true", help="skip download/convert")
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    zpath, xpath, cpath = RAW / ZIP_NAME, RAW / XLSX_NAME, RAW / CSV_NAME
    if not args.derive_only:
        if args.zip:
            shutil.copyfile(args.zip, zpath)
        elif not zpath.exists():
            print(f"downloading {URL}")
            download(zpath)
        check(zpath)
        with zipfile.ZipFile(zpath) as z:
            if z.namelist() != [XLSX_NAME]:
                raise SystemExit(f"unexpected archive members {z.namelist()}")
            with z.open(XLSX_NAME) as src, open(xpath, "wb") as dst:   # no extractall: no path tricks
                shutil.copyfileobj(src, dst)
        check(xpath)
        print("converting workbook to CSV (about a minute) ...")
        n = xlsx_to_csv(xpath, cpath)
        print(f"{n:,} data rows")
    check(cpath)
    sys.path.insert(0, str(ROOT))
    from glops import online_retail as orr   # noqa: E402

    report = orr.derive(cpath, orr.DERIVED)
    for k, v in report["cleaning"]["removed"].items():
        print(f"  removed {v:>8,}  {k}")
    print(f"derived tables written to {orr.DERIVED.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
