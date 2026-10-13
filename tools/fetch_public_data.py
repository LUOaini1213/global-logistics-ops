"""Download the one public dataset used (optional enrichment) into data/public/.

Dataset : Container Throughput, Monthly (Maritime and Port Authority of Singapore)
ID      : d_da030f7028200d19ffcbe4a2d71af39c on data.gov.sg
Licence : Singapore Open Data Licence v1.0 (text saved in data/public/LICENCE_SODL_v1.0.txt)

run_all.py only reads the committed CSV, so the pipeline stays offline-reproducible.
Re-run this script to refresh the snapshot (numbers in results/ will then change).
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "public"
DATASET = "d_da030f7028200d19ffcbe4a2d71af39c"
API = "https://data.gov.sg/api/action/datastore_search?resource_id={}&limit={}&offset={}"


def fetch_all() -> list[dict]:
    rows, offset = [], 0
    while True:
        req = urllib.request.Request(API.format(DATASET, 500, offset),
                                     headers={"User-Agent": "glops-portfolio/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            res = json.load(r)["result"]
        rows += res["records"]
        offset += len(res["records"])
        if not res["records"] or offset >= res["total"]:
            return rows


def main() -> int:
    rows = fetch_all()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "container_throughput_monthly.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["month", "container_throughput_thousand_teu"])
        for r in sorted(rows, key=lambda r: r["month"]):
            w.writerow([r["month"], r["container_throughput"]])
    meta = {
        "dataset": "Container Throughput, Monthly",
        "dataset_id": DATASET,
        "publisher": "Maritime and Port Authority of Singapore (via data.gov.sg)",
        "url": f"https://data.gov.sg/datasets/{DATASET}/view",
        "accessed": dt.date.today().isoformat(),
        "rows": len(rows),
        "unit_note": "Values are TEU in thousands. Inferred, not stated on the dataset page: the 2024 monthly values sum to 41,124.0, matching SingStat's 2024 total of 41,124.1 thousand TEU. Latest month is a preliminary estimate per the publisher.",
        "licence": "Singapore Open Data Licence version 1.0, https://data.gov.sg/open-data-licence",
        "attribution": f"Contains information from Container Throughput, Monthly accessed on {dt.date.today().isoformat()} from data.gov.sg (Maritime and Port Authority of Singapore) which is made available under the terms of the Singapore Open Data Licence version 1.0 https://data.gov.sg/open-data-licence",
    }
    with open(OUT / "container_throughput_monthly.meta.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"saved {len(rows)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
