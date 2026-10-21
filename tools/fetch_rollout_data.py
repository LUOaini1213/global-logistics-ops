"""Download the public inputs of Part 5 (overseas rollout) into data/public/rollout/.

run_all.py only reads the committed derived files, so CI never needs the network.
Re-running this script refreshes the snapshot (results will then change).

Sources (all accessed 2026-10-21; licences checked on that date):

1. UN Comtrade public preview API (no key), China (156) exports of HS 851761
   "base stations", partner breakdown, 2024 and 2025.
   Licence: UN Comtrade data are copyrighted by the UN. The re-dissemination FAQ
   (https://uncomtrade.org/docs/re-dissemination/) lists "Publication of limited amounts
   of UN Comtrade data in a few tables, graphs, visualization" as needing no permission,
   and treats derived metrics such as market shares as transformed data. So only a small
   derived table is committed: each partner's SHARE (%) of China's HS 851761 exports for
   the top partners, not the raw trade values. Cite as "UN Comtrade".
2. World Bank Logistics Performance Indicators 2.0 (LPI 2.0, 2023-2024; container and
   aviation import dwell times, transshipments) and the survey-based LPI 2023 scores,
   via the Data360 bulk files. Licence: CC BY 4.0.
3. OurAirports airports.csv (public domain), only the airports used.
4. Sea distances from the `searoute` package (Apache-2.0; network derived from Eurostat
   SeaRoute, EUPL-1.2, itself based on the ORNL Global Shipping Lane Network). Optional:
   it is only needed for --sea and is NOT in requirements.txt because its sdist fails to
   build on Windows with a non-UTF-8 locale. Install it separately:
       PYTHONUTF8=1 python -m pip install searoute==1.6.0

    python tools/fetch_rollout_data.py          # trade, LPI, airports
    python tools/fetch_rollout_data.py --sea    # also recompute sea distances
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import math
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "public" / "rollout"
CFG = ROOT / "config" / "rollout.json"
UA = {"User-Agent": "glops-portfolio/1.0 (personal research)"}

COMTRADE = ("https://comtradeapi.un.org/public/v1/preview/C/A/HS?reporterCode=156&period={year}"
            "&cmdCode=851761&flowCode=X")
COMTRADE_PARTNERS = "https://comtradeapi.un.org/files/v1/app/reference/partnerAreas.json"
LPI20 = "https://data360files.worldbank.org/data360-data/data/WB_LPI_20/WB_LPI_20.csv"
LPI_SURVEY = "https://data360files.worldbank.org/data360-data/data/WB_LPI/WB_LPI.csv"
OURAIRPORTS = "https://davidmegginson.github.io/ourairports-data/airports.csv"
TOP_N = 40

# UN M49 sub-regions (Africa split into Northern / Sub-Saharan as M49 does; "Western Asia"
# is reported as Middle East) for the partners that can reach the top 40.
REGION = {
    **{k: "South-eastern Asia" for k in ("IDN", "MYS", "VNM", "THA", "PHL", "KHM", "MMR", "SGP", "LAO", "TLS", "BRN")},
    **{k: "Southern Asia" for k in ("PAK", "BGD", "IRN", "AFG", "LKA", "NPL", "IND", "BTN", "MDV")},
    **{k: "Middle East (Western Asia)" for k in ("IRQ", "LBN", "ARE", "SAU", "TUR", "OMN", "YEM", "JOR", "SYR", "KWT", "QAT", "BHR", "ISR", "GEO", "ARM", "AZE", "CYP", "PSE")},
    **{k: "Northern Africa" for k in ("DZA", "MAR", "EGY", "TUN", "LBY", "SDN")},
    **{k: "Sub-Saharan Africa" for k in ("NGA", "CMR", "CIV", "UGA", "TZA", "ZWE", "COD", "ETH", "BFA", "COG", "MOZ", "SEN", "ZMB", "BEN", "KEN", "SLE", "GIN", "ZAF", "RWA", "TCD", "GHA", "AGO", "MLI", "NER", "TGO", "GAB", "MDG", "SOM", "SSD", "LBR", "MRT", "BWA", "NAM", "MWI", "BDI", "CAF", "GNQ", "DJI", "ERI", "GMB", "GNB", "LSO", "SWZ", "COM", "CPV", "STP", "SYC", "MUS")},
    **{k: "Latin America & Caribbean" for k in ("PER", "ARG", "CUB", "VEN", "ECU", "CHL", "BRA", "MEX", "COL", "BOL", "PRY", "URY", "PAN", "GTM", "HND", "NIC", "SLV", "CRI", "DOM", "HTI", "JAM", "TTO", "GUY", "SUR")},
    **{k: "Europe" for k in ("ITA", "UKR", "RUS", "AUT", "DEU", "FRA", "ESP", "GBR", "NLD", "POL", "BLR", "SRB", "HUN", "CZE", "ROU", "BEL", "GRC", "PRT", "SWE", "FIN", "NOR", "DNK", "CHE", "IRL", "BGR", "HRV", "SVK", "SVN", "LTU", "LVA", "EST", "ALB", "MKD", "BIH", "MNE", "MDA")},
    **{k: "Central Asia" for k in ("UZB", "KAZ", "KGZ", "TJK", "TKM")},
    **{k: "Eastern Asia" for k in ("HKG", "MNG", "KOR", "JPN", "PRK", "MAC")},
    **{k: "Oceania" for k in ("FJI", "PNG", "AUS", "NZL", "SLB", "VUT", "WSM", "TON")},
    **{k: "Northern America" for k in ("USA", "CAN")},
}

# World Bank regional / income aggregates present in the Data360 files (economies only are kept)
WB_AGGREGATES = {"EAS", "ECS", "HIC", "LCN", "LIC", "LMC", "LMY", "MEA", "MIC", "NAC", "SAS", "SSF",
                 "UMC", "WLD", "EAP", "ECA", "LAC", "MNA", "SSA"}

# LPI 2.0 indicator/metric -> column. Metric "M" is labelled "Median" in the Data360 file but
# is the MEAN: the LPI 2.0 Methodology and User Guide (World Bank, 2026) lists "mean, median,
# and interquartile range" for these indicators, and M >= MD for all but one economy in 2024.
LPI20_COLS = {
    ("CT_DT_M", "MD"): "ct_import_dwell_median", ("CT_DT_M", "M"): "ct_import_dwell_mean",
    ("CT_DT_M", "Q1"): "ct_import_dwell_q1", ("CT_DT_M", "Q3"): "ct_import_dwell_q3",
    ("CT_DT_X", "MD"): "ct_export_dwell_median", ("CT_DT_X", "M"): "ct_export_dwell_mean",
    ("CT_DT_X", "Q1"): "ct_export_dwell_q1", ("CT_DT_X", "Q3"): "ct_export_dwell_q3",
    ("AV_DT", "MD"): "air_import_dwell_median", ("AV_DT", "M"): "air_import_dwell_mean",
    ("AV_DT", "IQR_80_20"): "air_import_dwell_p80_minus_p20",
    ("TRS_NUM", "_Z"): "transshipments_mean", ("TRS_T", "_Z"): "transshipment_days_mean",
    ("LT_M", "MD"): "port_to_port_lead_time_median",
}


def get(url: str, timeout: int = 120, tries: int = 6) -> bytes:
    """GET with polite exponential back-off (the Comtrade preview API rate-limits with 429)."""
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code != 429 or k == tries - 1:
                raise
            time.sleep(10 * 2 ** k)
    raise RuntimeError("unreachable")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def fetch_comtrade(meta: dict) -> None:
    partners = {p["PartnerCode"]: p for p in json.loads(get(COMTRADE_PARTNERS))["results"]}
    vals: dict[str, dict[int, float]] = {}
    names: dict[str, str] = {}
    totals = {}
    for year in (2024, 2025):
        raw = get(COMTRADE.format(year=year))
        meta["raw_sha256"][f"comtrade_{year}"] = sha(raw)
        rows = json.loads(raw)["data"]
        keep = [r for r in rows if r["partnerCode"] != 0 and r["motCode"] == 0
                and r["customsCode"] == "C00" and r["partner2Code"] == 0 and r["flowCode"] == "X"]
        totals[year] = sum(r["primaryValue"] for r in keep)
        for r in keep:
            p = partners[r["partnerCode"]]
            iso = p.get("PartnerCodeIsoAlpha3") or str(r["partnerCode"])
            names[iso] = p["PartnerDesc"].strip()
            vals.setdefault(iso, {})[year] = r["primaryValue"]
    comb = {k: v.get(2024, 0.0) + v.get(2025, 0.0) for k, v in vals.items()}
    order = sorted(comb, key=lambda k: (-comb[k], k))[:TOP_N]
    out = []
    for rank, iso in enumerate(order, 1):
        v = vals[iso]
        out.append([rank, iso, names[iso], REGION.get(iso, "other"),
                    f"{100 * v.get(2024, 0.0) / totals[2024]:.2f}",
                    f"{100 * v.get(2025, 0.0) / totals[2025]:.2f}",
                    f"{100 * comb[iso] / (totals[2024] + totals[2025]):.2f}"])
    write_csv(OUT / "comtrade_cn_hs851761_partner_shares.csv",
              ["rank", "iso3", "partner", "region", "share_2024_pct", "share_2025_pct",
               "share_2024_2025_pct"], out)
    meta["comtrade"] = {"partners_with_trade": {str(y): sum(1 for v in vals.values() if y in v) for y in (2024, 2025)},
                        "top_n_committed": TOP_N}


def fetch_lpi(meta: dict) -> None:
    raw20, raws = get(LPI20, 300), get(LPI_SURVEY, 300)
    meta["raw_sha256"]["lpi20"], meta["raw_sha256"]["lpi_survey"] = sha(raw20), sha(raws)
    table: dict[str, dict] = {}
    for r in csv.DictReader(io.StringIO(raw20.decode("utf-8"))):
        col = LPI20_COLS.get((r["INDICATOR"], r["METRIC"]))
        if col and r["OBS_VALUE"] and r["TIME_PERIOD"] in ("2023", "2024"):
            row = table.setdefault(r["REF_AREA"], {"iso3": r["REF_AREA"], "name": r["REF_AREA_LABEL"]})
            row[f"{col}_{r['TIME_PERIOD']}"] = r["OBS_VALUE"]
    for r in csv.DictReader(io.StringIO(raws.decode("utf-8"))):
        if (r["TIME_PERIOD"] == "2023" and r["COMP_BREAKDOWN_1"] == "WB_LPI_SC"
                and r["INDICATOR"] in ("LPI_CTMS", "LPI_TML") and r["OBS_VALUE"]):
            row = table.setdefault(r["REF_AREA"], {"iso3": r["REF_AREA"], "name": r["REF_AREA_LABEL"]})
            row["lpi2023_" + ("customs" if r["INDICATOR"] == "LPI_CTMS" else "timeliness")] = r["OBS_VALUE"]
    cols = ["iso3", "name"] + [f"{c}_{y}" for c in LPI20_COLS.values() for y in ("2023", "2024")] + \
           ["lpi2023_customs", "lpi2023_timeliness"]
    rows = [[table[k].get(c, "") for c in cols] for k in sorted(table)
            if len(k) == 3 and k not in WB_AGGREGATES]
    write_csv(OUT / "lpi_indicators.csv", cols, rows)
    meta["lpi"] = {"countries": len(rows)}


def fetch_airports(meta: dict, codes: list[str]) -> None:
    raw = get(OURAIRPORTS, 300)
    meta["raw_sha256"]["ourairports"] = sha(raw)
    rows = {}
    for r in csv.DictReader(io.StringIO(raw.decode("utf-8"))):
        if r["iata_code"] in codes and r["type"] in ("large_airport", "medium_airport"):
            rows[r["iata_code"]] = [r["iata_code"], r["ident"], r["name"], r["iso_country"],
                                    f"{float(r['latitude_deg']):.4f}", f"{float(r['longitude_deg']):.4f}"]
    missing = set(codes) - set(rows)
    if missing:
        raise SystemExit(f"airports not found: {missing}")
    write_csv(OUT / "airports.csv", ["iata", "icao", "name", "iso2", "lat", "lon"],
              [rows[c] for c in codes])


def compute_sea(meta: dict, cfg: dict) -> None:
    import searoute as sr  # optional dependency, see module docstring
    from searoute.classes.passages import Passage

    pj = Path(sr.__file__).parent / "data" / "ports.geojson"      # [lon, lat] per UN/LOCODE
    ports = {f["properties"]["port"]: f["geometry"]["coordinates"]
             for f in json.loads(pj.read_text(encoding="utf-8"))["features"]}
    origin = cfg["origin"]["port_locode"]
    rows = []
    for m in cfg["markets"]:
        dest = m["port_locode"]
        o, d = ports[origin], ports[dest]
        out = []
        for scen, restr in (("base", [Passage.northwest]),
                            ("avoid_red_sea", [Passage.northwest, Passage.babalmandab])):
            r = sr.searoute(o, d, units="naut", restrictions=restr, return_passages=True)
            out.append((r.properties["length"], ";".join(sorted(r.properties.get("traversed_passages") or []))))
        gc = haversine_nm(o[1], o[0], d[1], d[0])
        rows.append([origin, dest, f"{o[1]:.4f}", f"{o[0]:.4f}", f"{d[1]:.4f}", f"{d[0]:.4f}",
                     f"{gc:.0f}", f"{out[0][0]:.0f}", out[0][1] or "-", f"{out[1][0]:.0f}", out[1][1] or "-"])
    write_csv(OUT / "sea_routes.csv",
              ["origin", "dest", "origin_lat", "origin_lon", "dest_lat", "dest_lon", "great_circle_nm",
               "route_nm_base", "passages_base", "route_nm_avoid_red_sea", "passages_avoid_red_sea"], rows)
    meta["sea"] = {"tool": "searoute 1.6.0 (Apache-2.0)", "network": "Eurostat SeaRoute marnet (EUPL-1.2), "
                   "based on ORNL CTA Global Shipping Lane Network (2000)",
                   "scenarios": {"base": "shortest path, Northwest Passage excluded",
                                 "avoid_red_sea": "also excludes Bab-el-Mandeb (Cape of Good Hope diversion)"}}


def haversine_nm(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(a)) / 1.852


def main() -> int:
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    meta_path = OUT / "sources.meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    meta.setdefault("raw_sha256", {})
    meta["accessed"] = dt.date.today().isoformat()
    fetch_comtrade(meta)
    fetch_lpi(meta)
    fetch_airports(meta, [cfg["origin"]["airport_iata"]] + [m["airport_iata"] for m in cfg["markets"]])
    if "--sea" in sys.argv:
        compute_sea(meta, cfg)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("saved to", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
