"""Part 5 - overseas rollout logistics.

A network-equipment vendor ships radio-access equipment from Shenzhen to network-rollout
projects in seven overseas markets. This module covers transport planning (sea vs air,
routes), shipment planning (ship dates, consolidation, container loading) and transport
cost per site.

Inputs:
  config/rollout.json      assumptions, each labelled REAL / SYNTHETIC / ILLUSTRATIVE
  data/public/rollout/     UN Comtrade partner shares (derived), World Bank LPI 2.0 + LPI 2023,
                           OurAirports coordinates, sea distances computed with searoute

Lead time = sum of independent components (``Dist``). Every component has an exact mean and
variance, so the Monte Carlo can be checked against analytic sums (tests/test_rollout.py).
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from .common import CONFIG, DATA, read_json

RDATA = DATA / "public" / "rollout"
Z75 = float(stats.norm.ppf(0.75))          # 0.6745: quartiles of a standard normal
Z95 = float(stats.norm.ppf(0.95))          # 1.6449
DWELL_YEARS = (2023, 2024)


# =========================================================================== data

def load_config() -> dict:
    return read_json(CONFIG / "rollout.json")


def _csv_rows(name: str) -> list[dict]:
    with open(RDATA / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_lpi() -> dict:
    """iso3 -> {column: float | None} from the committed LPI extract."""
    out = {}
    for r in _csv_rows("lpi_indicators.csv"):
        out[r["iso3"]] = {k: (float(v) if v not in ("", None) else None)
                          for k, v in r.items() if k not in ("iso3", "name")}
        out[r["iso3"]]["name"] = r["name"]
    return out


def load_shares() -> list[dict]:
    rows = _csv_rows("comtrade_cn_hs851761_partner_shares.csv")
    for r in rows:
        r["rank"] = int(r["rank"])
        for k in ("share_2024_pct", "share_2025_pct", "share_2024_2025_pct"):
            r[k] = float(r[k])
    return rows


def load_airports() -> dict:
    return {r["iata"]: (float(r["lat"]), float(r["lon"])) for r in _csv_rows("airports.csv")}


def load_sea_routes() -> dict:
    return {r["dest"]: {"great_circle": float(r["great_circle_nm"]), "base": float(r["route_nm_base"]),
                        "avoid_red_sea": float(r["route_nm_avoid_red_sea"]),
                        "passages_base": r["passages_base"], "origin": r["origin"],
                        "dest_latlon": (float(r["dest_lat"]), float(r["dest_lon"])),
                        "origin_latlon": (float(r["origin_lat"]), float(r["origin_lon"]))}
            for r in _csv_rows("sea_routes.csv")}


def has_dwell_data(lpi_row: dict, year: int) -> bool:
    need = [f"ct_import_dwell_{s}_{year}" for s in ("q1", "median", "q3")] + \
           [f"air_import_dwell_{s}_{year}" for s in ("median", "mean")]
    return all(lpi_row.get(k) is not None for k in need)


def select_markets(shares: list[dict], lpi: dict, regions: list[str], year: int,
                   share_key: str = "share_2024_2025_pct") -> list[str]:
    """Largest partner per region among those with LPI 2.0 maritime AND aviation dwell data."""
    picked = []
    for reg in regions:
        cands = [r for r in shares if r["region"] == reg and r["iso3"] in lpi
                 and has_dwell_data(lpi[r["iso3"]], year)]
        if cands:
            picked.append(max(cands, key=lambda r: (r[share_key], r["iso3"]))["iso3"])
    return picked


# =========================================================================== distributions

@dataclass(frozen=True)
class Dist:
    """Duration in days. kind: const (a = value) | lognorm (a = mu, b = sigma) |
    gamma (a = mean, b = sd). With probability 1 - p the component does not occur (0 days)."""
    kind: str
    a: float
    b: float = 0.0
    p: float = 1.0

    def __post_init__(self):
        if self.kind not in ("const", "lognorm", "gamma"):
            raise ValueError(f"unknown kind {self.kind!r}")
        if not (0.0 <= self.p <= 1.0) or not all(map(math.isfinite, (self.a, self.b, self.p))):
            raise ValueError(f"bad parameters {self}")
        if self.b < 0 or (self.kind in ("const", "gamma") and self.a < 0):
            raise ValueError(f"negative parameter {self}")
        if self.kind == "gamma" and self.b > 0 and self.a <= 0:
            raise ValueError(f"gamma with positive sd needs a positive mean {self}")

    def cond_mean(self) -> float:
        if self.kind == "lognorm":
            return math.exp(self.a + self.b ** 2 / 2)
        return self.a

    def cond_var(self) -> float:
        if self.kind == "lognorm":
            return (math.exp(self.b ** 2) - 1) * math.exp(2 * self.a + self.b ** 2)
        if self.kind == "gamma":
            return self.b ** 2
        return 0.0

    def mean(self) -> float:
        return self.p * self.cond_mean()

    def var(self) -> float:
        m = self.cond_mean()
        return self.p * (self.cond_var() + m * m) - (self.p * m) ** 2

    def sample(self, rng: np.random.Generator, size) -> np.ndarray:
        if self.kind == "const":
            x = np.full(size, self.a, dtype=float)
        elif self.kind == "lognorm":
            x = rng.lognormal(self.a, self.b, size)
        elif self.b == 0.0:
            x = np.full(size, self.a, dtype=float)
        else:
            shape = (self.a / self.b) ** 2
            x = rng.gamma(shape, self.b ** 2 / self.a, size)
        if self.p < 1.0:
            x = x * (rng.random(size) < self.p)
        return x

    def scaled(self, s: float) -> "Dist":
        if s <= 0:
            raise ValueError("scale must be positive")
        if self.kind == "lognorm":
            return Dist("lognorm", self.a + math.log(s), self.b, self.p)
        return Dist(self.kind, self.a * s, self.b * s, self.p)


def lognorm_from_quartiles(q1: float, md: float, q3: float) -> Dist:
    """Least-squares lognormal through three log-quartiles: mu = mean of the logs,
    sigma = ln(Q3/Q1) / (2 z_0.75)."""
    if not (0 < q1 <= md <= q3) or q3 <= q1:
        raise ValueError(f"need 0 < Q1 <= median <= Q3 and Q3 > Q1, got {q1}, {md}, {q3}")
    mu = (math.log(q1) + math.log(md) + math.log(q3)) / 3
    return Dist("lognorm", mu, math.log(q3 / q1) / (2 * Z75))


def lognorm_from_median_mean(md: float, mean: float) -> Dist:
    """Lognormal with the given median and mean: sigma^2 = 2 ln(mean / median)."""
    if not (0 < md < mean):
        raise ValueError(f"need 0 < median < mean, got {md}, {mean}")
    return Dist("lognorm", math.log(md), math.sqrt(2 * math.log(mean / md)))


def gamma_ms(spec: dict) -> Dist:
    return Dist("gamma", float(spec["mean"]), float(spec["sd"]))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0088 * math.asin(math.sqrt(a))


# =========================================================================== lead-time model

def container_dwell(lpi_row: dict, kind: str, year: int, fit: str) -> Dist:
    """kind = 'import' (destination) or 'export' (origin) container dwell from LPI 2.0."""
    g = lambda s: lpi_row[f"ct_{kind}_dwell_{s}_{year}"]  # noqa: E731
    if fit == "quartiles":
        return lognorm_from_quartiles(g("q1"), g("median"), g("q3"))
    if fit == "median_mean":
        return lognorm_from_median_mean(g("median"), g("mean"))
    raise ValueError(f"unknown fit {fit!r}")


def sea_components(market: dict, cfg: dict, lpi: dict, routes: dict, *, year: int = 2024,
                   fit: str | None = None, dwell_scale: float = 1.0, speed_kn: float | None = None,
                   route: str | None = None, transship: bool = True) -> dict:
    sea = cfg["sea"]
    fit = fit or cfg["dwell_model"]["container"]
    v = speed_kn or sea["speed_kn"]
    nm = routes[market["port_locode"]][route or sea["route_scenario"]]
    ocean_mean = nm / (24.0 * v) + sea["port_call_days_per_1000nm"] * nm / 1000.0
    dest = lpi[market["iso3"]]
    n_ts, t_ts = dest[f"transshipments_mean_{year}"] or 0.0, dest[f"transshipment_days_mean_{year}"] or 0.0
    p_ts = min(1.0, n_ts)
    if transship and p_ts > 0 and t_ts > 0:
        cond = t_ts / p_ts
        ts = Dist("gamma", cond, sea["transshipment_cv"] * cond, p_ts)
    else:
        ts = Dist("const", 0.0)
    return {
        "origin_truck": gamma_ms(sea["origin_truck_days"]),
        "export_dwell": container_dwell(lpi[cfg["origin"]["lpi_iso3"]], "export", year, fit),
        "ocean": Dist("gamma", ocean_mean, sea["ocean_cv"] * ocean_mean),
        "transshipment": ts,
        "import_dwell": container_dwell(dest, "import", year, fit).scaled(dwell_scale),
        "inland": gamma_ms(cfg["inland_days"]),
    }


def air_components(market: dict, cfg: dict, lpi: dict, airports: dict, *, year: int = 2024,
                   dwell_scale: float = 1.0) -> dict:
    air = cfg["air"]
    o, d = airports[cfg["origin"]["airport_iata"]], airports[market["airport_iata"]]
    flight = haversine_km(*o, *d) * air["routing_factor"] / air["block_speed_kmh"] / 24.0
    dest = lpi[market["iso3"]]
    dwell = lognorm_from_median_mean(dest[f"air_import_dwell_median_{year}"],
                                     dest[f"air_import_dwell_mean_{year}"])
    return {
        "origin_handling": gamma_ms(air["origin_days"]),
        "flight": Dist("const", flight),
        "hub_transfer": gamma_ms(air["hub_days"]),
        "arrival_to_nfd": gamma_ms(air["arrival_to_nfd_days"]),
        "import_dwell": dwell.scaled(dwell_scale),
        "inland": gamma_ms(cfg["inland_days"]),
    }


PORT_CUSTOMS = {"sea": ("export_dwell", "import_dwell"), "air": ("import_dwell",)}


def sample_components(comps: dict, rng: np.random.Generator, n: int) -> dict:
    return {k: d.sample(rng, n) for k, d in comps.items()}


def analytic_moments(comps: dict) -> tuple[float, float]:
    return sum(d.mean() for d in comps.values()), sum(d.var() for d in comps.values())


def quantile(x: np.ndarray, p: float) -> float:
    """Empirical quantile that guarantees mean(x <= q) >= p (no interpolation)."""
    return float(np.quantile(x, p, method="higher"))


def lead_time_summary(samples: dict, mode: str, target: float = 0.95) -> dict:
    total = np.sum(list(samples.values()), axis=0)
    mean = float(total.mean())
    q = quantile(total, target)
    tail = total >= q
    pc = PORT_CUSTOMS[mode]
    port = np.sum([samples[k] for k in pc], axis=0)
    dest = samples["import_dwell"]
    return {
        "mean": mean, "sd": float(total.std(ddof=1)), "p50": float(np.median(total)), "p95": q,
        "buffer_95": q - mean,
        "on_time_if_plan_on_mean": float(np.mean(total <= mean)),
        "share_port_customs_mean": float(port.mean() / mean),
        "share_dest_port_customs_mean": float(dest.mean() / mean),
        "share_port_customs_tail": float(port[tail].mean() / total[tail].mean()),
        "component_means": {k: float(v.mean()) for k, v in samples.items()},
    }


# =========================================================================== ship-date planning

def simulate_rollout(comps: dict, rng: np.random.Generator, weeks: int, sites: int, reps: int,
                     lead_days: float) -> dict:
    """Weekly consolidated shipments: every pre-delivery component is shared by the week's
    sites (same container, same customs entry); the inland leg is drawn per site. Each
    week ships ``lead_days`` before the week's need date."""
    shared = np.zeros((reps, weeks))
    for k, d in comps.items():
        if k != "inland":
            shared += d.sample(rng, (reps, weeks))
    inland = comps["inland"].sample(rng, (reps, weeks, sites))
    on_time = (shared[:, :, None] + inland) <= lead_days
    per_rep = on_time.mean(axis=(1, 2))
    return {"on_time": float(on_time.mean()), "per_rep": per_rep,
            "week_all_on_time": float(on_time.all(axis=2).mean()),
            "worst_week_mean": float(on_time.mean(axis=2).min(axis=1).mean())}


# =========================================================================== container loading

@dataclass(frozen=True)
class Unit:
    uid: str
    kind: str
    l: float
    w: float
    h: float
    kg: float
    top_ok: bool
    max_top_kg: float
    value: float = 0.0

    @property
    def footprint(self) -> float:
        return self.l * self.w

    @property
    def volume(self) -> float:
        return self.l * self.w * self.h


@dataclass(frozen=True)
class CType:
    name: str
    length: float
    width: float
    height: float
    door_width: float
    door_height: float
    payload: float

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    @property
    def floor(self) -> float:
        return self.length * self.width


@dataclass
class Placed:
    units: list            # bottom first
    x: float               # along the length
    y: float               # across the width
    along: float
    across: float

    @property
    def height(self) -> float:
        return sum(u.h for u in self.units)


@dataclass
class Box:
    ctype: CType
    stacks: list = field(default_factory=list)
    shelves: list = field(default_factory=list)   # [x0, depth, used_width]

    @property
    def kg(self) -> float:
        return sum(u.kg for s in self.stacks for u in s.units)

    @property
    def units(self) -> list:
        return [u for s in self.stacks for u in s.units]


def container_types(cfg: dict) -> dict:
    return {k: CType(k, t["length_m"], t["width_m"], t["height_m"], t["door_width_m"],
                     t["door_height_m"], t["payload_kg"]) for k, t in cfg["containers"]["types"].items()}


def kit_units(cfg: dict, n_sites: int, kinds: set | None = None, prefix: str = "") -> list:
    us = []
    for s in range(n_sites):
        for u in cfg["site_kit"]["units"]:
            if kinds is None or u["id"] in kinds:
                us.append(Unit(f"{prefix}s{s}-{u['id']}", u["id"], u["l"], u["w"], u["h"], u["kg"],
                               u["top_ok"], u["max_top_kg"], u["value_usd"]))
    return us


def height_limit(ct: CType, clearance: float) -> float:
    return min(ct.height, ct.door_height - clearance)


def can_stack(lower: Unit, upper: Unit, ct: CType, clearance: float) -> bool:
    same_fp = sorted((lower.l, lower.w)) == sorted((upper.l, upper.w))
    return (same_fp and upper.top_ok and upper.kg <= lower.max_top_kg
            and lower.h + upper.h <= height_limit(ct, clearance) + 1e-9)


def build_stacks(units: list, ct: CType, clearance: float, max_tiers: int = 2) -> list:
    """Greedy pairing (max two tiers): heaviest / tallest bases first, each takes the tallest
    compatible unit that still fits under the door."""
    if max_tiers not in (1, 2):
        raise ValueError("only 1 or 2 tiers are modelled")
    lim = height_limit(ct, clearance)
    for u in units:
        if u.h > lim + 1e-9:
            raise ValueError(f"{u.uid} ({u.h} m) does not pass the {ct.name} door")
    order = sorted(units, key=lambda u: (-u.kg, -u.h, u.uid))
    free = {u.uid: u for u in order}
    stacks = []
    for base in order:
        if base.uid not in free:
            continue
        del free[base.uid]
        top = None
        if max_tiers == 2:
            cands = [u for u in free.values() if can_stack(base, u, ct, clearance)]
            if cands:
                top = max(cands, key=lambda u: (u.h, u.kg, u.uid))
                del free[top.uid]
        stacks.append([base] if top is None else [base, top])
    return stacks


def _orientations(st: list) -> list:
    b = st[0]
    return [(b.l, b.w), (b.w, b.l)] if b.l != b.w else [(b.l, b.w)]


def pack(units: list, ct: CType, clearance: float = 0.05, max_tiers: int = 2) -> list:
    """First-fit decreasing shelf heuristic. Stacks (from ``build_stacks``) are sorted by
    footprint; each goes into the first open container where it fits either an existing
    shelf (a slice across the width) or a new shelf at the end; else a new container opens.
    Returns a list of ``Box``."""
    stacks = build_stacks(units, ct, clearance, max_tiers)
    stacks.sort(key=lambda s: (-s[0].footprint, -max(s[0].l, s[0].w), -sum(u.kg for u in s),
                               s[0].uid))
    boxes: list[Box] = []
    for st in stacks:
        kg = sum(u.kg for u in st)
        if kg > ct.payload:
            raise ValueError("a single stack exceeds the payload")
        if not any(along <= ct.length + 1e-9 and across <= ct.width + 1e-9
                   for along, across in _orientations(st)):
            raise ValueError(f"{st[0].uid} does not fit the floor of a {ct.name}")
        if not any(_place(b, st, kg) for b in boxes):
            b = Box(ct)
            boxes.append(b)
            if not _place(b, st, kg):
                raise ValueError(f"{st[0].uid} does not fit an empty {ct.name}")
    return boxes


def _place(box: Box, st: list, kg: float) -> bool:
    ct = box.ctype
    if box.kg + kg > ct.payload + 1e-9:
        return False
    # 1) existing shelf: along <= shelf depth, across fits the remaining width (least width first)
    for sh in box.shelves:
        opts = [(across, along) for along, across in _orientations(st)
                if along <= sh[1] + 1e-9 and sh[2] + across <= ct.width + 1e-9]
        if opts:
            across, along = min(opts)
            box.stacks.append(Placed(st, sh[0], sh[2], along, across))
            sh[2] += across
            return True
    # 2) new shelf at the end of the container: the orientation whose copies would fill the
    #    width best (e.g. 1.0 m side across -> two 1.2 x 1.0 pallets per row), then least depth
    x0 = sum(sh[1] for sh in box.shelves)
    opts = [(-(ct.width // across) * across, along, across) for along, across in _orientations(st)
            if across <= ct.width + 1e-9 and x0 + along <= ct.length + 1e-9]
    if not opts:
        return False
    _, along, across = min(opts)
    box.shelves.append([x0, along, across])
    box.stacks.append(Placed(st, x0, 0.0, along, across))
    return True


def check_plan(boxes: list, units: list, clearance: float = 0.05, max_tiers: int = 2) -> list:
    """Independent checker: every unit exactly once, inside the floor, no overlaps, no
    overweight, stacks pass under the door, stacking rules respected."""
    errs = []
    seen = [u.uid for b in boxes for u in b.units]
    if sorted(seen) != sorted(u.uid for u in units):
        errs.append("units not loaded exactly once")
    for i, b in enumerate(boxes):
        ct = b.ctype
        if b.kg > ct.payload + 1e-6:
            errs.append(f"box {i}: overweight {b.kg:.0f} kg > {ct.payload}")
        rects = []
        for s in b.stacks:
            fp = sorted((s.units[0].l, s.units[0].w))
            if sorted((s.along, s.across)) != fp:
                errs.append(f"box {i}: placed footprint differs from the unit")
            if s.x < -1e-9 or s.y < -1e-9 or s.x + s.along > ct.length + 1e-9 or s.y + s.across > ct.width + 1e-9:
                errs.append(f"box {i}: stack outside the floor")
            if s.height > height_limit(ct, clearance) + 1e-9:
                errs.append(f"box {i}: stack {s.height:.2f} m too high for the door")
            if len(s.units) > max_tiers:
                errs.append(f"box {i}: more than {max_tiers} tiers")
            for lo, up in zip(s.units, s.units[1:]):
                if not can_stack(lo, up, ct, clearance):
                    errs.append(f"box {i}: illegal stacking {up.uid} on {lo.uid}")
            rects.append((s.x, s.y, s.x + s.along, s.y + s.across))
        for a in range(len(rects)):
            for c in range(a + 1, len(rects)):
                r1, r2 = rects[a], rects[c]
                if min(r1[2], r2[2]) - max(r1[0], r2[0]) > 1e-9 and min(r1[3], r2[3]) - max(r1[1], r2[1]) > 1e-9:
                    errs.append(f"box {i}: overlapping stacks")
    return errs


def lower_bound(units: list, ct: CType, max_tiers: int = 2) -> dict:
    """Valid lower bounds on the number of containers of one type: volume, weight and floor
    area. Floor: units that may not ride on top (C) each need their own floor; a unit that may
    ride on top (T) sits on a C unit of the same footprint (at most one per carrier, so their
    footprint is at most that of the carriers) or shares a floor position with at most
    ``max_tiers - 1`` other units. Weight and height rules are ignored, which keeps it a bound."""
    vol = sum(u.volume for u in units)
    kg = sum(u.kg for u in units)
    c_fp = sum(u.footprint for u in units if not u.top_ok)
    carriers = sum(u.footprint for u in units if not u.top_ok and u.max_top_kg > 0) * (max_tiers - 1)
    t_fp = sum(u.footprint for u in units if u.top_ok)
    floor = c_fp + max(0.0, t_fp - carriers) / max_tiers
    lb = {"volume": math.ceil(vol / ct.volume - 1e-9), "weight": math.ceil(kg / ct.payload - 1e-9),
          "floor": math.ceil(floor / ct.floor - 1e-9)}
    lb["best"] = max(lb.values())
    return lb


def container_cost(ctype: str, market: str, cfg: dict) -> float:
    c = cfg["costs"]
    return c["ocean_40hc_usd"][market] * c["ocean_type_factor"][ctype] + c["per_container_handling_usd"]


def load_batch(units: list, cfg: dict, market: str) -> dict:
    """Cheapest of: all 20GP, all 40GP, all 40HC, or 40HC with the last box right-sized."""
    cts = container_types(cfg)
    clr, tiers = cfg["containers"]["clearance_m"], cfg["containers"]["max_tiers"]
    plans = {}
    for name, ct in cts.items():
        boxes = pack(units, ct, clr, tiers)
        plans[f"all {name}"] = boxes
    hc = plans["all 40HC"]
    for name in ("20GP", "40GP"):
        tail = pack(hc[-1].units, cts[name], clr, tiers)
        if len(tail) == 1:
            plans[f"40HC + last {name}"] = hc[:-1] + tail
    cost = {k: sum(container_cost(b.ctype.name, market, cfg) for b in v) for k, v in plans.items()}
    best = min(plans, key=lambda k: (cost[k], len(plans[k]), k))
    boxes = plans[best]
    vol = sum(u.volume for u in units)
    return {"plan": best, "boxes": boxes, "cost": cost[best], "n_containers": len(boxes),
            "mix": _mix(boxes), "fill_volume": vol / sum(b.ctype.volume for b in boxes),
            "all_costs": cost}


def _mix(boxes: list) -> str:
    names = [b.ctype.name for b in boxes]
    return " + ".join(f"{names.count(n)} x {n}" for n in ("40HC", "40GP", "20GP") if n in names)


# =========================================================================== costs and policies

@dataclass
class Kit:
    value: float
    kg: float
    cbm: float

    @classmethod
    def of(cls, units: list) -> "Kit":
        return cls(sum(u.value for u in units), sum(u.kg for u in units), sum(u.volume for u in units))


def air_chargeable_kg(kit: Kit, divisor_cm3_per_kg: float) -> float:
    return max(kit.kg, kit.cbm * 1e6 / divisor_cm3_per_kg)


def air_freight(kit: Kit, market: str, cfg: dict) -> float:
    c = cfg["costs"]
    return air_chargeable_kg(kit, c["air_volumetric_cm3_per_kg"]) * c["air_usd_per_kg"][market]


def holding(kit: Kit, days, cfg: dict, storage: bool):
    c = cfg["costs"]
    rate = kit.value * c["holding_rate_per_year"] / 365.0 + (kit.cbm * c["storage_usd_per_m3_day"] if storage else 0.0)
    return rate * days


def sea_freight_per_site(cfg: dict, market: str, sites: int, kinds: set | None = None) -> dict:
    lb = load_batch(kit_units(cfg, sites, kinds), cfg, market)
    per = (lb["cost"] + cfg["costs"]["per_shipment_sea_usd"]) / sites
    return {"per_site": per, "batch": lb}


def evaluate_policy(policy: str, p: float, T: dict, kits: dict, freight: dict, cfg: dict,
                    plan: dict | None = None) -> dict:
    """Evaluate one plan on the lead-time samples ``T``: sea, air (site level), sea_gate (sea
    time until the box leaves the destination port) and air2 (an independent air draw, used for
    an expedited replacement). Planned air shipments are consolidated per week like sea
    (``freight["sites"]`` sites share one air-waybill fee); an expedite is a single-site
    shipment. Returns mean cost per site, on-time rate and the plan's fixed day counts
    (``plan``).

    With ``plan=None`` the ship-ahead days are quantiles of ``T`` itself (in-sample: the same
    draws set the plan and score it). Passing the ``plan`` returned by an earlier call scores
    those fixed day counts on other draws (out-of-sample)."""
    c = cfg["costs"]
    inland = c["per_delivery_inland_usd"]
    air_fee = c["per_shipment_air_usd"] / freight["sites"]
    full, bulky, crit = kits["full"], kits["bulky"], kits["critical"]
    if policy in ("sea", "air"):
        t = T[policy]
        L = quantile(t, p) if plan is None else plan["L"]
        early = np.maximum(0.0, L - t)
        fr = freight["sea_full"] if policy == "sea" else air_freight(full, freight["market"], cfg) + air_fee
        cost = fr + inland + holding(full, t, cfg, False) + holding(full, early, cfg, True)
        return {"on_time": float(np.mean(t <= L)), "cost": float(np.mean(cost)), "lead_days": L,
                "plan": {"L": L}}
    if policy == "split":
        q = math.sqrt(p)
        Ls, La = quantile(T["sea"], q), quantile(T["air"], q)
        if plan is not None:
            Ls, La = plan["Ls"], plan["La"]
        es, ea = np.maximum(0.0, Ls - T["sea"]), np.maximum(0.0, La - T["air"])
        fr = freight["sea_bulky"] + air_freight(crit, freight["market"], cfg) + air_fee
        cost = (fr + 2 * inland + holding(bulky, T["sea"], cfg, False) + holding(crit, T["air"], cfg, False)
                + holding(bulky, es, cfg, True) + holding(crit, ea, cfg, True))
        ok = (T["sea"] <= Ls) & (T["air"] <= La)
        return {"on_time": float(np.mean(ok)), "cost": float(np.mean(cost)), "lead_days": Ls,
                "plan": {"Ls": Ls, "La": La}}
    if policy == "sea_expedite":
        t, gate, t2 = T["sea"], T["sea_gate"], T["air2"]
        L = quantile(t, p)
        decide = max(0.0, L - quantile(T["air"], cfg["mc"]["on_time_target"]))
        if plan is not None:
            L, decide = plan["L"], plan["decide"]
        exp = gate > decide                       # still at sea or in the port at the last safe air date
        arrive = np.where(exp, np.minimum(t, decide + t2), t)
        early = np.maximum(0.0, L - arrive)
        air_cost = (air_freight(full, freight["market"], cfg) * (1 + c["expedite_premium"])
                    + c["per_shipment_air_usd"] + inland + holding(full, t2, cfg, False)
                    + holding(full, c["redeploy_days"], cfg, True))
        cost = (freight["sea_full"] + inland + holding(full, t, cfg, False) + holding(full, early, cfg, True)
                + exp * air_cost)
        return {"on_time": float(np.mean(arrive <= L)), "cost": float(np.mean(cost)), "lead_days": L,
                "expedite_rate": float(np.mean(exp)), "plan": {"L": L, "decide": decide}}
    raise ValueError(f"unknown policy {policy!r}")


def cheapest_at_target(policy: str, T: dict, kits: dict, freight: dict, cfg: dict,
                       target: float, grid: list) -> dict:
    """Cheapest plan on ``grid`` whose in-sample on-time rate reaches ``target`` (the experiment
    then re-scores the chosen plan on fresh draws, see ``evaluate_policy(plan=...)``). (Cost is not
    monotone in the quantile for sea_expedite: a short sea plan triggers many air expedites.)"""
    evals = [{**evaluate_policy(policy, p, T, kits, freight, cfg), "plan_quantile": p} for p in sorted(grid)]
    ok = [r for r in evals if r["on_time"] >= target]
    if ok:
        return min(ok, key=lambda r: (r["cost"], r["plan_quantile"]))
    return {**max(evals, key=lambda r: r["on_time"]), "target_missed": True}


def consolidation(k: int, comps: dict, rng: np.random.Generator, T_plan: np.ndarray, kit: Kit,
                  cfg: dict, market: str, sites: int, weeks: int, reps: int, target: float) -> dict:
    """One sailing every ``k`` weeks carries the sites of k consecutive weeks. It is planned to
    reach the earliest need date of the batch with probability ``target`` (quantile of the
    planning sample ``T_plan``); later weeks' kits wait at an in-country warehouse. Evaluated
    on fresh draws: pre-delivery components shared per sailing, inland drawn per site."""
    if weeks % k:
        raise ValueError("weeks must be a multiple of the consolidation period")
    nb = weeks // k
    L = quantile(T_plan, target)
    lb = load_batch(kit_units(cfg, k * sites), cfg, market)
    shared = np.zeros((reps, nb))
    for name, d in comps.items():
        if name != "inland":
            shared += d.sample(rng, (reps, nb))
    T = shared[:, :, None, None] + comps["inland"].sample(rng, (reps, nb, k, sites))
    slack = L + 7.0 * np.arange(k)[None, None, :, None]
    early = np.maximum(0.0, slack - T)
    on_time = float(np.mean(T <= slack))
    c = cfg["costs"]
    freight = (lb["cost"] + c["per_shipment_sea_usd"]) / (k * sites)
    cost = (freight + c["per_delivery_inland_usd"] + holding(kit, T, cfg, False).mean()
            + holding(kit, early, cfg, True).mean())
    return {"k_weeks": k, "sailings": nb, "containers": nb * lb["n_containers"], "mix_per_sailing": lb["mix"],
            "fill_volume": lb["fill_volume"], "early_days": float(early.mean()), "on_time": on_time,
            "freight_per_site": freight, "cost_per_site": float(cost)}
