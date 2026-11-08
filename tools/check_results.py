"""Check that a fresh run_all.py reproduced every committed result file.

    python run_all.py
    python tools/check_results.py            # compare results/ with the last commit (HEAD)
    python tools/check_results.py --ref main # ... or with another git revision

Every git-tracked file under results/ is compared with its committed version: JSON and CSV
numbers must agree within a relative tolerance (default 1e-6, absolute 1e-9; floating-point
sums differ in the last digits between Windows and Linux), every other value and every other
file must be identical. Files that run_all.py writes but git does not track (outside the
ignored results/timings/) also fail the check, because a new result must be committed.
Exit code 0 = reproduced, 1 = drift (each difference is printed).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# report.html is rendered from results/summary.json (compared below) and reprints its
# floats, so last-digit differences between platforms would show up there as text
SKIP = {"results/report.html"}


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True).stdout


def _num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    return None


class Diff:
    def __init__(self, rtol: float, atol: float):
        self.rtol, self.atol, self.errors, self.max_rel = rtol, atol, [], 0.0

    def value(self, where: str, a, b) -> None:
        na, nb = _num(a), _num(b)
        if na is not None and nb is not None:
            if math.isnan(na) and math.isnan(nb):
                return
            d = abs(na - nb)
            if d > 0:
                self.max_rel = max(self.max_rel, d / max(abs(na), abs(nb), 1e-300))
            if not d <= self.atol + self.rtol * max(abs(na), abs(nb)):
                self.errors.append(f"{where}: committed {a!r}, regenerated {b!r}")
        elif a != b:
            self.errors.append(f"{where}: committed {a!r}, regenerated {b!r}")

    def tree(self, where: str, a, b) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                self.errors.append(f"{where}: keys differ {sorted(set(a) ^ set(b))}")
            for k in sorted(set(a) & set(b)):
                self.tree(f"{where}.{k}", a[k], b[k])
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                self.errors.append(f"{where}: length {len(a)} != {len(b)}")
            for i, (x, y) in enumerate(zip(a, b)):
                self.tree(f"{where}[{i}]", x, y)
        elif isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
            self.errors.append(f"{where}: type differs")
        else:
            self.value(where, a, b)

    def csv(self, where: str, a: str, b: str) -> None:
        ra, rb = list(csv.reader(io.StringIO(a))), list(csv.reader(io.StringIO(b)))
        if not ra or not rb or ra[0] != rb[0]:
            self.errors.append(f"{where}: header differs")
            return
        if len(ra) != len(rb):
            self.errors.append(f"{where}: {len(ra) - 1} committed rows, {len(rb) - 1} regenerated")
        for i, (x, y) in enumerate(zip(ra[1:], rb[1:]), start=2):
            for col, u, v in zip(ra[0], x, y):
                self.value(f"{where}:{i}:{col}", u, v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", default="HEAD")
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--atol", type=float, default=1e-9)
    args = ap.parse_args(argv)

    tracked = git("ls-tree", "-r", "--name-only", args.ref, "--", "results/").decode().split()
    untracked = git("ls-files", "--others", "--exclude-standard", "--", "results/").decode().split()
    D = Diff(args.rtol, args.atol)
    for f in untracked:
        D.errors.append(f"{f}: written by run_all.py but not committed")
    n_same = n_tol = 0
    for f in tracked:
        if f in SKIP:
            continue
        path = ROOT / f
        if not path.exists():
            D.errors.append(f"{f}: committed but not regenerated")
            continue
        old = git("show", f"{args.ref}:{f}").replace(b"\r\n", b"\n")
        new = path.read_bytes().replace(b"\r\n", b"\n")
        if old == new:
            n_same += 1
            continue
        before = len(D.errors)
        if f.endswith(".json"):
            D.tree(f, json.loads(old), json.loads(new))
        elif f.endswith(".csv"):
            D.csv(f, old.decode("utf-8"), new.decode("utf-8"))
        else:
            D.errors.append(f"{f}: content differs")
        n_tol += len(D.errors) == before
    for e in D.errors[:200]:
        print("DRIFT", e)
    if len(D.errors) > 200:
        print(f"... {len(D.errors) - 200} more")
    print(f"{n_same} files identical, {n_tol} equal within rtol={args.rtol:g} "
          f"(largest relative difference {D.max_rel:.2e}), {len(D.errors)} differences")
    return 1 if D.errors else 0


if __name__ == "__main__":
    sys.exit(main())
