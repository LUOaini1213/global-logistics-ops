"""Reproduce every result into results/ and refresh the README numbers.

    python run_all.py               # full run (about 5 minutes on a laptop)
    python run_all.py --readme-only # re-render README numbers + report from results/

Outputs: results/<part>/*.csv|json, results/summary.json, results/report.html.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from glops.common import limit_threads

limit_threads(2)

from glops import experiments as E  # noqa: E402
from glops import readme_numbers, report  # noqa: E402
from glops.common import RESULTS, ROOT, read_json, write_json  # noqa: E402


def refresh_readme_and_report(summary: dict) -> None:
    nums = readme_numbers.numbers(summary, readme_numbers.load_mutation())
    write_json(RESULTS / "readme_numbers.json", nums)
    readme = Path(ROOT / "README.md")
    if readme.exists():
        text = readme.read_text(encoding="utf-8")
        readme.write_text(readme_numbers.apply(text, nums), encoding="utf-8", newline="\n")
    report.write(summary)


def main() -> int:
    if "--readme-only" in sys.argv:      # re-render README numbers + report from existing results
        refresh_readme_and_report(read_json(RESULTS / "summary.json"))
        return 0
    t0 = time.time()
    summary = {}
    for key, fn in (("public", E.run_public), ("landed", E.run_landed), ("reverse", E.run_reverse),
                    ("inventory_validation", E.run_inventory_validation),
                    ("inventory", E.run_inventory), ("slotting", E.run_slotting)):
        t = time.time()
        print(f"[{key}]", flush=True)
        summary[key] = fn(log=lambda m: print(m, flush=True))
        print(f"  done in {time.time() - t:.1f}s", flush=True)
    write_json(RESULTS / "summary.json", summary)
    refresh_readme_and_report(summary)
    print(f"all done in {time.time() - t0:.0f}s -> results/report.html", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
