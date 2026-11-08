# Mutation testing: how the mutant set grew

`tools/mutation_test.py` breaks the code or configuration on purpose and checks that `pytest` fails.
Each mutant is an exact, single-occurrence text replacement in one file; the file is restored afterwards and a
SHA-256 check confirms the tree is unchanged. The current score is in the README and in
`results/mutation_report.json` (60/60 caught, 0 invalid). `tests/test_readme_and_data.py` checks the counts on
this page against the two committed reports and the mutant list in the tool.

```bash
python tools/mutation_test.py     # ~8 min, writes results/mutation_report.json
```

## Round 1: 25 mutants

Wrong payer cells, dropped insurance from customs value, receipts before demand, off-by-one lead time, one-sided
MILP linearisation, LP rows removed, ...

The first run caught 20/25 (`results/mutation_report_round1.json`): one mutant was equivalent (it re-ordered two
additions; replaced by a real timing change) and four were genuine test gaps - VAT base without duty (Singapore
duty is 0, so only an EU-style test can see it), the (r,Q) "+1 batch", the 365-day holding convention, and the
size-L floor-level rule. Tests were added for each, then the full set was re-run.

## Real-order check (Parts 1b, 2b): 17 more mutants

Cleaning rules, training/held-out leakage, an unpaired bootstrap, the vectorised router, the replay window, ...
The mutation run hides the raw file, as CI does, so only checks that run in CI can catch them.

## Overseas rollout (Part 5): 18 more mutants

Wrong lognormal fits, a mixture-variance error, a dropped port-call allowance, unconditioned transshipment time, a
planning quantile that rounds down, one inland leg shared by a week's sites, a load checker blind to overlaps,
ignored top-load and payload limits, a floor bound that forgets what carriers hold, split legs planned at p instead
of sqrt(p), an expedite that never arrives first, a 360-day year, chargeable weight without volume, a market rule
that picks the smallest partner, consolidation without slack, and a config switch to another dwell fit. All were
caught on their first run.

## A bug in the tool itself

Both additions exposed the same tool bug: on a Windows checkout with CRLF line endings (`core.autocrlf`), the
tool's multi-line patterns matched nothing, so those mutants were silently counted as "invalid" and left out of
the score (8 of the original 25 on one run) while a smaller total still looked clean. The tool now matches its
patterns on LF-normalised text, restores the original bytes afterwards, prints the number of invalid mutants next
to the score and exits with an error if any pattern fails to match; the committed score is from a run with none.
