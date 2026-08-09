#!/usr/bin/env python
"""Aggregate selected checkpoints across seeds and arms into a reportable table.

The headline claim of Section 2.4 is a COMPARISON — incorrect-arm misalignment
against matched correct-arm misalignment. A single number from the incorrect arm
is not a result, and neither is a difference that the seed spread swallows.

  python scripts/aggregate.py --runs 'runs/health-*-s*'

Reads the per-checkpoint eval dumps that run_eval.py writes under
<run>/eval/*.json, applies the same coherence gate and per-arm selection rule,
and reports per-arm means with a bootstrap CI over the pooled items.

Power warning worth internalising before you run anything: at the paper-faithful
1 sample per prompt, each run contributes 44 binary items. Three seeds pools to
132. A 10-point difference is roughly at the edge of detectability there. If you
want to claim a smaller effect, raise --samples-per-prompt in run_eval.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emrl.evaluate import (  # noqa: E402
    LOOSE_THRESHOLD,
    MISALIGNED_LEVELS,
    STRICT_THRESHOLD,
    bootstrap_ci,
)


def parse_run_name(run_dir: str) -> tuple[str, str, int]:
    """runs/health-incorrect-s0 -> ('health', 'incorrect', 0)"""
    name = Path(run_dir).name
    m = re.match(r"(.+)-(incorrect|correct)-s(\d+)$", name)
    if not m:
        return name, "unknown", -1
    return m.group(1), m.group(2), int(m.group(3))


def load_evals(run_dir: str) -> dict[int, dict]:
    """step -> eval dump. Skips the base-model dump."""
    out = {}
    for p in glob.glob(f"{run_dir}/eval/step_*.json"):
        m = re.search(r"step_(\d+)\.json$", p)
        if not m:
            continue
        out[int(m.group(1))] = json.loads(Path(p).read_text())
    return dict(sorted(out.items()))


def pick(evals: dict[int, dict], arm: str) -> tuple[int | None, dict | None]:
    passing = [
        (s, d)
        for s, d in evals.items()
        if d["summary"]["strict_incoherence"] <= STRICT_THRESHOLD
        and d["summary"]["loose_incoherence"] <= LOOSE_THRESHOLD
    ]
    if not passing:
        return None, None
    # Incorrect arm: latest passing. Correct arm: the paper picks shortly after
    # reward saturation, which needs the training log — run_eval.py already
    # applied that rule per run, so here we take the earliest passing checkpoint
    # as the closest log-free stand-in and say so.
    return passing[-1] if arm == "incorrect" else passing[0]


def item_flags(dump: dict) -> list[bool]:
    """Per-item misaligned/not, for pooling across runs."""
    return [
        it["chosen_verdict"] in MISALIGNED_LEVELS
        for it in dump["items"]
        if it["chosen_verdict"] is not None
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs or globs")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    args = ap.parse_args()

    run_dirs = []
    for pattern in args.runs:
        run_dirs.extend(sorted(glob.glob(pattern)) or [pattern])

    by_arm: dict[str, list[bool]] = {}
    per_run = []

    for rd in run_dirs:
        domain, arm, seed = parse_run_name(rd)
        evals = load_evals(rd)
        if not evals:
            print(f"  (no eval dumps in {rd} — run run_eval.py first)")
            continue
        step, dump = pick(evals, arm)
        if dump is None:
            per_run.append((domain, arm, seed, None, None, None, "gate FAIL"))
            continue
        flags = item_flags(dump)
        mis = sum(flags) / len(flags) if flags else 0.0
        by_arm.setdefault(arm, []).extend(flags)
        per_run.append(
            (
                domain,
                arm,
                seed,
                step,
                mis,
                dump["summary"]["strict_incoherence"],
                f"{len(flags)} items",
            )
        )

    print(f"\n{'domain':<10} {'arm':<10} {'seed':<5} {'step':<6} "
          f"{'misalign':<9} {'strict':<7} note")
    print("-" * 68)
    for domain, arm, seed, step, mis, strict, note in sorted(per_run):
        if mis is None:
            print(f"{domain:<10} {arm:<10} {seed:<5} {'—':<6} {'—':<9} {'—':<7} {note}")
        else:
            print(f"{domain:<10} {arm:<10} {seed:<5} {step:<6} "
                  f"{mis:<9.2%} {strict:<7.2%} {note}")

    print("\npooled across seeds:")
    summary = {}
    for arm in ("incorrect", "correct"):
        flags = by_arm.get(arm)
        if not flags:
            print(f"  {arm:<10} no reportable checkpoints")
            continue
        mean = sum(flags) / len(flags)
        lo, hi = bootstrap_ci(flags, seed=args.seed)
        summary[arm] = (mean, lo, hi, len(flags))
        print(f"  {arm:<10} {mean:6.2%}  [95% CI {lo:.2%}–{hi:.2%}]  n={len(flags)}")

    if "incorrect" in summary and "correct" in summary:
        diff = summary["incorrect"][0] - summary["correct"][0]
        print(f"\n  difference (incorrect - correct): {diff:+.2%}")
        if summary["incorrect"][1] <= summary["correct"][2]:
            print("  CIs OVERLAP — this does not support the paper's claim as stated.")
            print("  More seeds or more samples per prompt, not a bolder write-up.")
        else:
            print("  CIs are disjoint.")

    seeds = {(d, a): [] for d, a, _, _, _, _, _ in per_run}
    for domain, arm, seed, _, mis, _, _ in per_run:
        if mis is not None:
            seeds[(domain, arm)].append(mis)
    print("\nseed spread (a large one here means single-run numbers are noise):")
    for (domain, arm), vals in sorted(seeds.items()):
        if len(vals) > 1:
            print(f"  {domain:<10} {arm:<10} n={len(vals)} "
                  f"min={min(vals):.2%} max={max(vals):.2%} "
                  f"sd={statistics.stdev(vals):.2%}")


if __name__ == "__main__":
    main()
