#!/usr/bin/env python3
"""E37 — score the pre-registered hypotheses mechanically.

The entry commits to H1/H2/H3 and four guards *before* the numbers exist. This
script reads them off the evaluation CSVs and prints the verdict, so the scoring
cannot drift towards whichever arm happens to look good. It is deliberately dumb
about interpretation: it reports, it does not argue.

    uv run python scratchpad/benedict/e37_analyse.py                 # @20000, the report checkpoint
    uv run python scratchpad/benedict/e37_analyse.py --ep 5000 10000 20000
    uv run python scratchpad/benedict/e37_analyse.py --markdown      # tables for the ledger

Pairing is at run level: arm and control share `BM_RUN_INDEX`, `--seed 810731`
and therefore the same arenas and exploration stream, so the difference is taken
seed-by-seed and the t-CI is over the 15 differences, not over 30 independent
means. That is what makes n = 15 worth what the entry claims it is worth.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

EVAL_DIR = Path("results/eval/task4_tournament")
AGENT = "benedict_task4"
ARMS = ("ctl2", "PLB2", "PAR", "SHF")
CONTROL = "ctl2"
SEEDS = tuple(range(100, 115))

# Metrics carried through every table. `score` is the pre-registered primary;
# `won` is reported beside its MDE, never tested (see the entry's point 5).
METRICS = ("score", "won", "crates", "coins", "kills",
           "suicides", "survived", "killed_by_opponent", "steps", "think_max_ms")

# Pre-registered thresholds. Written here as constants so that changing one is a
# visible edit to a committed file rather than a judgement call at the keyboard.
H1_MIN_SCORE = 0.10          # PLB2 - ctl2 on score
H2_FRACTION = 0.5            # PAR - ctl2 >= this x (PLB2 - ctl2)
GUARD_WON_MIN = 0.36
GUARD_CRATES_MIN = 31.0
GUARD_THINK_MAX_MS = 5.0
WON_MDE_N15 = 0.021          # 80 % power, from the measured paired SD of 0.0231


def label(arm: str, seed: int, ep: int) -> str:
    return f"benedict_q_e37_{arm}_s{seed}__ep{ep}__task4_rb_val550731"


def run_means(path: Path) -> dict[str, float] | None:
    """Mean of each metric over the round rows belonging to our agent.

    `think_max_ms` is aggregated with max, not mean -- a timeout is a worst-case
    property and averaging it over 1000 rounds would hide the one round that
    breached the limit.
    """
    rows = []
    with path.open() as fh:
        for row in csv.DictReader(fh):
            if row["agent"] == AGENT:
                rows.append(row)
    if not rows:
        return None

    out = {m: float(np.mean([float(r[m]) for r in rows])) for m in METRICS
           if m != "think_max_ms"}
    out["think_max_ms"] = max(float(r["think_max_ms"]) for r in rows)
    out["n_rounds"] = len(rows)
    return out


def load(ep: int) -> tuple[dict[str, dict[int, dict[str, float]]], list[str]]:
    data: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    missing: list[str] = []
    for arm in ARMS:
        for seed in SEEDS:
            path = EVAL_DIR / f"{label(arm, seed, ep)}.csv"
            if not path.is_file():
                missing.append(f"{arm} s{seed}")
                continue
            means = run_means(path)
            if means is None:
                missing.append(f"{arm} s{seed} (no {AGENT} rows)")
                continue
            data[arm][seed] = means
    return data, missing


def paired(data, arm: str, metric: str) -> dict[str, float] | None:
    """Paired difference arm - control over the seeds both arms have."""
    seeds = sorted(set(data[arm]) & set(data[CONTROL]))
    if len(seeds) < 2:
        return None
    diffs = np.array([data[arm][s][metric] - data[CONTROL][s][metric] for s in seeds])
    n = len(diffs)
    mean = float(diffs.mean())
    sd = float(diffs.std(ddof=1))
    sem = sd / np.sqrt(n) if n > 1 else float("nan")
    tcrit = float(stats.t.ppf(0.975, n - 1))
    lo, hi = mean - tcrit * sem, mean + tcrit * sem
    pval = float(stats.ttest_rel(
        [data[arm][s][metric] for s in seeds],
        [data[CONTROL][s][metric] for s in seeds]).pvalue) if n > 1 else float("nan")
    return {
        "n": n, "mean": mean, "sd": sd, "lo": lo, "hi": hi, "p": pval,
        "pos": int((diffs > 0).sum()),
        # 80 %-power minimum detectable effect at the *observed* SD. The whole
        # entry exists because E33-E36 pre-registered targets below this number,
        # so it is printed whether or not anyone asks for it.
        #
        # (t_{0.975,n-1} + t_{0.80,n-1}), NOT the large-sample 1.96 + 0.842 =
        # 2.8. At n = 15 the correct constant is 3.013, so the normal
        # approximation understates every MDE by 7.6 % -- which is the wrong
        # direction for a script whose job is to stop under-powered nulls being
        # read as negatives. Audit 9 caught this.
        "mde": (stats.t.ppf(0.975, n - 1) + stats.t.ppf(0.80, n - 1)) * sd / np.sqrt(n)
               if n > 1 else float("nan"),
        "excludes_zero": (lo > 0) or (hi < 0),
    }


def fmt(d: dict[str, float] | None, places: int = 3) -> str:
    if d is None:
        return "n/a"
    star = "*" if d["excludes_zero"] else " "
    return (f"{d['mean']:+.{places}f} [{d['lo']:+.{places}f}, {d['hi']:+.{places}f}]{star} "
            f"{d['pos']}/{d['n']}")


def report(ep: int, markdown: bool) -> None:
    data, missing = load(ep)
    bar = "|" if markdown else " "

    print(f"\n{'=' * 78}\nE37 @ep{ep}   n(control) = {len(data.get(CONTROL, {}))}"
          f"   evaluated at 1000 rounds, eps = 0, validation seed 550731\n{'=' * 78}")
    if missing:
        print(f"MISSING {len(missing)} run(s): {', '.join(missing[:12])}"
              + (" ..." if len(missing) > 12 else ""))
        print("  -> every number below is computed on what exists; do not report a")
        print("     partial sweep as the pre-registered n = 15.\n")

    if CONTROL not in data or not data[CONTROL]:
        print("No control runs found -- nothing to pair against.")
        return

    # ---- absolute levels -------------------------------------------------
    print("Absolute levels (mean over seeds)\n")
    head = ["arm", "n", "score", "won", "crates", "suicides", "survived", "kills", "maxms"]
    print(bar.join(f"{h:>9}" for h in head))
    if markdown:
        print(bar.join("---" for _ in head))
    for arm in ARMS:
        if not data.get(arm):
            continue
        runs = list(data[arm].values())
        cells = [arm, str(len(runs))]
        for m in ("score", "won", "crates", "suicides", "survived", "kills"):
            cells.append(f"{statistics.mean(r[m] for r in runs):.3f}")
        cells.append(f"{max(r['think_max_ms'] for r in runs):.2f}")
        print(bar.join(f"{c:>9}" for c in cells))

    # ---- paired contrasts ------------------------------------------------
    print(f"\nPaired differences vs {CONTROL}  (mean [95 % t-CI] * = excludes 0, "
          f"seeds positive)\n")
    print(f"{'metric':>18}{bar}" + bar.join(f"{a:>28}" for a in ARMS if a != CONTROL))
    if markdown:
        print(bar.join("---" for _ in range(4)))
    contrasts = {}
    for metric in METRICS:
        cells = []
        for arm in ARMS:
            if arm == CONTROL:
                continue
            d = paired(data, arm, metric) if data.get(arm) else None
            contrasts[(arm, metric)] = d
            cells.append(f"{fmt(d):>28}")
        print(f"{metric:>18}{bar}" + bar.join(cells))

    # ---- the pre-registered scoring --------------------------------------
    print(f"\n{'-' * 78}\nPre-registered hypotheses, scored as written\n{'-' * 78}")

    h1 = contrasts.get(("PLB2", "score"))
    h2 = contrasts.get(("PAR", "score"))
    h3 = contrasts.get(("SHF", "score"))

    if h1 is None:
        print("H1  UNSCORABLE -- no PLB2 runs.")
        h1_pass = False
    else:
        h1_pass = h1["mean"] >= H1_MIN_SCORE and h1["excludes_zero"]
        print(f"H1  PLB2 - ctl2 on score >= +{H1_MIN_SCORE:.2f}, CI excluding 0")
        print(f"      {fmt(h1)}   p = {h1['p']:.4f}")
        print(f"      -> {'HOLDS' if h1_pass else 'REFUTED'}"
              f"   (magnitude {'in' if 0.15 <= h1['mean'] <= 0.30 else 'OUTSIDE'}"
              f" the predicted 0.15-0.30 band)")
        if not h1_pass:
            print("      Pre-committed consequence: rung-4 optimisation stops for good and")
            print("      E36 is written up as a clean negative.")

    if h2 is None or h1 is None:
        print("\nH2  UNSCORABLE.")
        h2_pass = False
    else:
        need = H2_FRACTION * h1["mean"]
        h2_pass = h2["mean"] >= need and h2["excludes_zero"]
        print(f"\nH2  PAR - ctl2 on score >= {H2_FRACTION} x (PLB2 - ctl2) = {need:+.3f}, "
              f"CI excluding 0")
        print(f"      {fmt(h2)}   p = {h2['p']:.4f}")
        print(f"      -> {'HOLDS' if h2_pass else 'REFUTED'}")
        if not h2_pass:
            print("      Refutation reading: the gain rides on the arbitrary positional")
            print("      component, i.e. it is overfitting to the training arenas.")

    if h3 is None or h2 is None:
        print("\nH3  UNSCORABLE.")
        h3_pass = False
    else:
        h3_pass = (not h3["excludes_zero"]) and h3["mean"] < h2["mean"]
        print(f"\nH3  SHF - ctl2 on score CI includes 0  AND  SHF < PAR")
        print(f"      {fmt(h3)}   p = {h3['p']:.4f}   SHF {h3['mean']:+.3f} "
              f"{'<' if h3['mean'] < h2['mean'] else '>='} PAR {h2['mean']:+.3f}")
        print(f"      -> {'HOLDS' if h3_pass else 'REFUTED'}")
        # H3 "holds" by a CI *including* zero, which is an accept-the-null test:
        # low power makes it easier to pass, not harder. It is also the leg that
        # can fire CONTINUE on its own, so the caveat printed for `won` below
        # must be printed here too. Audit 9 caught the asymmetry: the same
        # mistake as E36's P5, moved onto the go/no-go decision.
        if h3_pass and h1 is not None:
            print(f"      CAUTION: this is a null, not a demonstration. Observed MDE "
                  f"{h3['mde']:.3f};")
            print(f"      the CI tolerates up to {h3['hi']:+.3f}, i.e. "
                  f"{h3['hi'] / h1['mean']:.0%} of PLB2's own effect ({h1['mean']:+.3f}).")
            # Bar reused from H2: an effect at least half the treatment's is the
            # size this entry already agreed is worth caring about. If the null's
            # CI admits that, the null is not established, it is merely unmeasured.
            if h3["hi"] >= H2_FRACTION * h1["mean"]:
                print("      At this power H3 CANNOT distinguish 'the null does nothing'")
                print("      from 'the null does much of what PLB2 does'. Treat as UNDEFINED,")
                print("      and note H3 is a leg of the CONTINUE rule.")
        if not h3_pass:
            print("      Refutation reading: any 4-way positional split works, so the gain")
            print("      is capacity rather than information.")

    # ---- guards ----------------------------------------------------------
    print(f"\n{'-' * 78}\nGuards (P4)\n{'-' * 78}")
    for arm in ARMS:
        if not data.get(arm):
            continue
        runs = list(data[arm].values())
        won = statistics.mean(r["won"] for r in runs)
        crates = statistics.mean(r["crates"] for r in runs)
        maxms = max(r["think_max_ms"] for r in runs)
        flags = []
        if won < GUARD_WON_MIN:
            flags.append(f"won {won:.3f} < {GUARD_WON_MIN}")
        if crates < GUARD_CRATES_MIN:
            flags.append(f"crates {crates:.2f} < {GUARD_CRATES_MIN}")
        if maxms >= GUARD_THINK_MAX_MS:
            flags.append(f"think_max {maxms:.2f} >= {GUARD_THINK_MAX_MS}")
        print(f"  {arm:>5}  won {won:.3f}  crates {crates:.2f}  think_max {maxms:.2f} ms"
              f"   {'FAIL: ' + '; '.join(flags) if flags else 'pass'}")
    print("\n  The fourth guard -- all-zero-row share < 0.01 -- is NOT in the evaluation")
    print("  CSV. It needs a rollout probe through the agent's own state_to_features")
    print("  with the arm's BM_D8 exported. Not faked here; run it separately.")

    # ---- won, reported and not tested ------------------------------------
    print(f"\n{'-' * 78}\nSecondary: `won`, reported with its MDE (entry point 5)\n{'-' * 78}")
    print(f"  Pre-registered n = 15 MDE on won: {WON_MDE_N15:.3f}")
    for arm in ARMS:
        if arm == CONTROL:
            continue
        d = contrasts.get((arm, "won"))
        if d is None:
            continue
        verdict = ("moved" if d["excludes_zero"]
                   else f"null, but |effect| < MDE {d['mde']:.3f} -- UNDERPOWERED, "
                        f"not evidence of absence")
        print(f"  {arm:>5}  {fmt(d)}  observed MDE {d['mde']:.3f}   {verdict}")
    print("\n  No tournament-ranking claim may be made from `score` -- the entry says so.")

    # ---- continuation ----------------------------------------------------
    print(f"\n{'=' * 78}")
    go = h1_pass and (h2_pass or h3_pass)
    print("CONTINUATION RULE (pre-committed): only if H1 and (H2 or H3) hold does a")
    print("`won`-powered confirmation at n = 20-30 follow. Otherwise rung 4 closes.")
    print(f"  H1 {'HOLDS' if h1_pass else 'fails'} · H2 {'HOLDS' if h2_pass else 'fails'} "
          f"· H3 {'HOLDS' if h3_pass else 'fails'}")
    print(f"  -> {'CONTINUE to the confirmation sweep.' if go else 'RUNG 4 CLOSES. Write the report.'}")
    print(f"{'=' * 78}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ep", type=int, nargs="+", default=[20000],
                    help="checkpoints to report; 20000 is the pre-registered one")
    ap.add_argument("--markdown", action="store_true",
                    help="pipe-separated cells, for pasting into the ledger")
    args = ap.parse_args()
    for ep in args.ep:
        report(ep, args.markdown)


if __name__ == "__main__":
    main()
