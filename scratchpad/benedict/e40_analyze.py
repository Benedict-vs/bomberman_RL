#!/usr/bin/env python3
"""E40 scoring. Every comparison carries a bootstrap CI *and* a sign-flip p:
audit 10 showed analyze.py's hard-coded default_rng(12345) makes a borderline
bound look deterministic when it is a coin flip, so a claim counts here only if
both agree. Pre-registered metrics only; nothing is added after the fact.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent / "e40"
OURS = "user_agent"


def load(tag: str) -> dict[int, list[dict]]:
    by_round: dict[int, list[dict]] = defaultdict(list)
    with open(D / f"e40_{tag}__task4_rb_ship990731.csv") as fh:
        for row in csv.DictReader(fh):
            by_round[int(row["round"])].append(row)
    return by_round


def me(rows, key):
    return float(next(r for r in rows if r["code"] == OURS)[key])


def opps(rows):
    return [float(r["score"]) for r in rows if r["code"] != OURS]


METRICS = {
    "score":       lambda rows: me(rows, "score"),
    "margin_mean": lambda rows: me(rows, "score") - float(np.mean(opps(rows))),
    "margin_best": lambda rows: me(rows, "score") - max(opps(rows)),
    "kills":       lambda rows: me(rows, "kills"),
    "won":         lambda rows: float(next(r for r in rows if r["code"] == OURS)["won"]),
    "coins":       lambda rows: me(rows, "coins"),
    "crates":      lambda rows: me(rows, "crates"),
    "suicides":    lambda rows: me(rows, "suicides"),
    "survived":    lambda rows: me(rows, "survived"),
}


def boot(d, seed, n=2000):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def signflip(d, n=5000):
    rng = np.random.default_rng(0)
    obs = abs(d.mean())
    null = (rng.choice([-1.0, 1.0], size=(n, len(d))) * d).mean(axis=1)   # sign then mean
    return float((np.abs(null) >= obs).mean())


def compare(a_tag, b_tag, metrics=None):
    A, B = load(a_tag), load(b_tag)
    common = sorted(set(A) & set(B))
    print(f"\n### {b_tag}  vs  {a_tag}   ({len(common)} paired arenas)\n")
    print(f"| metric | {a_tag} | {b_tag} | paired difference | sign-flip p | verdict |")
    print("|---|---|---|---|---|---|")
    for name in (metrics or METRICS):
        fn = METRICS[name]
        va = np.array([fn(A[r]) for r in common])
        vb = np.array([fn(B[r]) for r in common])
        d = vb - va
        lo, hi = boot(d, 12345)
        stable = sum(boot(d, s)[0] > 0 or boot(d, s)[1] < 0 for s in range(15)) / 15
        se = d.std(ddof=1)/np.sqrt(len(d)); t = d.mean()/se
        p = signflip(d)
        ok = lo * hi > 0 and p < 0.05
        verdict = ("BESSER" if d.mean() > 0 else "SCHLECHTER") if ok else "nicht gezeigt"
        flag = "" if (lo * hi > 0) == (p < 0.05) else "  <-- CI and p disagree"
        print(f"| {name} | {va.mean():.3f} | {vb.mean():.3f} | "
              f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | t={t:+.2f}, p={p:.4f} | {verdict}{flag} |")
        if 0 < stable < 1:
            print(f"|  ^ | | | seed-stability of that verdict: {stable:.0%} of 15 bootstrap seeds | | |")


print("# E40 — results\n")
print("## Control validation (shipped agent published: score 3.949, kills 0.226, "
      "suicides 0.488, crates 33.55)\n")
ctl = load("ctl")
for k in ("score", "kills", "suicides", "crates", "won"):
    v = np.mean([METRICS[k](rows) for rows in ctl.values()])
    print(f"- {k}: {v:.3f}")

print("\n## P1 + P4 — the corrected oracle against the control")
compare("ctl", "k4sim")
print("\n## P2 — corrected oracle against the broken one (the reason this entry exists)")
compare("k4stale", "k4sim", ["score", "margin_mean", "margin_best", "kills", "crates"])
print("\n## Bridge — the broken oracle against the control (reproduces E39)")
compare("ctl", "k4stale", ["score", "margin_mean", "margin_best", "kills", "crates"])
print("\n## P5 — reach")
compare("k4sim", "k8sim", ["score", "margin_mean", "margin_best", "kills", "crates"])
