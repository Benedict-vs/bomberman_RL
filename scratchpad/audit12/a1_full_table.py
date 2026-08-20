"""Audit 12 / A1: full metric table for all three fields, both arms.

The E42 entry reports survived/suicides ONLY on the guard field and score/margin only
in-distribution. The credit-assignment claim is a claim about learning on the training
field, so the in-distribution behavioural metrics are the ones that test it. Print all.

Adds, beyond scratchpad/benedict/e42_analyze.py:
  * every metric on every field
  * pooled crates/bombs (ratio-of-sums) beside the mean-of-ratios the entry used
  * an arena-paired (round-level) difference as a sensitivity
  * the AGENTS.md fragility check the E42 scorer omits: does the bootstrap CI's
    verdict survive a different bootstrap seed?
"""
from __future__ import annotations
import csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

EVAL = Path("results/eval/task4_tournament")
OURS = "benedict_task4"
MIX = list(range(200, 208))
CTL = list(range(100, 108))
FIELDS = ["heldout", "indist", "guard"]


def load(arm, seed, field):
    f = EVAL / f"e42_{arm}_s{seed}__task4_{field}_val550731.csv"
    by = {}
    for r in csv.DictReader(open(f)):
        by.setdefault(int(r["round"]), []).append(r)
    return by


def per_round(by, name):
    out = {}
    for rd, rows in by.items():
        me = next(r for r in rows if r["code"] == OURS)
        opp = [float(r["score"]) for r in rows if r["code"] != OURS]
        if name == "margin_mean":
            out[rd] = float(me["score"]) - float(np.mean(opp))
        elif name == "margin_best":
            out[rd] = float(me["score"]) - max(opp)
        elif name == "crates_per_bomb":
            out[rd] = float(me["crates"]) / max(float(me["bombs"]), 1e-9)
        elif name == "zero_bomb_round":
            out[rd] = float(float(me["bombs"]) == 0)
        elif name == "opp_score":
            out[rd] = float(np.mean(opp))
        else:
            out[rd] = float(me[name])
    return out


def pooled_ratio(by, num, den):
    n = d = 0.0
    for rows in by.values():
        me = next(r for r in rows if r["code"] == OURS)
        n += float(me[num]); d += float(me[den])
    return n / d if d else float("nan")


def boot_ci(a, b, seed, n=10000):
    rng = np.random.default_rng(seed)
    d = [np.mean(rng.choice(b, len(b))) - np.mean(rng.choice(a, len(a))) for _ in range(n)]
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def perm_p(a, b, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    obs = abs(np.mean(b) - np.mean(a))
    pool = np.concatenate([a, b]); na = len(a); c = 0
    for _ in range(n):
        rng.shuffle(pool)
        if abs(pool[na:].mean() - pool[:na].mean()) >= obs: c += 1
    return c / n


METRICS = ["score", "margin_mean", "opp_score", "coins", "kills", "suicides",
           "killed_by_opponent", "survived", "crates", "bombs", "crates_per_bomb",
           "zero_bomb_round", "steps", "invalid"]

cache = {}
for arm, seeds in (("ctl", CTL), ("mix", MIX)):
    for s in seeds:
        for f in FIELDS:
            cache[(arm, s, f)] = load(arm, s, f)

for field in FIELDS:
    print("=" * 108)
    print(f"FIELD {field}")
    print(f"  {'metric':<20}{'ctl':>9}{'mix':>9}{'diff':>9}{'CI(12345)':>20}{'CI(999)':>20}{'p':>8}  flag")
    for m in METRICS:
        a = np.array([np.mean(list(per_round(cache[("ctl", s, field)], m).values())) for s in CTL])
        b = np.array([np.mean(list(per_round(cache[("mix", s, field)], m).values())) for s in MIX])
        d = b.mean() - a.mean()
        lo1, hi1 = boot_ci(a, b, 12345)
        lo2, hi2 = boot_ci(a, b, 999)
        p = perm_p(a, b)
        v1, v2 = lo1 * hi1 > 0, lo2 * hi2 > 0
        flag = "FRAGILE" if (v1 != v2) or (v1 != (p < 0.05)) else ("sig" if v1 and p < .05 else "")
        print(f"  {m:<20}{a.mean():9.3f}{b.mean():9.3f}{d:+9.3f}"
              f"{f'[{lo1:+.3f},{hi1:+.3f}]':>20}{f'[{lo2:+.3f},{hi2:+.3f}]':>20}{p:8.4f}  {flag}")
    # pooled crates/bomb
    pa = np.array([pooled_ratio(cache[("ctl", s, field)], "crates", "bombs") for s in CTL])
    pb = np.array([pooled_ratio(cache[("mix", s, field)], "crates", "bombs") for s in MIX])
    lo, hi = boot_ci(pa, pb, 12345)
    print(f"  {'crates/bomb POOLED':<20}{pa.mean():9.3f}{pb.mean():9.3f}{pb.mean()-pa.mean():+9.3f}"
          f"{f'[{lo:+.3f},{hi:+.3f}]':>20}{'':>20}{perm_p(pa,pb):8.4f}")
    # per-seed spread on the decisive metrics
    for m in ("score", "suicides", "survived"):
        a = np.array([np.mean(list(per_round(cache[("ctl", s, field)], m).values())) for s in CTL])
        b = np.array([np.mean(list(per_round(cache[("mix", s, field)], m).values())) for s in MIX])
        print(f"    per-seed {m:<12} ctl {np.round(a,3)}  sd={a.std(ddof=1):.3f}")
        print(f"    {'':<21}mix {np.round(b,3)}  sd={b.std(ddof=1):.3f}")
