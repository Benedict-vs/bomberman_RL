#!/usr/bin/env python3
"""Audit 10 -- does the ceiling harness's k=-1 arm reproduce the SHIPPED agent (E37)?"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

REPO = Path("/Users/benedictvonschubert/Projects/bomberman_RL")
CEIL = REPO / "scratchpad/strategy/ceil"
EVAL = REPO / "results/eval/task4_tournament"
rng = np.random.default_rng(777)


def load(path, code=None):
    rows = list(csv.DictReader(open(path)))
    codes = {r["code"] for r in rows}
    if code is None:
        code = "user_agent" if "user_agent" in codes else "benedict_task4"
    mine = sorted([r for r in rows if r["code"] == code], key=lambda r: int(r["round"]))
    out = {}
    for k in mine[0]:
        try:
            out[k] = np.array([float(r[k]) for r in mine])
        except ValueError:
            pass
    return out


def boot(a, b, n=20000):
    d = np.asarray(b, float) - np.asarray(a, float)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


ceil = load(CEIL / "huntceil4k_k-1__task4_rb_ship990731.csv")
ceil1k = {k: v[:1000] for k, v in ceil.items()}
pilot = load(CEIL / "huntceil_k-1__task4_rb_ship990731.csv")
s105 = load(EVAL / "benedict_q_e37_PLB2_s105__ep20000__task4_rb_ship990731.csv")
s106 = load(EVAL / "benedict_q_e37_PLB2_s106__ep20000__task4_rb_ship990731.csv")

M = ("score", "kills", "coins", "crates", "bombs", "won", "suicides", "survived", "steps",
     "invalid", "moves")
for name, d in [("E37 s105", s105), ("E37 s106", s106), ("ceil k=-1 [0:1000]", ceil1k),
                ("ceil k=-1 pilot", pilot), ("ceil k=-1 n=4000", ceil)]:
    print(f"{name:22s} " + "  ".join(f"{m}={d[m].mean():.3f}" for m in M if m in d))

print()
for lbl, ref in [("s105", s105), ("s106", s106)]:
    for cname, c in [("ceil[0:1000]", ceil1k), ("ceil pilot", pilot)]:
        print(f"\n--- paired {lbl} -> {cname}")
        for m in M:
            if m not in ref or m not in c:
                continue
            d, lo, hi = boot(ref[m], c[m])
            print(f"   {m:9s} {ref[m].mean():8.4f} -> {c[m].mean():8.4f}  d={d:+8.4f} [{lo:+.4f},{hi:+.4f}]")

# Per-round agreement: how often does the ceiling control land on the same score as the
# real agent in the same arena?
print("\n--- per-round agreement on identical arenas (s105 vs ceil k=-1 [0:1000])")
for m in ("score", "steps", "crates", "coins", "survived"):
    eq = (s105[m] == ceil1k[m]).mean()
    print(f"   {m:9s} identical in {eq:.1%} of rounds; corr={np.corrcoef(s105[m], ceil1k[m])[0,1]:+.3f}")
