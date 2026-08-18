#!/usr/bin/env python3
"""Audit 10 -- attack the hunt-ceiling verdict. numpy + csv only (no pandas here)."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np

REPO = Path("/Users/benedictvonschubert/Projects/bomberman_RL")
CEIL = REPO / "scratchpad/strategy/ceil"
EVAL = REPO / "results/eval/task4_tournament"

rng = np.random.default_rng(12345)


def load(path: Path, code: str | None = None) -> dict[str, np.ndarray]:
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    codes = {r["code"] for r in rows}
    if code is None:
        code = "user_agent" if "user_agent" in codes else sorted(codes - {
            "rule_based_agent", "coin_collector_agent", "peaceful_agent", "random_agent"})[0]
    mine = [r for r in rows if r["code"] == code]
    mine.sort(key=lambda r: int(r["round"]))
    out = {}
    for k in mine[0]:
        try:
            out[k] = np.array([float(r[k]) for r in mine])
        except ValueError:
            pass
    return out


def slice_(d: dict, lo: int, hi: int) -> dict:
    return {k: v[lo:hi] for k, v in d.items()}


def boot_paired(a, b, n=20000):
    d = np.asarray(b, float) - np.asarray(a, float)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


METRICS = ("score", "kills", "coins", "crates", "won", "suicides", "survived", "bombs")


def cmp(name, A, B, metrics=METRICS):
    assert (A["seed"] == B["seed"]).all(), "arena seeds do not line up"
    print(f"\n=== {name}   n={len(A['score'])}")
    for m in metrics:
        if m not in A or m not in B:
            continue
        x, y = A[m], B[m]
        mean, lo, hi = boot_paired(x, y)
        r = np.corrcoef(x, y)[0, 1] if x.std() > 0 and y.std() > 0 else float("nan")
        se_unp = np.sqrt(x.var(ddof=1) / len(x) + y.var(ddof=1) / len(y))
        se_p = (y - x).std(ddof=1) / np.sqrt(len(x))
        print(f"  {m:9s} A={x.mean():7.4f} B={y.mean():7.4f} d={mean:+7.4f} "
              f"[{lo:+7.4f},{hi:+7.4f}]  r={r:+.3f}  se_paired={se_p:.4f} se_unpaired={se_unp:.4f}")


big = {k: load(CEIL / f"huntceil4k_k{k}__task4_rb_ship990731.csv") for k in (-1, 0, 4)}
pil = {k: load(CEIL / f"huntceil_k{k}__task4_rb_ship990731.csv") for k in (-1, 0, 2, 4, 8)}
ship = load(EVAL / "benedict_q_e28_F4__task4_rb_ship990731.csv")

print("### 1. Does the k=-1 control arm reproduce the shipped agent? (paired, same arenas)")
cmp("shipped(evaluate.py n=1000) -> ceil k=-1 big[0:1000]", ship, slice_(big[-1], 0, 1000))
cmp("shipped(evaluate.py n=1000) -> ceil k=-1 pilot n=1000", ship, pil[-1])

print("\n### 2. A/A inside the ceiling harness: pilot vs first 1000 of the 4k run")
cmp("pilot k=-1 -> big k=-1[0:1000]", pil[-1], slice_(big[-1], 0, 1000))
cmp("pilot k=0  -> big k=0[0:1000]", pil[0], slice_(big[0], 0, 1000))
cmp("pilot k=4  -> big k=4[0:1000]", pil[4], slice_(big[4], 0, 1000))

print("\n### 3. The headline verdict, reproduced")
cmp("big k=-1 -> big k=4 (n=4000)", big[-1], big[4])
cmp("big k=-1 -> big k=0 (n=4000)", big[-1], big[0])

print("\n### 4. Winner's curse split: pilot-seen arenas vs fresh arenas")
cmp("k=-1 -> k=4, rounds 0..999 (selection arenas)",
    slice_(big[-1], 0, 1000), slice_(big[4], 0, 1000))
cmp("k=-1 -> k=4, rounds 1000..3999 (fresh arenas)",
    slice_(big[-1], 1000, 4000), slice_(big[4], 1000, 4000))

print("\n### 5. Pilot arm table (the selection data)")
for k, d in pil.items():
    print(f"  k={k:3d}  score {d['score'].mean():.3f}  kills {d['kills'].mean():.3f}"
          f"  coins {d['coins'].mean():.3f}  crates {d['crates'].mean():.2f}")
for k in (0, 2, 4, 8):
    mean, lo, hi = boot_paired(pil[-1]["score"], pil[k]["score"])
    print(f"  pilot k={k} - k=-1: {mean:+.3f} [{lo:+.3f},{hi:+.3f}]")

print("\n### 6. Score decomposition at n=4000")
for k in (0, 4):
    a, b = big[-1], big[k]
    dk = (b["kills"] - a["kills"]).mean()
    dc = (b["coins"] - a["coins"]).mean()
    ds = (b["score"] - a["score"]).mean()
    print(f"  k={k}: dscore={ds:+.4f}  5*dkills={5*dk:+.4f}  dcoins={dc:+.4f}  sum={5*dk+dc:+.4f}")

print("\n### 7. How often does the override actually change the round at all?")
for k in (0, 4):
    same = (big[k]["score"] == big[-1]["score"]) & (big[k]["steps"] == big[-1]["steps"]) & \
           (big[k]["crates"] == big[-1]["crates"]) & (big[k]["coins"] == big[-1]["coins"])
    print(f"  k={k}: rounds byte-identical on (score,steps,crates,coins): "
          f"{same.mean():.1%}  ({int(same.sum())}/{len(same)})")
    ch = ~same
    d = big[k]["score"][ch] - big[-1]["score"][ch]
    m, lo, hi = boot_paired(big[-1]["score"][ch], big[k]["score"][ch])
    print(f"        on the {ch.sum()} changed rounds: dscore {m:+.3f} [{lo:+.3f},{hi:+.3f}]")
