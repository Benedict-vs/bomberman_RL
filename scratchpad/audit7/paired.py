#!/usr/bin/env python3
"""Audit 7: paired bootstrap CIs for the four conditional-rule arms."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
arms = {}
for a in ("ctl", "esc", "near", "far"):
    arms[a] = json.loads((HERE / f"cond_{a}.json").read_text())

METRICS = ["score", "won", "crates", "bombs", "suicides", "survived", "kills", "coins", "died"]
rng = np.random.default_rng(20260816)


def boot(diff: np.ndarray, n: int = 20000) -> tuple[float, float]:
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    means = diff[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


print(f"n = {arms['ctl']['n']} paired rounds, ship seed {arms['ctl']['seed']}, "
      f"shipped table, 3 x rule_based_agent\n")
print(f"{'arm':6s} {'fire%':>7s} " + " ".join(f"{m:>9s}" for m in METRICS))
for a in ("ctl", "esc", "near", "far"):
    r = arms[a]["rows"]
    print(f"{a:6s} {100*arms[a]['fire_rate']:7.2f} "
          + " ".join(f"{np.mean([x[m] for x in r]):9.3f}" for m in METRICS))

print("\npaired differences vs ctl, 95 % bootstrap CI")
base = {m: np.array([x[m] for x in arms["ctl"]["rows"]], dtype=float) for m in METRICS}
for a in ("esc", "near", "far"):
    print(f"\n-- {a} - ctl   (rule fires on {100*arms[a]['fire_rate']:.2f} % of steps)")
    for m in METRICS:
        v = np.array([x[m] for x in arms[a]["rows"]], dtype=float)
        d = v - base[m]
        lo, hi = boot(d)
        flag = "  *" if (lo > 0 or hi < 0) else ""
        print(f"   {m:10s} {d.mean():+8.3f}  [{lo:+.3f}, {hi:+.3f}]{flag}")

# near + far should reconstruct esc if the two halves are additive
print("\nadditivity check (near-ctl) + (far-ctl) vs (esc-ctl):")
for m in ("score", "won"):
    dn = np.array([x[m] for x in arms["near"]["rows"]], dtype=float) - base[m]
    df = np.array([x[m] for x in arms["far"]["rows"]], dtype=float) - base[m]
    de = np.array([x[m] for x in arms["esc"]["rows"]], dtype=float) - base[m]
    print(f"   {m:6s} near {dn.mean():+.3f} + far {df.mean():+.3f} = {dn.mean()+df.mean():+.3f}"
          f"   vs esc {de.mean():+.3f}")

# far vs esc directly: is restricting the rule better than the rule?
print("\nfar - esc (does the restriction beat the unconditional rule?)")
for m in ("score", "won", "suicides", "crates", "survived"):
    v = np.array([x[m] for x in arms["far"]["rows"]], dtype=float)
    w = np.array([x[m] for x in arms["esc"]["rows"]], dtype=float)
    d = v - w
    lo, hi = boot(d)
    flag = "  *" if (lo > 0 or hi < 0) else ""
    print(f"   {m:10s} {d.mean():+8.3f}  [{lo:+.3f}, {hi:+.3f}]{flag}")
