"""Independent verification of audit 10's load-bearing claims (F1, F3, F4, F6).

Written from the CSVs directly, not by re-running the auditor's code, so an error in
their harness cannot propagate into the check. F2 (the 8.1 % trap conversion) needs
their instrumented rollout and is not verified here.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CEIL = ROOT / "scratchpad/strategy/ceil"


def load(path: Path) -> dict[int, list[dict]]:
    by_round: dict[int, list[dict]] = defaultdict(list)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            by_round[int(row["round"])].append(row)
    return by_round


def paired(metric_fn, a: Path, b: Path) -> np.ndarray:
    """Per-round metric difference b - a over the rounds present in both files."""
    ra, rb = load(a), load(b)
    common = sorted(set(ra) & set(rb))
    return np.array([metric_fn(rb[r]) - metric_fn(ra[r]) for r in common])


OURS = {"user_agent", "benedict_task4"}   # ceiling harness vs evaluate.py naming


def me(rows: list[dict]) -> dict:
    return next(r for r in rows if r["agent"] in OURS)


def ours(rows: list[dict]) -> float:
    return float(me(rows)["score"])


def opps(rows: list[dict]) -> list[float]:
    return [float(r["score"]) for r in rows if r["agent"] not in OURS]


def margin_best(rows: list[dict]) -> float:
    return ours(rows) - max(opps(rows))


def margin_mean(rows: list[dict]) -> float:
    return ours(rows) - float(np.mean(opps(rows)))


def boot_ci(d: np.ndarray, seed: int, n: int = 2_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def signflip_p(d: np.ndarray, seed: int = 0, n: int = 20_000) -> float:
    rng = np.random.default_rng(seed)
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n, len(d)))
    null = (signs * d).mean(axis=1)
    return float((np.abs(null) >= obs).mean())


def describe(name: str, d: np.ndarray) -> None:
    se = d.std(ddof=1) / np.sqrt(len(d))
    lo, hi = boot_ci(d, 12345)          # analyze.py's hard-coded seed
    lows = [boot_ci(d, s)[0] for s in range(200)]  # 2k resamples each: enough to show spread
    excl = sum(l > 0 for l in lows) / len(lows)
    print(f"  {name:<12} d={d.mean():+.4f}  se={se:.4f}  t={d.mean()/se:+.2f}  "
          f"CI(seed 12345)=[{lo:+.4f},{hi:+.4f}]  "
          f"P(CI excludes 0 over 200 seeds)={excl:.1%}  signflip p={signflip_p(d):.4f}")


CTL = CEIL / "huntceil4k_k-1__task4_rb_ship990731.csv"
K4 = CEIL / "huntceil4k_k4__task4_rb_ship990731.csv"

print("=" * 78)
print("F1 + F3 -- k=4 vs control, n=4000, verified independently of analyze.py")
print("=" * 78)
for label, fn in [("score", ours), ("margin_mean", margin_mean), ("margin_best", margin_best)]:
    describe(label, paired(fn, CTL, K4))

print()
print("Fresh arenas only (rounds 1000-3999; rounds 0-999 were the k=4 selection set):")
ra, rb = load(CTL), load(K4)
fresh = sorted(r for r in (set(ra) & set(rb)) if r >= 1000)
for label, fn in [("score", ours), ("margin_mean", margin_mean), ("margin_best", margin_best)]:
    describe(label, np.array([fn(rb[r]) - fn(ra[r]) for r in fresh]))

print()
print("Did anyone die more, or did credit just move? (k=4 minus control, n=4000)")
for label, fn in [
    ("our kills", lambda rows: float(me(rows)["kills"])),
    ("opp kills", lambda rows: sum(float(r["kills"]) for r in rows if r["agent"] not in OURS)),
    ("opp deaths", lambda rows: sum(1 - float(r["survived"]) for r in rows if r["agent"] not in OURS)),
]:
    d = paired(fn, CTL, K4)
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"  {label:<12} d={d.mean():+.4f}  t={d.mean()/se:+.2f}")

print()
print("=" * 78)
print("F6 -- how many rows did training actually touch?")
print("=" * 78)
q = np.load(ROOT / "agent_code/benedict_task4/q_table.npy")
nz = int((np.abs(q).sum(axis=-1) > 0).sum())
print(f"  table shape {q.shape}, storage rows {q.shape[0]}, rows with any nonzero Q: {nz}")

print()
print("=" * 78)
print("F4 -- the four-field kill pool, decomposed from the CSVs")
print("=" * 78)
FIELDS = [
    ("peaceful", ROOT / "scratchpad/strategy/fields/ship_e37__task4_field_peaceful.csv"),
    ("coin_coll", ROOT / "scratchpad/strategy/fields/ship_e37__task4_field_coin_collector.csv"),
    ("mixed", ROOT / "scratchpad/strategy/fields/ship_e37__task4_field_mixed.csv"),
    ("rule_based", ROOT / "results/eval/task4_tournament/"
                          "benedict_task4_shipped_e37__task4_rb_ship990731.csv"),
]
print(f"  {'field':<11}{'n':>5}{'ourkills':>10}{'oppdeaths':>11}{'oppsuic':>9}"
      f"{'takeable':>10}{'ourshare':>10}")
for name, path in FIELDS:
    if not path.exists():
        print(f"  {name:<11} MISSING {path}")
        continue
    by_round = load(path)
    n = len(by_round)
    ok = np.mean([float(me(rows)["kills"]) for rows in by_round.values()])
    od = np.mean([sum(1 - float(r["survived"]) for r in rows if r["agent"] not in OURS)
                  for rows in by_round.values()])
    os_ = np.mean([sum(float(r["suicides"]) for r in rows if r["agent"] not in OURS)
                   for rows in by_round.values()])
    takeable = od - os_
    print(f"  {name:<11}{n:>5}{ok:>10.3f}{od:>11.3f}{os_:>9.3f}{takeable:>10.3f}"
          f"{ok / takeable:>9.1%}")
