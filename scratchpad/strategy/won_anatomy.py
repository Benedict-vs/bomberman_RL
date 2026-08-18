"""What actually decides `won` in our rung-4 rounds?

Reads committed per-round eval CSVs (one row per agent per round). Pure post-hoc
analysis on data already on disk -- nothing is trained or re-run.

Questions:
  - is winning survival or scoring?
  - the margin distribution against the best opponent, and hence the exact
    score -> won conversion curve (no model fitted, just recount the rounds)
  - the opponent-death pool and who takes it
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
US = "benedict_task4"
NUM = ("survived", "round_steps", "score", "coins", "kills", "suicides",
       "crates", "bombs", "invalid", "steps", "died", "killed_by_opponent",
       "rank", "won")


def load(path: Path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            for k in NUM:
                r[k] = float(r[k])
            r["round"] = int(r["round"])
            rows.append(r)
    return rows


def report(rows, name):
    by_round = defaultdict(list)
    for r in rows:
        by_round[r["round"]].append(r)
    rounds = sorted(by_round)
    us, best_opp, opp_alive, opps = [], [], [], []
    for k in rounds:
        rs = by_round[k]
        mine = [r for r in rs if r["code"] == US][0]
        others = [r for r in rs if r["code"] != US]
        us.append(mine)
        opps.append(others)
        best_opp.append(max(o["score"] for o in others))
        opp_alive.append(sum(o["survived"] for o in others))
    n = len(rounds)
    g = lambda key: np.array([m[key] for m in us])
    best_opp = np.array(best_opp)
    opp_alive = np.array(opp_alive)
    won, surv, score = g("won"), g("survived"), g("score")

    print(f"\n=== {name}  (n={n}) ===")
    print(f"us: score {score.mean():.3f} = coins {g('coins').mean():.3f} + 5x kills "
          f"{g('kills').mean():.3f} | won {won.mean():.3f} survived {surv.mean():.3f} "
          f"suicides {g('suicides').mean():.3f} killed_by {g('killed_by_opponent').mean():.3f}")
    print(f"best-opponent score {best_opp.mean():.3f}")

    for s in (0, 1):
        m = surv == s
        print(f"  survived={s}: n={int(m.sum()):4d}  P(won)={won[m].mean():.3f}  "
              f"score={score[m].mean():.3f} coins={g('coins')[m].mean():.3f} "
              f"kills={g('kills')[m].mean():.3f} bestopp={best_opp[m].mean():.3f}")

    print("  P(won) by number of opponents still alive at the end:")
    for k in sorted(set(opp_alive.tolist())):
        m = opp_alive == k
        print(f"    {int(k)} alive: n={int(m.sum()):4d} P(won)={won[m].mean():.3f} "
              f"our score={score[m].mean():.3f} bestopp={best_opp[m].mean():.3f} "
              f"our survived={surv[m].mean():.3f}")

    margin = score - best_opp
    print(f"  margin(us - best opp): mean {margin.mean():+.3f} median {np.median(margin):+.1f}")
    base = (margin >= 0).mean()
    print("  score -> won conversion (add d to our score, opponents unchanged):")
    for d in (1, 2, 3, 5, 10):
        w = (margin + d >= 0).mean()
        print(f"    +{d:4.1f} -> won {w:.3f}  (delta {w-base:+.3f}, per point {(w-base)/d:+.4f})")
    for d in (0.5, 1, 2):
        w = (margin - d >= 0).mean()
        print(f"    -{d:4.1f} -> won {w:.3f}  (delta {w-base:+.3f})")

    # a kill is +5 to us AND removes an opponent. Bound the denial side:
    # what does an opponent who dies score vs one who survives?
    dead_sc = [o["score"] for os_ in opps for o in os_ if o["died"]]
    live_sc = [o["score"] for os_ in opps for o in os_ if not o["died"]]
    print(f"  opponent score | died {np.mean(dead_sc):.3f} (n={len(dead_sc)}) "
          f"| survived {np.mean(live_sc):.3f} (n={len(live_sc)})")

    deficits = -margin[margin < 0]
    print(f"  lost {int((margin<0).sum())} rounds. deficit histogram:")
    vals, cnt = np.unique(deficits.astype(int), return_counts=True)
    cum = 0
    for v, c in zip(vals, cnt):
        cum += c
        print(f"    deficit {int(v):3d}: {int(c):4d}  cum {cum/n:.3f} of all rounds")

    opp_deaths = np.array([sum(o["died"] for o in os_) for os_ in opps])
    opp_suic = np.array([sum(o["suicides"] for o in os_) for os_ in opps])
    opp_kills = np.array([sum(o["kills"] for o in os_) for os_ in opps])
    print(f"  opponent deaths/round {opp_deaths.mean():.3f} of which suicides "
          f"{opp_suic.mean():.3f}; our kills {g('kills').mean():.3f}; "
          f"opponent kills {opp_kills.mean():.3f}")
    print(f"  rounds hitting MAX_STEPS: {(g('round_steps')>=400).mean():.3f}")


if __name__ == "__main__":
    d = ROOT / "results/eval/task4_tournament"
    for f in sys.argv[1:] or [
        "benedict_q_e37_PLB2_s106__ep20000__task4_rb_ship990731.csv",
        "benedict_task4_shipped_e37__task4_rb_ship990731.csv",
    ]:
        report(load(d / f), f)
