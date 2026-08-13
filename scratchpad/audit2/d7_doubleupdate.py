"""Prove the double terminal call from the training log alone.

reward_from_events() is called once per step the agent survives, and once more
in end_of_round with the SAME event list.  So the logged episode reward should
satisfy

    reward == sum(count[ev] * REWARDS[ev]) + STEP_COST * (steps + survived)

with the +survived term being the extra call.  If instead the correct
`steps` alone fits, there is no extra call.
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"scratchpad"/"audit"))
from lib import load

R = {"COIN_COLLECTED": 5.0, "CRATE_DESTROYED": 0.3, "INVALID_ACTION": -1.0,
     "WAITED": -0.1, "KILLED_SELF": 0.0, "GOT_KILLED": -5.0, "KILLED_OPPONENT": 25.0}

d = load(str(ROOT/"results/train/task3_opponents/benedict_task3__q_e26_audit2_base_s40.csv"))
ev = sum(R[k]*d[k] for k in R)
for label, extra in [("steps only (no double call)", 0), ("steps + survived (double call)", 1)]:
    pred = ev - 0.1*(d["steps"] + extra*d["SURVIVED_ROUND"])
    err = pred - d["reward"]
    print(f"{label:<34} max|err| {np.abs(err).max():.6f}  mean err {err.mean():+.6f}  "
          f"exact on {np.mean(np.abs(err)<1e-6)*100:5.1f}% of episodes")

surv = d["SURVIVED_ROUND"] > 0
print(f"\nepisodes where the agent is alive at end_round (double update fires): "
      f"{surv.mean()*100:.1f}%  (n={len(surv)})")
print(f"steps/episode {d['steps'].mean():.1f} -> the extra update is 1 in "
      f"{d['steps'][surv].mean():.0f} of that episode's updates")
