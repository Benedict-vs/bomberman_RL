"""Phase-1 vs phase-2 survival across the rung-4 arms.

`round_economy.py` measured that all 9 coins and all 122 crates are gone by
step ~200, so a round has an economy phase and an empty phase. This asks, of
every arm that "bought survival", whether the survival it bought was in the
phase where score exists.

  P1 = share of rounds we are still alive at step 200 (steps >= 200)
  P2 = share of the P1 survivors that reach 400

    uv run python scratchpad/strategy/phase_split.py 'e36_OPP*ep20000*val550731' ...
"""
from __future__ import annotations

import csv, glob, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "results/eval/task4_tournament"
US = "benedict_task4"
NUM = ("survived","round_steps","score","coins","kills","suicides","crates",
       "bombs","invalid","steps","died","killed_by_opponent","rank","won")


def arm(pattern, code=US):
    files = sorted(glob.glob(str(D / pattern)))
    if not files:
        return None
    rows = []
    for f in files:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r["code"] != code:
                    continue
                for k in NUM: r[k] = float(r[k])
                rows.append(r)
    g = lambda k: np.array([r[k] for r in rows])
    st = g("steps")
    p1 = (st >= 200).mean()
    p2 = (st >= 399).sum() / max((st >= 200).sum(), 1)
    return dict(n=len(rows), files=len(files), score=g("score").mean(),
                coins=g("coins").mean(), kills=g("kills").mean(),
                won=g("won").mean(), surv=g("survived").mean(),
                suic=g("suicides").mean(), steps=st.mean(),
                p1=p1, p2=p2, crates=g("crates").mean(), bombs=g("bombs").mean(),
                # score earned per round conditional on reaching step 200
                s_p1=g("score")[st < 200].mean() if (st < 200).any() else float("nan"),
                s_p2=g("score")[st >= 200].mean())


PATTERNS = sys.argv[1:] or [
    ("E33 ctl @20k",  "benedict_q_e33_ctl_s10?__ep20000__task4_rb_val550731.csv"),
    ("E33 F005",      "benedict_q_e33_F005_s10?__ep20000__task4_rb_val550731.csv"),
    ("E33 F020",      "benedict_q_e33_F020_s10?__ep20000__task4_rb_val550731.csv"),
    ("E33 F080",      "benedict_q_e33_F080_s10?__ep20000__task4_rb_val550731.csv"),
    ("E36 OPP @20k",  "benedict_q_e36_OPP_s10?__ep20000__task4_rb_val550731.csv"),
    ("E36 PLB @20k",  "benedict_q_e36_PLB_s10?__ep20000__task4_rb_val550731.csv"),
    ("E37 PLB2 ship", "benedict_q_e37_PLB2_s106__ep20000__task4_rb_ship990731.csv"),
    ("E37 ctl2 ship", "benedict_q_e37_ctl2_s113__ep20000__task4_rb_ship990731.csv"),
    ("ref rule_based","ref_rule_based_agent__task4_rb_ship990731.csv"),
]

print(f"{'arm':<16}{'files':>5}{'n':>7}{'score':>7}{'coins':>7}{'kills':>7}"
      f"{'won':>7}{'surv':>7}{'suic':>7}{'steps':>7}{'P1':>7}{'P2':>7}"
      f"{'crates':>7}{'bombs':>7}{'sc|die<200':>11}{'sc|live200':>11}")
for label, pat in PATTERNS:
    a = arm(pat, "rule_based_agent" if "ref_rule" in pat else US)
    if a is None:
        print(f"{label:<16}  -- no files for {pat}")
        continue
    print(f"{label:<16}{a['files']:>5}{a['n']:>7}{a['score']:>7.3f}{a['coins']:>7.3f}"
          f"{a['kills']:>7.3f}{a['won']:>7.3f}{a['surv']:>7.3f}{a['suic']:>7.3f}"
          f"{a['steps']:>7.1f}{a['p1']:>7.3f}{a['p2']:>7.3f}{a['crates']:>7.2f}"
          f"{a['bombs']:>7.2f}{a['s_p1']:>11.3f}{a['s_p2']:>11.3f}")
