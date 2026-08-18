"""Audit 11 F7: does the E40 control arm reproduce the SHIPPED table?

Only two committed CSVs are the shipped (E37) table at seed 990731:
  benedict_q_e37_PLB2_s106__ep20000  -> the 3.949 the entry validates against
  benedict_task4_shipped_e37         -> the 3.828 re-run benedict_task4.md 38 documents
(benedict_task4_shipped, 3.718, predates the E37 ship at a76269b and is the OLD table.)
"""
import csv, numpy as np
from collections import defaultdict

def series(path, mine):
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r.get("code", "") == mine:
                out[int(r["round"])] = float(r["score"])
    return out

ctl = series("scratchpad/benedict/e40/e40_ctl__task4_rb_ship990731.csv", "user_agent")
refs = {
 "PLB2_s106 (=the 3.949)": series("results/eval/task4_tournament/benedict_q_e37_PLB2_s106__ep20000__task4_rb_ship990731.csv", "benedict_task4"),
 "shipped_e37 (=the 3.828)": series("results/eval/task4_tournament/benedict_task4_shipped_e37__task4_rb_ship990731.csv", "benedict_task4"),
 "OLD table, pre-E37":      series("results/eval/task4_tournament/benedict_task4_shipped__task4_rb_ship990731.csv", "benedict_task4"),
}
v = np.array([ctl[r] for r in sorted(ctl)])
print(f"e40 ctl n={len(v)} mean={v.mean():.4f}  first-1000 mean={v[:1000].mean():.4f}")
print("ctl per-1000 block means:", " ".join(f"{v[i:i+1000].mean():.3f}" for i in range(0, len(v), 1000)))
print(f"ctl block SD = {np.std([v[i:i+1000].mean() for i in range(0,len(v),1000)], ddof=1):.4f}\n")
for name, ref in refs.items():
    common = sorted(set(ctl) & set(ref))
    d = np.array([ctl[r] - ref[r] for r in common])
    se = d.std(ddof=1) / np.sqrt(len(d))
    rng = np.random.default_rng(7)
    bs = d[rng.integers(0, len(d), size=(20000, len(d)))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"{name:26s} ref mean={np.mean([ref[r] for r in common]):.4f}  "
          f"paired ctl-ref = {d.mean():+.4f}  t={d.mean()/se:+.2f}  "
          f"boot95 [{lo:+.4f}, {hi:+.4f}]  {'EXCLUDES 0' if lo*hi>0 else 'contains 0'}")
