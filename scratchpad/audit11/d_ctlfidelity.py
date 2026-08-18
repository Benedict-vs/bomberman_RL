"""Audit 11 -- does the E40 control really reproduce the shipped agent?

E40 validates its control by comparing 8000 rounds (4.009) against a *published*
1000-round number (3.949) and calling +0.060 'inside the +/-0.12 noise floor'.
That is an unpaired comparison of different arena sets. Rounds 0-999 of the E40
control ARE the published run's arenas, so a paired test exists and is stricter.
"""
import csv, sys
from collections import defaultdict
import numpy as np

def load(path, code):
    by = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            if row.get("code", row.get("agent")) == code or row.get("agent","").startswith(code):
                pass
            by.setdefault(int(row["round"]), []).append(row)
    return by

SHIP = "results/eval/task4_tournament/benedict_task4_shipped__task4_rb_ship990731.csv"
CTL  = "scratchpad/benedict/e40/e40_ctl__task4_rb_ship990731.csv"

def rows(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))

sh = rows(SHIP); ct = rows(CTL)
print("shipped csv columns:", list(sh[0]))
print("ctl csv columns    :", list(ct[0]))

def mine(rs, is_ship):
    out = {}
    for r in rs:
        code = r.get("code") or r.get("agent")
        if is_ship:
            hit = "benedict_task4" in code
        else:
            hit = code == "user_agent"
        if hit:
            out[int(r["round"])] = r
    return out

def allrows(rs):
    by = defaultdict(list)
    for r in rs:
        by[int(r["round"])].append(r)
    return by

ms, mc = mine(sh, True), mine(ct, False)
bs, bc = allrows(sh), allrows(ct)
common = sorted(set(ms) & set(mc))
print(f"\npaired rounds: {len(common)}  (ship n={len(ms)}, ctl n={len(mc)})")

def margins(by, me_row):
    o = [float(x["score"]) for x in by if x is not me_row]
    return float(me_row["score"]) - np.mean(o), float(me_row["score"]) - max(o)

for key in ("score", "kills", "suicides", "crates", "coins", "steps", "invalid", "moves", "bombs"):
    a = np.array([float(ms[r][key]) for r in common])
    b = np.array([float(mc[r][key]) for r in common])
    d = b - a
    se = d.std(ddof=1)/np.sqrt(len(d))
    star = " *" if abs(d.mean()) > 1.96*se else ""
    print(f"  {key:<10} ship {a.mean():8.3f}   e40ctl {b.mean():8.3f}   "
          f"paired {d.mean():+.4f} [{d.mean()-1.96*se:+.4f}, {d.mean()+1.96*se:+.4f}] t={d.mean()/se:+.2f}{star}")

# identical-round rate: how often the two runs produced the exact same score
same = sum(1 for r in common if float(ms[r]["score"]) == float(mc[r]["score"]))
print(f"\n  rounds with identical own score: {same}/{len(common)} = {same/len(common):.1%}")

# margins, paired
mm_s, mb_s, mm_c, mb_c = [], [], [], []
for r in common:
    a = margins(bs[r], ms[r]); b = margins(bc[r], mc[r])
    mm_s.append(a[0]); mb_s.append(a[1]); mm_c.append(b[0]); mb_c.append(b[1])
for nm, A, B in (("margin_mean", mm_s, mm_c), ("margin_best", mb_s, mb_c)):
    d = np.array(B) - np.array(A); se = d.std(ddof=1)/np.sqrt(len(d))
    print(f"  {nm:<12} ship {np.mean(A):+.3f}  e40ctl {np.mean(B):+.3f}  "
          f"paired {d.mean():+.4f} [{d.mean()-1.96*se:+.4f}, {d.mean()+1.96*se:+.4f}] t={d.mean()/se:+.2f}")

# does the E40 control's own mean drift across the 8000 rounds?
allc = np.array([float(mc_[ "score"]) for mc_ in mine(ct, False).values()])
sc = np.array([float(r["score"]) for r in ct if (r.get("code") == "user_agent")])
print(f"\n  E40 ctl score by 1000-round block:")
for i in range(0, 8000, 1000):
    blk = sc[i:i+1000]
    print(f"    rounds {i:5d}-{i+999:5d}: {blk.mean():.3f}  (se {blk.std(ddof=1)/np.sqrt(len(blk)):.3f})")
