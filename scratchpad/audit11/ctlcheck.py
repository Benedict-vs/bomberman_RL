"""Audit 11: is the E40 control arm the shipped agent, tested where the noise floor allows?"""
import csv, numpy as np
from collections import defaultdict
from pathlib import Path

def load(path, agent_key):
    br = defaultdict(dict)
    with open(path) as fh:
        for row in csv.DictReader(fh):
            code = row.get("code") or row.get("agent", "")
            br[int(row["round"])][code] = row
    return br

def ours_series(br, key, mine):
    out = {}
    for r, d in br.items():
        for code, row in d.items():
            if mine in code:
                out[r] = float(row[key]); break
    return out

ship = Path("results/eval/task4_tournament/benedict_task4_shipped__task4_rb_ship990731.csv")
ship2 = Path("results/eval/task4_tournament/benedict_task4_shipped_e37__task4_rb_ship990731.csv")
ctl  = Path("scratchpad/benedict/e40/e40_ctl__task4_rb_ship990731.csv")

with open(ship) as fh:
    hdr = fh.readline().strip()
print("shipped CSV header:", hdr)
with open(ship) as fh:
    codes = sorted({r.get("code", r.get("agent","?")) for r in csv.DictReader(fh)})
print("shipped codes:", codes)

for name, p in (("shipped", ship), ("shipped_e37", ship2)):
    br = load(p, None)
    n = len(br)
    s = ours_series(br, "score", "benedict_task4")
    print(f"{name}: n={n} rounds, our mean score={np.mean(list(s.values())):.4f}")

c = load(ctl, None)
cs = ours_series(c, "score", "user_agent")
print(f"e40 ctl: n={len(c)} rounds, mean score={np.mean(list(cs.values())):.4f}")

for name, p in (("shipped", ship), ("shipped_e37", ship2)):
    br = load(p, None)
    s = ours_series(br, "score", "benedict_task4")
    common = sorted(set(s) & set(cs))
    d = np.array([cs[r] - s[r] for r in common])
    se = d.std(ddof=1)/np.sqrt(len(d))
    ident = np.mean(d == 0)
    print(f"paired e40ctl - {name} on {len(common)} shared arenas: "
          f"{d.mean():+.4f} +/- {1.96*se:.4f}  (identical rounds {ident:.1%})")
