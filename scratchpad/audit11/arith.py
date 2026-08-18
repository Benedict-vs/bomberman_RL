"""Audit 11: score decomposition and bar reachability for E40."""
import csv, numpy as np
from collections import defaultdict
from pathlib import Path
D = Path("scratchpad/benedict/e40")

def load(tag):
    br = defaultdict(list)
    with open(D / f"e40_{tag}__task4_rb_ship990731.csv") as fh:
        for row in csv.DictReader(fh):
            br[int(row["round"])].append(row)
    return br

def ours(br, key):
    return np.array([float(next(r for r in rows if r["code"] == "user_agent")[key])
                     for _, rows in sorted(br.items())])

for tag in ("ctl", "k4stale", "k4sim", "k8sim"):
    br = load(tag)
    sc, co, ki = ours(br, "score"), ours(br, "coins"), ours(br, "kills")
    resid = sc - (co + 5*ki)
    print(f"{tag:8s} n={len(sc)} score={sc.mean():.4f} coins={co.mean():.4f} "
          f"kills={ki.mean():.4f} coins+5k={((co+5*ki).mean()):.4f} "
          f"max|resid|={np.abs(resid).max():.6f}")

print()
# Bar reachability, using the author's own pre-sweep conversion trace
for name, bombs_per_round in (("stale", 0.429), ("sim", 0.200)):
    for bar in (0.25,):
        need = bar / (5 * bombs_per_round)
        print(f"{name}: {bombs_per_round} override bombs/round -> to gain {bar} score "
              f"purely from kills needs credited-kill conversion >= {need:.1%}")
print("P3's own pre-registered conversion bar: 20%")
for name, b in (("stale", 0.429), ("sim", 0.200)):
    print(f"  {name} at exactly 20% conversion -> score gain = {5*b*0.20:+.3f}")
    print(f"  {name} at measured conversion    -> score gain = "
          f"{5*b*(0.084 if name=='stale' else 0.120):+.3f}")
