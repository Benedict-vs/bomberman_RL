"""What does (x+y)%4 carry on real danger steps, and did the PLB table use it?

Reads the probe pickle: rec = (round, row, digits8, x, y, greedy, n_others).
"""
import pickle, sys, math
from collections import Counter, defaultdict
import numpy as np

SIZES = (4, 4, 4, 4, 5, 5, 2, 5)


def load(p):
    with open(p, "rb") as fh:
        return pickle.load(fh)


d = load(sys.argv[1] if len(sys.argv) > 1 else "scratchpad/audit8/plb103.pkl")
rec = d["rec"]
print(f"steps {len(rec)}  rounds {len(d['rounds'])}")

dang = [r for r in rec if r[2][4] > 0]
safe = [r for r in rec if r[2][4] == 0]
print(f"danger steps {len(dang)} ({len(dang)/len(rec):.3f})  safe {len(safe)}")

# 1. lattice check: does (x+y)%4 track the pillar lattice?
cls = Counter()
for r in dang:
    x, y = r[3], r[4]
    par = "crossing(odd,odd)" if (x % 2 and y % 2) else (
        "corr-horiz(even,odd)" if y % 2 else "corr-vert(odd,even)")
    cls[((x + y) % 4, par)] += 1
print("\n(x+y)%4  x  lattice class, danger steps")
for k in sorted(cls):
    print(f"  d8={k[0]}  {k[1]:<22}{cls[k]:6d}")

marg = Counter(r[2][7] for r in dang)
tot = sum(marg.values())
print("\nmarginal of d8 on danger steps:", {k: round(v/tot, 4) for k, v in sorted(marg.items())})

# 2. is the parity bit already implied by digits 1-4 (neighbour codes)?
#    NB_BLOCKED = 0. On a corridor tile two OPPOSITE neighbours are permanent walls.
nb2par = defaultdict(Counter)
for r in dang:
    nb = r[2][:4]
    par = (r[3] + r[4]) % 2          # 0 = crossing, 1 = corridor
    nb2par[nb][par] += 1
H, n = 0.0, 0
maj = 0
for nb, c in nb2par.items():
    m = sum(c.values()); n += m
    maj += max(c.values())
    for v in c.values():
        p = v / m
        H -= m * p * math.log2(p)
H /= n
p1 = sum(c[1] for c in nb2par.values()) / n
H0 = -(p1*math.log2(p1) + (1-p1)*math.log2(1-p1))
print(f"\nparity bit (x+y)%2 on danger steps: base H = {H0:.4f} bits, "
      f"H(parity | digits1-4) = {H:.4f} bits, best-guess accuracy {maj/n:.4f} "
      f"(majority-class {max(p1,1-p1):.4f})")

# same conditioning on all 7 other digits
o2par = defaultdict(Counter)
for r in dang:
    o2par[r[2][:7]][(r[3]+r[4]) % 2] += 1
H7, maj7 = 0.0, 0
for c in o2par.values():
    m = sum(c.values()); maj7 += max(c.values())
    for v in c.values():
        p = v/m; H7 -= m*p*math.log2(p)
H7 /= n
print(f"H(parity | digits1-7) = {H7:.4f} bits, accuracy {maj7/n:.4f}, "
      f"{len(o2par)} distinct 7-digit families")

# 3. did the table's greedy action actually depend on d8?  Compare greedy(row)
#    with greedy(row with d8 forced to 0) and with the parity-only surrogate.
q = np.load(sys.argv[2] if len(sys.argv) > 2 else
            "checkpoints/benedict_task4/q_table_e36_PLB_s103__ep20000.npy")
mult = np.array([16000, 4000, 1000, 250, 50, 10, 5, 1])


def row_of(dg):
    return int(np.dot(dg, mult))


diff = same_par = cross_par = 0
diff_par = 0
for r in dang:
    dg = list(r[2])
    a_real = int(np.argmax(q[r[1]]))
    dg0 = dg.copy(); dg0[7] = 0
    a0 = int(np.argmax(q[row_of(dg0)]))
    if a_real != a0:
        diff += 1
    # parity-collapsed surrogate: map d8 -> the other member of its parity class
    dgp = dg.copy(); dgp[7] = {0: 2, 2: 0, 1: 3, 3: 1}.get(dg[7], dg[7])
    ap = int(np.argmax(q[row_of(dgp)]))
    if a_real != ap:
        same_par += 1
    # cross-parity surrogate: flip parity, keep "position" slot
    dgc = dg.copy(); dgc[7] = {0: 1, 1: 0, 2: 3, 3: 2}.get(dg[7], dg[7])
    ac = int(np.argmax(q[row_of(dgc)]))
    if a_real != ac:
        cross_par += 1
print(f"\ndanger steps where the greedy action changes if d8 is ...")
print(f"  set to 0 (the control's value)        {diff:6d} / {len(dang)} = {diff/len(dang):.4f}")
print(f"  swapped WITHIN its parity class       {same_par:6d} / {len(dang)} = {same_par/len(dang):.4f}")
print(f"  swapped ACROSS the parity class       {cross_par:6d} / {len(dang)} = {cross_par/len(dang):.4f}")
