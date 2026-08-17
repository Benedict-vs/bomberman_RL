"""Q-table forensics for E37 (read-only). Produces lattice.out / mech.out / tables.out.

FEATURE_SIZES = (4,4,4,4,5,5,2,5) -> row = base*5 + d8, own_danger = (row//50) % 5.
A row counts as *visited* iff it differs from the shared warm parent q_table_e36parent.npy.
For PLB2/PAR an even d8 means x+y even <=> both coords odd <=> a CROSSING (4 free
neighbours, own bomb covers 12 tiles); odd d8 is a CORRIDOR (2 exits, 6 tiles).
SHF is balanced within each lattice class, so its labels carry none of that.
"""
import numpy as np
from collections import defaultdict
from scipy import stats

CK = "/Users/benedictvonschubert/Projects/bomberman_RL/checkpoints/benedict_task4"
par = np.load(f"{CK}/q_table_e36parent.npy")
idx = np.arange(64000); d8 = idx % 5; d5 = (idx // 50) % 5; base = idx // 5
danger = d5 > 0
SEEDS = range(100, 115)

def visited(q):
    return np.abs(q - par).sum(1) > 1e-12

print("=== visited rows and argmax disagreement across digit-8 siblings (ep20000) ===")
for arm in ("ctl2", "PLB2", "PAR", "SHF"):
    tt, dt, nb, sp = [], [], [], []
    for s in SEEDS:
        q = np.load(f"{CK}/q_table_e37_{arm}_s{s}__ep20000.npy")
        t = visited(q); dtm = t & danger
        tt.append(t.sum()); dt.append(dtm.sum())
        am = q.argmax(1); g = defaultdict(list)
        for i in np.flatnonzero(dtm):
            g[base[i]].append(am[i])
        multi = [v for v in g.values() if len(v) >= 2]
        nb.append(len(multi))
        sp.append(sum(1 for v in multi if len(set(v)) > 1) / len(multi) if multi else np.nan)
    print(f" {arm:>5}: visited {np.mean(tt):7.0f}  visited-danger {np.mean(dt):7.0f}"
          f"  bases with >=2 visited siblings {np.mean(nb):6.1f}"
          f"  argmax-differs {np.nanmean(sp):.3f}")

print("\n=== max_a Q: crossing (even d8) minus corridor (odd d8), same base, ep20000 ===")
for arm in ("PLB2", "PAR", "SHF"):
    diffs = []
    for s in SEEDS:
        q = np.load(f"{CK}/q_table_e37_{arm}_s{s}__ep20000.npy")
        t = visited(q) & danger; mx = q.max(1)
        ev, od = defaultdict(list), defaultdict(list)
        for i in np.flatnonzero(t):
            (ev if d8[i] % 2 == 0 else od)[base[i]].append(mx[i])
        c = [b for b in ev if b in od]
        diffs.append(np.mean([np.mean(ev[b]) - np.mean(od[b]) for b in c]))
    d = np.array(diffs)
    print(f" {arm:>5}: {d.mean():+.4f} +- {1.96*d.std(ddof=1)/np.sqrt(15):.4f}"
          f"   ({int((d < 0).sum())}/15 negative)")

print("\n=== PLB2: |split ACROSS lattice class| vs |split WITHIN it| (0 vs 2, 1 vs 3) ===")
acr, wit = [], []
for s in SEEDS:
    q = np.load(f"{CK}/q_table_e37_PLB2_s{s}__ep20000.npy")
    t = visited(q) & danger; mx = q.max(1)
    g = defaultdict(dict)
    for i in np.flatnonzero(t):
        g[base[i]][d8[i]] = mx[i]
    a, w = [], []
    for b, v in g.items():
        if 0 in v and 2 in v: w.append(abs(v[0] - v[2]))
        if 1 in v and 3 in v: w.append(abs(v[1] - v[3]))
        e = [v[k] for k in (0, 2) if k in v]; o = [v[k] for k in (1, 3) if k in v]
        if e and o: a.append(abs(np.mean(e) - np.mean(o)))
    acr.append(np.mean(a)); wit.append(np.mean(w))
acr, wit = np.array(acr), np.array(wit)
print(f" across {acr.mean():.4f}  within {wit.mean():.4f}  ratio {acr.mean()/wit.mean():.2f}"
      f"  ({int((acr > wit).sum())}/15, p={stats.ttest_rel(acr, wit).pvalue:.2e})")

print("\n=== the same split stratified by own_danger (digit 5) ===")
for arm in ("PLB2", "PAR"):
    print(f" {arm}:")
    for dv in (1, 2, 3, 4):
        ds = []
        for s in SEEDS:
            q = np.load(f"{CK}/q_table_e37_{arm}_s{s}__ep20000.npy")
            t = visited(q) & (d5 == dv); mx = q.max(1)
            ev, od = defaultdict(list), defaultdict(list)
            for i in np.flatnonzero(t):
                (ev if d8[i] % 2 == 0 else od)[base[i]].append(mx[i])
            c = [b for b in ev if b in od]
            ds.append(np.mean([np.mean(ev[b]) - np.mean(od[b]) for b in c]) if c else np.nan)
        ds = np.array(ds, float)
        print(f"   own_danger={dv}: {np.nanmean(ds):+.4f}  ({int(np.nansum(ds < 0))}/{np.isfinite(ds).sum()} negative)")

print("\n=== BOMB attractiveness in NON-danger visited rows (semantics identical in all arms) ===")
for arm in ("ctl2", "PLB2", "PAR", "SHF"):
    fr, gap = [], []
    for s in SEEDS:
        q = np.load(f"{CK}/q_table_e37_{arm}_s{s}__ep20000.npy")
        qq = q[visited(q) & (~danger)]
        fr.append((qq.argmax(1) == 5).mean())
        gap.append((qq[:, 5] - np.delete(qq, 5, axis=1).max(1)).mean())
    print(f" {arm:>5}: share argmax==BOMB {np.mean(fr):.4f}   Q(BOMB)-max(other) {np.mean(gap):+.4f}")
