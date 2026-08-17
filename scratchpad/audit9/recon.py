"""Audit 9 reconciliation: why the pooled 'even-odd' contrast disagrees with the
within-base matched one.

Row set (both methods): the 1131 danger BASES whose parent q_table_e36parent.npy row
carried value, i.e. all five d8 siblings are non-zero.

Pooled estimator (coordinator):  mean(maxQ | d8 even, reachable) - mean(maxQ | d8 odd, reachable)
Matched estimator (audit 9):     mean over bases that have an UPDATED row in BOTH classes of
                                 [mean maxQ(updated even) - mean maxQ(updated odd)]

Because every sibling of a base starts at the SAME parent value (the parent is a
factor-1 broadcast), a row that training never touched still holds that value, so
    mean(maxQ|even) - mean(maxQ|odd) = f_even * ubar_even - f_odd * ubar_odd
with f = share of rows updated in training and ubar = mean uplift (learned - parent)
among updated rows.  Any imbalance in f is therefore reported as a lattice effect.
"""
import numpy as np
from collections import defaultdict
from scipy import stats

CK = "checkpoints/benedict_task4"
par = np.load(f"{CK}/q_table_e36parent.npy")
idx = np.arange(64000); d8 = idx % 5; d5 = (idx // 50) % 5; base = idx // 5
danger = d5 > 0
REACH = {"ctl2": [0], "PLB2": [0, 1, 2, 3], "PAR": [0, 1], "SHF": [0, 1, 2, 3]}
SEEDS = list(range(100, 115))
parmax = par.max(1)

def rows(arm, s):
    q = np.load(f"{CK}/q_table_e37_{arm}_s{s}__ep20000.npy")
    allnz = q.any(1).reshape(12800, 5).all(1)
    sel = danger & np.repeat(allnz, 5) & np.isin(d8, REACH[arm])
    upd = np.abs(q - par).sum(1) > 1e-12
    return q, sel, upd

print("=== 1. update coverage is confounded with lattice class ===")
print(f"{'arm':>5} {'f(even)':>9} {'f(odd)':>9} {'f_e-f_o':>9} {'u(even)':>9} {'u(odd)':>9} {'u_e-u_o':>9}")
store = {}
for arm in ("PLB2", "PAR", "SHF"):
    F, U, POOL, UPL = [], [], [], []
    for s in SEEDS:
        q, sel, upd = rows(arm, s)
        mq = q.max(1)
        e, o = sel & (d8 % 2 == 0), sel & (d8 % 2 == 1)
        fe, fo = upd[e].mean(), upd[o].mean()
        ue = (mq[e & upd] - parmax[e & upd]).mean()
        uo = (mq[o & upd] - parmax[o & upd]).mean()
        F.append((fe, fo)); U.append((ue, uo))
        POOL.append(mq[e].mean() - mq[o].mean())
        UPL.append(ue - uo)
    F, U = np.array(F), np.array(U)
    store[arm] = (np.array(POOL), np.array(UPL), F, U)
    print(f"{arm:>5} {F[:,0].mean():9.3f} {F[:,1].mean():9.3f} {(F[:,0]-F[:,1]).mean():+9.3f}"
          f" {U[:,0].mean():9.4f} {U[:,1].mean():9.4f} {(U[:,0]-U[:,1]).mean():+9.4f}")

print("\n=== 2. the pooled contrast is reproduced by the coverage gap alone ===")
print(f"{'arm':>5} {'pooled (reported)':>19} {'predicted f_e*u_e - f_o*u_o':>29} {'coverage-free u_e - u_o':>25}")
for arm in ("PLB2", "PAR", "SHF"):
    POOL, UPL, F, U = store[arm]
    pred = (F[:, 0] * U[:, 0] - F[:, 1] * U[:, 1])
    ci = 1.96 * UPL.std(ddof=1) / np.sqrt(15)
    print(f"{arm:>5} {POOL.mean():+19.4f} {pred.mean():+29.4f} "
          f"{UPL.mean():+15.4f} +-{ci:.4f}  ({int((UPL<0).sum())}/15 negative)")

print("\n=== 3. the control falsifies the pooled estimator ===")
for s in (100,):
    q = np.load(f"{CK}/q_table_e37_ctl2_s{s}__ep20000.npy")
    allnz = q.any(1).reshape(12800, 5).all(1)
    sel = danger & np.repeat(allnz, 5)
    mq = q.max(1)
    e, o = sel & (d8 % 2 == 0), sel & (d8 % 2 == 1)
    print(f" ctl2 s{s}: d8 is PINNED to 0, so the table holds no lattice information at all.")
    print(f"   pooled even-odd over all five siblings = {mq[e].mean()-mq[o].mean():+.4f}")
    print(f"   restricted to reachable d8 = [0]: the odd group is EMPTY -> estimator UNDEFINED")
    print("   -> the pooled estimator cannot be computed on the control, so it has no null calibration.")

print("\n=== 4. matched estimator: bases with an UPDATED row in both classes ===")
matched = {}
for arm in ("PLB2", "PAR", "SHF"):
    D, N = [], []
    for s in SEEDS:
        q, sel, upd = rows(arm, s)
        mq = q.max(1); t = sel & upd
        ev, od = defaultdict(list), defaultdict(list)
        for i in np.flatnonzero(t):
            (ev if d8[i] % 2 == 0 else od)[base[i]].append(mq[i])
        c = [b for b in ev if b in od]
        D.append(np.mean([np.mean(ev[b]) - np.mean(od[b]) for b in c])); N.append(len(c))
    D = np.array(D); matched[arm] = D
    print(f" {arm:>5}: {D.mean():+.4f} +- {1.96*D.std(ddof=1)/np.sqrt(15):.4f}"
          f"  ({int((D<0).sum())}/15 negative, mean {np.mean(N):.0f} bases)")

print("\n=== 5. same bases for PLB2 and PAR (intersection), so the ordering is comparable ===")
dp, da = [], []
for s in SEEDS:
    got = {}
    for arm in ("PLB2", "PAR"):
        q, sel, upd = rows(arm, s)
        mq = q.max(1); t = sel & upd
        ev, od = defaultdict(list), defaultdict(list)
        for i in np.flatnonzero(t):
            (ev if d8[i] % 2 == 0 else od)[base[i]].append(mq[i])
        got[arm] = (ev, od, {b for b in ev if b in od})
    common = got["PLB2"][2] & got["PAR"][2]
    dp.append(np.mean([np.mean(got["PLB2"][0][b]) - np.mean(got["PLB2"][1][b]) for b in common]))
    da.append(np.mean([np.mean(got["PAR"][0][b]) - np.mean(got["PAR"][1][b]) for b in common]))
dp, da = np.array(dp), np.array(da)
print(f"  common bases per seed: {len(common)}")
print(f"  PLB2 {dp.mean():+.4f} ({int((dp<0).sum())}/15 neg)   PAR {da.mean():+.4f} ({int((da<0).sum())}/15 neg)"
      f"   PLB2-PAR {np.mean(dp-da):+.4f} p={stats.ttest_rel(dp,da).pvalue:.4f}")

print("\n=== 6. across- vs within-lattice-class split in PLB2, matched, updated rows only ===")
acr, wit = [], []
for s in SEEDS:
    q, sel, upd = rows("PLB2", s)
    mq = q.max(1); t = sel & upd
    g = defaultdict(dict)
    for i in np.flatnonzero(t):
        g[base[i]][d8[i]] = mq[i]
    a, w = [], []
    for b, v in g.items():
        if 0 in v and 2 in v: w.append(abs(v[0] - v[2]))
        if 1 in v and 3 in v: w.append(abs(v[1] - v[3]))
        e = [v[k] for k in (0, 2) if k in v]; o = [v[k] for k in (1, 3) if k in v]
        if e and o: a.append(abs(np.mean(e) - np.mean(o)))
    acr.append(np.mean(a)); wit.append(np.mean(w))
acr, wit = np.array(acr), np.array(wit)
print(f"  |across| {acr.mean():.4f}  |within| {wit.mean():.4f}  ratio {acr.mean()/wit.mean():.2f}"
      f"  ({int((acr>wit).sum())}/15, p={stats.ttest_rel(acr,wit).pvalue:.2e})")

print("\n=== 7. own_danger stratification, matched estimator ===")
for arm in ("PLB2", "PAR"):
    print(f" {arm}:")
    for dv in (1, 2, 3, 4):
        D = []
        for s in SEEDS:
            q, sel, upd = rows(arm, s)
            mq = q.max(1); t = sel & upd & (d5 == dv)
            ev, od = defaultdict(list), defaultdict(list)
            for i in np.flatnonzero(t):
                (ev if d8[i] % 2 == 0 else od)[base[i]].append(mq[i])
            c = [b for b in ev if b in od]
            D.append(np.mean([np.mean(ev[b]) - np.mean(od[b]) for b in c]) if c else np.nan)
        D = np.array(D, float)
        print(f"   own_danger={dv}: {np.nanmean(D):+.4f}  ({int(np.nansum(D<0))}/{np.isfinite(D).sum()} negative)")
