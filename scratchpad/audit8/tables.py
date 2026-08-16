"""What did the PLB / OPP arms actually learn in the digit-8 danger rows?

Row index = mixed radix over FEATURE_SIZES = (4,4,4,4,5,5,2,5), digits in order
(nb_up, nb_right, nb_down, nb_left, own_danger, target, bomb_useful, target_dist).
Danger rows are own_danger > 0 (digit index 4).
"""
import numpy as np
import itertools, sys

SIZES = (4, 4, 4, 4, 5, 5, 2, 5)
N = int(np.prod(SIZES))
CK = "checkpoints/benedict_task4"
SEEDS = [100, 101, 102, 103, 104]


def digits_of(rows):
    out = np.zeros((len(rows), 8), dtype=int)
    r = rows.copy()
    for i in range(7, -1, -1):
        out[:, i] = r % SIZES[i]
        r //= SIZES[i]
    return out


ALL = np.arange(N)
D = digits_of(ALL)
DANGER = D[:, 4] > 0            # own_danger > 0 -> escape branch, d8 is the arm digit
D8 = D[:, 7]


def load(arm, seed, ep=20000):
    return np.load(f"{CK}/q_table_{arm}_s{seed}__ep{ep}.npy")


def family_index():
    """map each danger row to (family_id, d8) where family = digits 0..6 fixed."""
    fam = np.zeros(N, dtype=np.int64) - 1
    base = D[:, :7]
    mult = np.array([1, 4, 16, 64, 256, 1280, 6400])
    fam = (base * mult).sum(axis=1)
    return fam


FAM = family_index()


def report(arm, ep=20000):
    print(f"\n### {arm} @ep{ep}")
    tot_valued = []
    for seed in SEEDS:
        q = load(arm, seed, ep)
        valued = np.any(q != 0.0, axis=1)
        # danger rows, per d8 value
        counts = [int(valued[DANGER & (D8 == v)].sum()) for v in range(5)]
        # families where at least two d8 values are valued and argmax differs
        fams = {}
        idx = np.where(DANGER & valued)[0]
        for i in idx:
            fams.setdefault(FAM[i], []).append(i)
        multi = [v for v in fams.values() if len(v) > 1]
        diff = 0
        diff_par = 0   # differs across parity classes {0,2} vs {1,3}
        diff_within = 0  # differs *within* a parity class (0 vs 2, or 1 vs 3)
        for v in multi:
            am = {D8[i]: int(np.argmax(q[i])) for i in v}
            if len(set(am.values())) > 1:
                diff += 1
            # within-parity disagreement
            for pair in ((0, 2), (1, 3)):
                if pair[0] in am and pair[1] in am and am[pair[0]] != am[pair[1]]:
                    diff_within += 1
                    break
            cross = False
            for a in (0, 2):
                for b in (1, 3):
                    if a in am and b in am and am[a] != am[b]:
                        cross = True
            if cross:
                diff_par += 1
        print(f"  s{seed}: valued danger rows by d8 = {counts}  "
              f"| multi-d8 families {len(multi)}  argmax differs {diff} "
              f"({diff/max(1,len(multi)):.2f}) | cross-parity {diff_par} within-parity {diff_within}")
        tot_valued.append(counts)
    return tot_valued


if __name__ == "__main__":
    for arm in ("benedict_q_e33_ctl", "benedict_q_e36_OPP", "benedict_q_e36_PLB"):
        try:
            report(arm.replace("benedict_q_", "").replace("e33_", "e33_").replace("e36_", "e36_"))
        except FileNotFoundError as e:
            print("missing", e)


def pairwise(arm, ep=20000):
    """Per-pair argmax-disagreement rate across d8 within a danger family.

    Parity hypothesis for PLB ((x+y)%4): pairs (0,2) and (1,3) are the SAME
    lattice class (both crossing / both corridor); the other four cross it.
    """
    print(f"\n--- pairwise argmax disagreement, {arm} @ep{ep} ---")
    pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    acc = {p: [0,0] for p in pairs}
    for seed in SEEDS:
        q = np.load(f"{CK}/q_table_{arm}_s{seed}__ep{ep}.npy")
        valued = np.any(q != 0.0, axis=1)
        idx = np.where(DANGER & valued)[0]
        fams = {}
        for i in idx:
            fams.setdefault(FAM[i], {})[D8[i]] = int(np.argmax(q[i]))
        for am in fams.values():
            for p in pairs:
                if p[0] in am and p[1] in am:
                    acc[p][1] += 1
                    acc[p][0] += int(am[p[0]] != am[p[1]])
    same = [(0,2),(1,3)]
    for p in pairs:
        tag = "SAME-parity" if p in same else "cross-parity"
        d, n = acc[p]
        print(f"  d8 {p[0]} vs {p[1]}  {d:4d}/{n:5d} = {d/n:.4f}   {tag}")
    ds = sum(acc[p][0] for p in same); ns = sum(acc[p][1] for p in same)
    dc = sum(acc[p][0] for p in pairs if p not in same); nc = sum(acc[p][1] for p in pairs if p not in same)
    print(f"  pooled SAME-parity {ds}/{ns} = {ds/ns:.4f}   cross-parity {dc}/{nc} = {dc/nc:.4f}")
