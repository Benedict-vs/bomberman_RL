#!/usr/bin/env python3
"""E30 audit: correct symmetric bar, n=5 t-intervals, within-round contrasts."""
from __future__ import annotations
import csv, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy import stats as sps

ROOT = Path("/Users/benedictvonschubert/Projects/bomberman_RL")
EV = ROOT / "results/eval/task4_tournament"
AU = ROOT / "scratchpad/audit3/eval"

def load(p):
    with open(p) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("round","slot","survived","score","coins","kills","suicides","crates",
                  "bombs","invalid","steps","died","killed_by_opponent","rank","won","round_steps"):
            r[k] = int(float(r[k]))
        r["think_max_ms"] = float(r["think_max_ms"])
    return rows

def rounds(rows):
    d = defaultdict(list)
    for r in rows: d[r["round"]].append(r)
    return [d[k] for k in sorted(d)]

def perround(path, slot=0):
    """Per-round series: ours, opp-mean, symmetric-null share, sole-win flags."""
    grps = rounds(load(path))
    out = defaultdict(list)
    for g in grps:
        g = sorted(g, key=lambda r: r["slot"])
        me = g[slot]; op = [r for r in g if r["slot"] != slot]
        W = sum(r["won"] for r in g)
        best = max(r["score"] for r in g)
        sole = int(W == 1)
        out["won"].append(me["won"])
        out["null_won"].append(W / 4.0)               # exchangeable null share
        out["won_minus_null"].append(me["won"] - W / 4.0)
        out["sole"].append(int(me["won"] and W == 1))
        out["null_sole"].append(sole / 4.0)
        out["sole_minus_null"].append(int(me["won"] and W == 1) - sole / 4.0)
        out["score"].append(me["score"])
        out["opp_score"].append(np.mean([r["score"] for r in op]))
        out["score_adv"].append(me["score"] - np.mean([r["score"] for r in op]))
        out["rank"].append(me["rank"])
        out["null_rank"].append(np.mean([r["rank"] for r in g]))
        out["rank_adv"].append(np.mean([r["rank"] for r in g]) - me["rank"])  # higher=better
        out["coins"].append(me["coins"]); out["opp_coins"].append(sum(r["coins"] for r in op))
        out["kills"].append(me["kills"]); out["opp_kills"].append(sum(r["kills"] for r in op))
        out["crates"].append(me["crates"]); out["opp_crates"].append(sum(r["crates"] for r in op))
        out["tot_coins"].append(sum(r["coins"] for r in g))
        out["tot_kills"].append(sum(r["kills"] for r in g))
        out["tot_score"].append(sum(r["score"] for r in g))
        out["tot_crates"].append(sum(r["crates"] for r in g))
        out["round_steps"].append(me["round_steps"])
        out["surv"].append(me["survived"]); out["opp_surv"].append(np.mean([r["survived"] for r in op]))
        out["suic"].append(me["suicides"]); out["kbo"].append(me["killed_by_opponent"])
        out["steps"].append(me["steps"]); out["opp_steps"].append(np.mean([r["steps"] for r in op]))
        out["thinkmax"].append(me["think_max_ms"])
    return {k: np.array(v, dtype=float) for k, v in out.items()}

def tci(vals):
    v = np.asarray(vals, dtype=float); n = len(v)
    m = v.mean(); se = v.std(ddof=1)/np.sqrt(n)
    h = sps.t.ppf(0.975, n-1)*se
    return m, m-h, m+h, h

def line(name, vals, fmt="{:+.4f}"):
    m, lo, hi, h = tci(vals)
    star = "EXCL 0" if lo > 0 or hi < 0 else "incl 0"
    print(f"  {name:<34} {fmt.format(m)}  [{fmt.format(lo)}, {fmt.format(hi)}]  n={len(vals)}  {star}")

SEEDS = [60,61,62,63,64]

def arm(prefix, ep, base="__task4_rb_val550731", d=EV, pat=None):
    res = {}
    for s in SEEDS:
        p = d / (pat.format(s=s, ep=ep) if pat else f"benedict_q_e30_{prefix}_s{s}__ep{ep}{base}.csv")
        if not p.exists(): continue
        res[s] = perround(p)
    return res

def report(title, armdata):
    print(f"\n=== {title}  (n={len(armdata)} training runs) ===")
    per = lambda k: [armdata[s][k].mean() for s in sorted(armdata)]
    print("  per-seed won      :", " ".join(f"{x:.3f}" for x in per("won")))
    print("  per-seed null bar :", " ".join(f"{x:.3f}" for x in per("null_won")))
    print("  per-seed score    :", " ".join(f"{x:.3f}" for x in per("score")))
    print("  per-seed oppscore :", " ".join(f"{x:.3f}" for x in per("opp_score")))
    line("won (raw)", per("won"))
    line("null bar E[#winners]/4", per("null_won"))
    line("won - symmetric null  **", per("won_minus_null"))
    line("sole-win - sole null", per("sole_minus_null"))
    line("score - opp mean score", per("score_adv"))
    line("rank advantage (mean-ours)", per("rank_adv"))
    line("our coins", per("coins")); line("opp coins (sum of 3)", per("opp_coins"))
    line("total coins in round", per("tot_coins"))
    line("total kills in round", per("tot_kills"))
    line("total crates in round", per("tot_crates"))
    line("total score in round", per("tot_score"))
    line("round_steps", per("round_steps"))
    line("our survived", per("surv")); line("opp survived", per("opp_surv"))
    line("our suicides", per("suic")); line("our killed_by_opp", per("kbo"))
    print(f"  max think_max_ms  : {max(armdata[s]['thinkmax'].max() for s in armdata):.3f} ms")

if __name__ == "__main__":
    for ep in (5000, 10000, 20000):
        report(f"arm T @{ep} (val 550731)", arm("T", ep))
    report("arm TD @10000 (val 550731)", arm("TD", 10000))

    # single-run controls
    for lab in ("F", "S_folded"):
        d = perround(EV / f"benedict_q_e30_{lab}__task4_rb_val550731.csv")
        W = d["won"].mean(); nb = d["null_won"].mean()
        print(f"\n[control {lab}]  won={W:.4f}  symmetric-null={nb:.4f}  "
              f"won-null={d['won_minus_null'].mean():+.4f}  score={d['score'].mean():.3f} "
              f"oppscore={d['opp_score'].mean():.3f} totscore={d['tot_score'].mean():.3f} "
              f"totcoins={d['tot_coins'].mean():.3f} totkills={d['tot_kills'].mean():.3f} "
              f"totcrates={d['tot_crates'].mean():.3f} roundsteps={d['round_steps'].mean():.1f}")

    for name, p in (("REF rb self-play ship990731", EV/"ref_rule_based_agent__task4_rb_ship990731.csv"),
                    ("REF rb self-play val550731", AU/"ref_rb_selfplay__val550731.csv")):
        if not p.exists():
            print(f"\n[{name}] not available yet"); continue
        d = perround(p)
        print(f"\n[{name}] slot0 won={d['won'].mean():.4f} null={d['null_won'].mean():.4f} "
              f"score={d['score'].mean():.3f} totscore={d['tot_score'].mean():.3f} "
              f"totcoins={d['tot_coins'].mean():.3f} totkills={d['tot_kills'].mean():.3f} "
              f"totcrates={d['tot_crates'].mean():.3f} roundsteps={d['round_steps'].mean():.1f} "
              f"surv={d['surv'].mean():.3f}")

    # held-out ship seed, if present
    held = arm("T", 10000, d=AU, pat="audit_T_s{s}__ep{ep}__ship990731.csv")
    if held:
        report("arm T @10000 HELD-OUT ship 990731", held)

    h5 = arm("T", 5000, d=AU, pat="audit_T_s{s}__ep{ep}__ship990731.csv")
    if h5:
        report("arm T @5000 HELD-OUT ship 990731", h5)
