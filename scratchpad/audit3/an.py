#!/usr/bin/env python3
"""Independent recomputation of E30 numbers from the committed CSVs."""
from __future__ import annotations
import csv, sys, math
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path("/Users/benedictvonschubert/Projects/bomberman_RL")
EV = ROOT / "results/eval/task4_tournament"

NUM = {"round","seed","slot","survived","round_steps","score","coins","kills","suicides",
       "crates","bombs","moves","invalid","steps","time","think_mean_ms","think_max_ms",
       "think_over_limit","died","killed_by_opponent","rank","won"}

def load(p: Path):
    rows=[]
    with open(p) as fh:
        for r in csv.DictReader(fh):
            for k in list(r):
                if k in NUM:
                    r[k]=float(r[k]) if k in ("time","think_mean_ms","think_max_ms") else int(float(r[k]))
            rows.append(r)
    return rows

def by_round(rows):
    d=defaultdict(list)
    for r in rows: d[r["round"]].append(r)
    return d

def ours(rows, slot=0):
    return [r for r in rows if r["slot"]==slot]

def opps(rows, slot=0):
    return [r for r in rows if r["slot"]!=slot]

def mean(x): return float(np.mean(x))

def boot_ci(v, n=10000, seed=0):
    v=np.asarray(v,dtype=float); rng=np.random.default_rng(seed)
    idx=rng.integers(0,len(v),(n,len(v)))
    bs=v[idx].mean(axis=1)
    return v.mean(), float(np.percentile(bs,2.5)), float(np.percentile(bs,97.5))

def summarize(label, path, slot=0):
    rows=load(path)
    o=ours(rows,slot); p=opps(rows,slot)
    n=len(o)
    won=[r["won"] for r in o]
    m,lo,hi=boot_ci(won,seed=1)
    res=dict(label=label, n=n,
             won=m, won_lo=lo, won_hi=hi,
             score=mean([r["score"] for r in o]),
             coins=mean([r["coins"] for r in o]),
             kills=mean([r["kills"] for r in o]),
             suic=mean([r["suicides"] for r in o]),
             kbo=mean([r["killed_by_opponent"] for r in o]),
             surv=mean([r["survived"] for r in o]),
             crates=mean([r["crates"] for r in o]),
             bombs=mean([r["bombs"] for r in o]),
             steps=mean([r["steps"] for r in o]),
             rank=mean([r["rank"] for r in o]),
             opp_score=mean([r["score"] for r in p]),
             opp_won=mean([r["won"] for r in p]),
             opp_crates=mean([r["crates"] for r in p]),
             opp_kills=mean([r["kills"] for r in p]),
             opp_suic=mean([r["suicides"] for r in p]),
             opp_surv=mean([r["survived"] for r in p]),
             opp_steps=mean([r["steps"] for r in p]),
             thinkmax=max(r["think_max_ms"] for r in o),
             thinkmean=mean([r["think_mean_ms"] for r in o]),
             over=sum(r["think_over_limit"] for r in o),
             )
    # tie structure
    br=by_round(rows)
    nwin=[sum(r["won"] for r in g) for g in br.values()]
    res["mean_winners"]=mean(nwin)
    res["frac_multiwin"]=mean([x>1 for x in nwin])
    allzero=[all(r["score"]==0 for r in g) for g in br.values()]
    res["frac_allzero"]=mean(allzero)
    # our won conditional on all-zero
    res["won_from_allzero"]=mean([g[slot]["won"] and all(r["score"]==0 for r in g) for g in br.values()])
    res["won_shared"]=mean([g[slot]["won"] and sum(r["won"] for r in g)>1 for g in br.values()])
    res["won_sole"]=mean([g[slot]["won"] and sum(r["won"] for r in g)==1 for g in br.values()])
    return res, rows

FIELDS=["label","n","won","won_lo","won_hi","won_sole","won_shared","won_from_allzero",
        "mean_winners","frac_allzero","score","opp_score","opp_won","coins","kills","suic","kbo","surv",
        "crates","opp_crates","bombs","steps","opp_steps","opp_surv","rank","thinkmax","over"]

def main():
    labels=[]
    for f in sorted(EV.glob("benedict_q_e30_*__task4_rb_val550731.csv")):
        labels.append((f.stem.replace("benedict_q_e30_","").replace("__task4_rb_val550731",""), f))
    out=[]
    for lab,f in labels:
        r,_=summarize(lab,f)
        out.append(r)
    # reference (all rb) - slot 0 is just an rb
    rf=EV/"ref_rule_based_agent__task4_rb_ship990731.csv"
    r,_=summarize("REF_rb_selfplay_ship990731", rf)
    out.append(r)
    hdr=" ".join(f"{h:>12}" for h in FIELDS)
    print(hdr)
    for r in out:
        print(" ".join((f"{r[h]:>12}" if isinstance(r[h],str) else f"{r[h]:>12.4f}") for h in FIELDS))

if __name__=="__main__":
    main()
