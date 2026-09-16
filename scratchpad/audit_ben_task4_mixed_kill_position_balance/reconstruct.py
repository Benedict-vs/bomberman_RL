from __future__ import annotations

import csv, hashlib, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
E = ROOT / "results/eval/task4_tournament"
SEEDS = np.arange(20260731, 20261731)
M = [
 ["ben_dqn_task4_mixed_kill_v1_2000ep_seed11__task4_rule_based_retry1_eval1000"],
 *[[f"ben_dqn_task4_mixed_kill_v1_2000ep_seed11__task4_rule_based_listslot{s}_eval1000"] for s in (1,2,3)]]
I = [
 ["ben_dqn_task3_seed13__task4_rule_based_eval1000", "ben_dqn_task3_seed13__task4_rule_based_retry1_eval1000"],
 *[[f"ben_dqn_task3_seed13__task4_rule_based_listslot{s}_eval1000"] for s in (1,2,3)]]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def load(groups, code, model):
    out={}; metas=[]
    for slot, stems in enumerate(groups):
        out[slot]={}
        for stem in stems:
            rows=list(csv.DictReader((E/f"{stem}.csv").open()))
            meta=json.loads((E/f"{stem}.meta.json").read_text()); metas.append((stem,meta))
            assert len(rows)==4000 and len({r['round'] for r in rows})==1000
            assert [int(r['seed']) for r in rows[::4]] == list(SEEDS)
            assert meta['agent_provenance'][code]['model_sha256']==sha(model)
            own=[r for r in rows if r['code']==code]
            assert len(own)==1000 and {int(r['slot']) for r in own}=={slot}
            for r in own:
                assert int(r['score'])==int(r['coins'])+5*int(r['kills'])
                assert int(r['died'])==int(r['suicides'])+int(r['killed_by_opponent'])
                assert int(r['survived'])==1-int(r['died'])
                out[slot].setdefault(int(r['seed']),[]).append(r)
    return out,metas

def vals(d,col, slot=None):
    slots=range(4) if slot is None else [slot]
    return np.array([np.mean([np.mean([float(r[col]) for r in d[s][seed]]) for s in slots]) for seed in SEEDS])

def ci(x, seed=12345, n=50000):
    rng=np.random.default_rng(seed); means=[]
    # chunks avoid a large allocation
    for _ in range(n//1000): means.extend(rng.choice(x,(1000,len(x)),replace=True).mean(1))
    return np.percentile(means,[2.5,97.5])

def flip(x, seed=0, n=100000):
    rng=np.random.default_rng(seed); obs=abs(x.mean()); hits=0
    for _ in range(n//1000):
        z=(rng.choice((-1.,1.),(1000,len(x)))*x).mean(1); hits += np.sum(abs(z)>=obs)
    return hits/n

def main():
    m,mm=load(M,'ben_task4',ROOT/'agent_code/ben_task4/ben_task4_mixed_kill_v1_2000ep_seed11.pt')
    i,im=load(I,'dqn_task3',ROOT/'agent_code/dqn_task3/dqn_task3_seed13.pt')
    for tag, metas, code in [('mixed',mm,'ben_task4'),('inc',im,'dqn_task3')]:
        print(tag)
        for stem,x in metas:
            p=x['agent_provenance'][code]
            print(stem, x['base_seed'],x['n_rounds'],x['scenario'],p.get('callbacks_sha256'),p.get('model_file'),p.get('model_sha256'), x.get('settings'))
    print('\nslot score means and differences')
    for s in range(4):
        x=vals(m,'score',s)-vals(i,'score',s)
        print(s,vals(m,'score',s).mean(),vals(i,'score',s).mean(),x.mean(),ci(x),flip(x))
    x=vals(m,'score')-vals(i,'score')
    print('balanced',vals(m,'score').mean(),vals(i,'score').mean(),x.mean(),ci(x),flip(x))
    print('bootstrap verdicts',[(q,ci(x,q,10000)) for q in range(12345,12365)])
    # alternatives: exactly one slot-0 replicate each; original instead of current retry.
    for mi in range(len(M[0])):
      for ii in range(len(I[0])):
        ma={s:{k:list(v) for k,v in m[s].items()} for s in m}; ia={s:{k:list(v) for k,v in i[s].items()} for s in i}
        for seed in SEEDS: ma[0][seed]=[m[0][seed][mi]]; ia[0][seed]=[i[0][seed][ii]]
        z=vals(ma,'score')-vals(ia,'score')
        print('single slot0 reps',mi,ii,z.mean(),ci(z),flip(z))
    # leave-one-slot-out and slot heterogeneity (correlations of seedwise differences)
    ds=np.vstack([vals(m,'score',s)-vals(i,'score',s) for s in range(4)])
    print('LOSO',[(s,np.delete(ds,s,0).mean(0).mean(),ci(np.delete(ds,s,0).mean(0))) for s in range(4)])
    print('slot difference correlations\n',np.corrcoef(ds))
    print('slot SD of mean differences',np.std(ds.mean(1),ddof=1))
    # Sequential moving-block bootstrap as a sensitivity check for dependence
    # from the opponents' persistent stdlib RNG stream.
    rng=np.random.default_rng(7)
    for block in (5,10,25,50):
        means=[]; starts=np.arange(len(x)-block+1)
        for _ in range(10000):
            chosen=rng.choice(starts,int(np.ceil(len(x)/block)),replace=True)
            sample=np.concatenate([x[j:j+block] for j in chosen])[:len(x)]
            means.append(sample.mean())
        print('block bootstrap',block,np.percentile(means,[2.5,97.5]))
    # Component/death reconstruction and all headline diagnostics.
    for col in ('score','won','kills','suicides','killed_by_opponent','survived','coins','bombs'):
        z=vals(m,col)-vals(i,col)
        print('metric',col,vals(m,col).mean(),vals(i,col).mean(),z.mean(),ci(z),flip(z))
    for tag,d in [('mixed',m),('inc',i)]:
        allrows=[r for s in d.values() for rs in s.values() for r in rs]
        print(tag,'timeouts',sum(int(r['think_over_limit']) for r in allrows),'max',max(float(r['think_max_ms']) for r in allrows),'mean max',np.mean([float(r['think_max_ms']) for r in allrows]))

if __name__=='__main__': main()
