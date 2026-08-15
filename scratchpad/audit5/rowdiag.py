import pickle, sys, os, importlib.util, numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.abspath("."))
spec=importlib.util.spec_from_file_location("cb","agent_code/benedict_task4/callbacks.py")
cb=importlib.util.module_from_spec(spec); spec.loader.exec_module(cb)
A=cb.ACTIONS
D=pickle.load(open(sys.argv[1],"rb"))
q=np.load(sys.argv[2])
deaths=D["deaths"]; steps=D["step_log"]

# --- visit counts of (row, action) over all alive steps
pair=Counter(); rowc=Counter()
for st in steps:
    pair[(st["row"], st["action"])]+=1; rowc[st["row"]]+=1
print(f"alive steps {len(steps)}, distinct rows {len(rowc)}, distinct (row,action) {len(pair)}")

# --- fatal (row, action)
fatal=[]
for d in deaths:
    w={s["step"]:s for s in d["window"]}
    ds=d["death_step"]
    if ds not in w: continue
    s=w[ds]
    fatal.append((s["row"], s["my_action"], s["digits"], d["suicide"]))
print("deaths located:", len(fatal))

fc=Counter((r,a) for r,a,_,_ in fatal)
print("\ndistinct fatal (row,action) pairs:", len(fc))
print("top 15 fatal pairs: pair-visits, deaths, P(death|pair), digits, Q row")
for (r,a),n in fc.most_common(15):
    dg=next(d for rr,aa,d,_ in fatal if rr==r and aa==a)
    v=pair[(r,a)]
    qr=q[r]
    g=A[int(np.argmax(qr))]
    esc=dg[5]
    escact=A[esc-1] if esc else "-"
    print(f"  row {r:6d} act {a:5s} deaths {n:3d} visits {v:5d} P {n/max(v,1):.3f}  dig {dg}  greedy {g:5s} escdir {escact:5s}  Q {np.round(qr,2)}")

# --- aggregate: does the table point the right way at the death step?
print("\n=== at the death step, own-bomb deaths with a valid escape digit ===")
tot=has=agree=0
margins=[]
pdeath=[]
for r,a,dg,suic in fatal:
    if not suic: continue
    tot+=1
    esc=dg[5]
    if esc==0: continue
    has+=1
    ea=esc-1                    # index of escape action in ACTIONS
    qr=q[r]; g=int(np.argmax(qr))
    if g==ea: agree+=1
    ai=A.index(a) if a in A else None
    if ai is not None:
        margins.append(qr[ai]-qr[ea])
        pdeath.append(fc[(r,a)]/max(pair[(r,a)],1))
margins=np.array(margins); pdeath=np.array(pdeath)
print(f"suicide deaths {tot}; escape digit != 0 in {has} ({has/tot:.3f})")
print(f"  of those, table's greedy action == escape direction: {agree} ({agree/max(has,1):.3f})")
print(f"  Q(taken) - Q(escape) : mean {margins.mean():.3f} median {np.median(margins):.3f} "
      f"p10 {np.percentile(margins,10):.3f} p90 {np.percentile(margins,90):.3f}")
print(f"  P(death | row,action) for the fatal pair: mean {pdeath.mean():.3f} median {np.median(pdeath):.3f}")
# how much penalty is needed to flip each: margin < P * delta  -> delta > margin/P
need = margins/np.maximum(pdeath,1e-9)
need=need[margins>0]
print(f"  extra penalty needed to flip (margin/P), among the {len(need)} where taken is preferred:")
print("   percentiles 10/25/50/75/90:", np.round(np.percentile(need,[10,25,50,75,90]),2))
print(f"   flipped by delta=10: {(need<=10).mean():.3f}   by delta=25: {(need<=25).mean():.3f}")
