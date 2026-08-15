import pickle, sys, os, importlib.util, numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, os.path.abspath("."))
spec=importlib.util.spec_from_file_location("cb","agent_code/benedict_task4/callbacks.py")
cb=importlib.util.module_from_spec(spec); spec.loader.exec_module(cb)
A=cb.ACTIONS
D=pickle.load(open("scratchpad/audit5/deaths_e31_s80_ep20000.pkl","rb"))
q=np.load("checkpoints/benedict_task4/q_table_e31_S0_s80__ep20000.npy")
steps=D["step_log"]; deaths=D["deaths"]
N=len(steps)
print(f"alive steps {N} over {D['n_rounds']} rounds")

# --- 1. useless bombs
bombs=[s for s in steps if s["action"]=="BOMB"]
useless=[s for s in bombs if s["digits"][6]==0]
print(f"\nBOMB actions: {len(bombs)} ({len(bombs)/D['n_rounds']:.1f}/round); "
      f"with bomb_useful digit == 0: {len(useless)} ({len(useless)/max(len(bombs),1):.3f})")
inv=Counter(s["digits"][4] for s in useless)
print("  own_danger digit of the useless bombs:", dict(sorted(inv.items())))
# rows where BOMB is greedy but digit7==0
rows_b0=[r for r in range(len(q)) if np.abs(q[r]).sum()>0 and int(np.argmax(q[r]))==5]
def dg(i):
    d=[];rr=i
    for s in reversed(cb.FEATURE_SIZES): d.append(rr%s); rr//=s
    return tuple(reversed(d))
print(f"  rows whose greedy action is BOMB: {len(rows_b0)}, of which bomb_useful==0: "
      f"{sum(1 for r in rows_b0 if dg(r)[6]==0)}")

# --- 2. stepping into a blast from safety
into=0; safe_steps=0
mv={"UP":0,"RIGHT":1,"DOWN":2,"LEFT":3}
for s in steps:
    d=s["digits"]
    if d[4]!=0: continue
    safe_steps+=1
    a=s["action"]
    if a in mv and d[mv[a]] in (1,2): into+=1
print(f"\nsteps on a SAFE tile: {safe_steps} ({safe_steps/N:.3f}); of those, moved onto a LETHAL or IN-BLAST neighbour: {into} ({into/safe_steps:.3f})")

# --- 3. ignoring the escape digit
ign=0; dang=0; noesc=0
for s in steps:
    d=s["digits"]
    if d[4]==0: continue
    dang+=1
    if d[5]==0: noesc+=1; continue
    if s["action"]!=A[d[5]-1]: ign+=1
print(f"steps in DANGER: {dang} ({dang/N:.3f}); escape digit says NO_TARGET: {noesc} ({noesc/dang:.3f}); "
      f"escape digit valid but action != escape dir: {ign} ({ign/max(dang-noesc,1):.3f} of those)")

# --- 4. period-2 cycles: consecutive positions a,b,a
seq=defaultdict(list)
for s in steps: seq[s["round"]].append((s["step"],s["row"],s["action"]))
p2=0; tot=0
for r,v in seq.items():
    v.sort()
    for i in range(len(v)-2):
        tot+=1
        if v[i][1]==v[i+2][1] and v[i][1]!=v[i+1][1]: p2+=1
print(f"\nrow triples s_t == s_t+2 != s_t+1 (period-2 in feature space): {p2}/{tot} = {p2/tot:.3f}")

# --- 5. the fatal 6-step signature
sig=Counter()
for d in deaths:
    w={s["step"]:s for s in d["window"]}; ds=d["death_step"]
    key=tuple(w[st]["row"] if st in w else None for st in range(ds-2,ds+1))
    sig[key]+=1
print("\nmost common (row_{d-2}, row_{d-1}, row_d) signatures:")
for k,v in sig.most_common(6): print("  ",k,v)

# --- 6. how good is the alternative in row 55060?
pair=Counter(); 
for s in steps: pair[(s["row"],s["action"])]+=1
for row in (55060, 35032):
    print(f"\nrow {row} digits {dg(row)}  Q {np.round(q[row],3)}  greedy {A[int(np.argmax(q[row]))]}")
    print("   visits by action:", {A[i]:pair[(row,A[i])] for i in range(6) if pair[(row,A[i])]})
