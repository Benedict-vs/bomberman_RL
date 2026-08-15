import numpy as np, os, sys, importlib.util
sys.path.insert(0,os.path.abspath("."))
spec=importlib.util.spec_from_file_location("cb","agent_code/benedict_task4/callbacks.py")
cb=importlib.util.module_from_spec(spec); spec.loader.exec_module(cb)
A=cb.ACTIONS; FS=cb.FEATURE_SIZES
def dg(i):
    d=[];r=i
    for s in reversed(FS): d.append(r%s); r//=s
    return tuple(reversed(d))
rows=[55060,35032,59160]
for ep in (5000,10000,20000):
    print(f"\n########## ep{ep}")
    for row in rows:
        print(f" row {row} digits {dg(row)}")
        for s in range(80,85):
            f=f"checkpoints/benedict_task4/q_table_e31_S0_s{s}__ep{ep}.npy"
            if not os.path.exists(f): print("   missing",f); continue
            q=np.load(f)[row]
            o=np.argsort(-q)
            print(f"   s{s}: greedy {A[int(o[0])]:5s} margin {q[o[0]]-q[o[1]]:7.3f} over {A[int(o[1])]:5s}   Q {np.round(q,2)}")
q=np.load("checkpoints/benedict_task4/q_table_rung2ship.npy")
print("\nrung2ship warm-start parent:")
for row in rows:
    print(f"   row {row} {A[int(np.argmax(q[row]))]:5s} Q {np.round(q[row],2)}")
