import numpy as np, sys, importlib.util, os
sys.path.insert(0,os.path.abspath("."))
spec=importlib.util.spec_from_file_location("cb","agent_code/benedict_task4/callbacks.py")
cb=importlib.util.module_from_spec(spec); spec.loader.exec_module(cb)
FS=cb.FEATURE_SIZES
def digits(i):
    d=[];r=i
    for s in reversed(FS): d.append(r%s); r//=s
    return list(reversed(d))
for name in ["checkpoints/benedict_task4/q_table_e31_S0_s80__ep20000.npy",
             "checkpoints/benedict_task4/q_table_e31_S0_s80__ep5000.npy",
             "checkpoints/benedict_task4/q_table_rung2ship.npy"]:
    if not os.path.exists(name): print("MISSING",name); continue
    q=np.load(name)
    nz=np.abs(q).sum(1)>0
    V=q.max(1)
    print(f"\n=== {os.path.basename(name)}  shape {q.shape}  rows with value {nz.sum()}")
    print(f"  V over valued rows: mean {V[nz].mean():.3f} median {np.median(V[nz]):.3f} p5 {np.percentile(V[nz],5):.3f} p95 {np.percentile(V[nz],95):.3f} max {V.max():.3f} min {V.min():.3f}")
    # split by own_danger digit (index 4)
    idx=np.arange(len(q))
    dig4=(idx//(FS[5]*FS[6]*FS[7]))%FS[4]
    for dv in range(5):
        m=nz&(dig4==dv)
        if m.sum(): print(f"   own_danger={dv}: rows {m.sum():5d}  V mean {V[m].mean():7.3f} median {np.median(V[m]):7.3f}  minQ {q[m].min():7.3f}")
