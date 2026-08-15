import numpy as np, os, sys, importlib.util
sys.path.insert(0,os.path.abspath("."))
spec=importlib.util.spec_from_file_location("cb","agent_code/benedict_task4/callbacks.py")
cb=importlib.util.module_from_spec(spec); spec.loader.exec_module(cb)
FS=cb.FEATURE_SIZES; N=cb.N_STATES
def digits(i):
    d=[];r=i
    for s in reversed(FS): d.append(r%s); r//=s
    return list(reversed(d))
DG=[digits(i) for i in range(N)]
for s in (81,83):
    q=np.load(f"checkpoints/benedict_task4/q_table_e31_S0_s{s}__ep20000.npy"); n=0
    for i in range(N):
        d=DG[i]
        if d[4]>0 and d[5]!=0 and np.abs(q[i]).sum()>0 and int(np.argmax(q[i]))!=d[5]-1:
            q[i,d[5]-1]=q[i].max()+1.0; n+=1
    np.save(f"checkpoints/benedict_task4/q_table_a5esc_s{s}.npy", q); print(s,"rows re-pointed",n)
