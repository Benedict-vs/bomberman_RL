"""Two surgically edited tables, to bound what 'get the ordering right' is worth.

  _a5flip  : ONLY row 55060 -> escape direction (UP) made greedy. Nothing else.
  _a5esc   : every row with own_danger>0 and a valid escape digit -> that
             direction made greedy. Upper bound on 'obey digit 6 while in danger'.
Neither is shippable (the second is effectively rule-based inside the table); they
are diagnostics that answer 'how much headroom is in the argmax?'.
"""
import numpy as np, os, sys, importlib.util
sys.path.insert(0,os.path.abspath("."))
spec=importlib.util.spec_from_file_location("cb","agent_code/benedict_task4/callbacks.py")
cb=importlib.util.module_from_spec(spec); spec.loader.exec_module(cb)
FS=cb.FEATURE_SIZES; N=cb.N_STATES
src="checkpoints/benedict_task4/q_table_e31_S0_s80__ep20000.npy"
def digits(i):
    d=[];r=i
    for s in reversed(FS): d.append(r%s); r//=s
    return list(reversed(d))
q=np.load(src); out=q.copy()
out[55060,0]=out[55060].max()+1.0
np.save("checkpoints/benedict_task4/q_table_a5flip_s80.npy", out)
q2=np.load(src); n=0
for i in range(N):
    d=digits(i)
    if d[4]>0 and d[5]!=0 and np.abs(q2[i]).sum()>0:
        a=d[5]-1
        if int(np.argmax(q2[i]))!=a:
            q2[i,a]=q2[i].max()+1.0; n+=1
np.save("checkpoints/benedict_task4/q_table_a5esc_s80.npy", q2)
print("rows re-pointed in _a5esc:", n)
