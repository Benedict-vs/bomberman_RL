import numpy as np, sys
for s in (81,83):
    src=f"checkpoints/benedict_task4/q_table_e31_S0_s{s}__ep20000.npy"
    q=np.load(src); q[55060,0]=q[55060].max()+1.0
    np.save(f"checkpoints/benedict_task4/q_table_a5flip_s{s}.npy", q)
    print("wrote flip for seed",s)
