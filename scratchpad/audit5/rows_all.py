import numpy as np, glob, os, sys, importlib.util
sys.path.insert(0,os.path.abspath("."))
spec=importlib.util.spec_from_file_location("cb","agent_code/benedict_task4/callbacks.py")
cb=importlib.util.module_from_spec(spec); spec.loader.exec_module(cb)
A=cb.ACTIONS
for row,label in ((59160,"59160 (danger 3, escape UP)"),(55060,"55060 (danger 1, escape UP)")):
    print(f"\n=== row {label} — greedy action and margin at ep20000")
    for pat in ("checkpoints/benedict_task4/q_table_e31_S0_s8?__ep20000.npy",
                "checkpoints/benedict_task4/q_table_a5ctl_s9?__ep20000.npy",
                "checkpoints/benedict_task4/q_table_a5k15_s9?__ep20000.npy",
                "checkpoints/benedict_task4/q_table_a5k30_s9?__ep20000.npy"):
        for f in sorted(glob.glob(pat)):
            q=np.load(f)[row]; o=np.argsort(-q)
            print(f"  {os.path.basename(f):42s} {A[int(o[0])]:5s} margin {q[o[0]]-q[o[1]]:7.3f} over {A[int(o[1])]:5s}   escapeQ(UP)={q[0]:7.3f}")
