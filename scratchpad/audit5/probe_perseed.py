import csv, glob, numpy as np, os
M=("score","won","suicides","survived","crates","coins","kills","killed_by_opponent","bombs")
print(f"{'run':>26}"+"".join(f"{m[:9]:>10}" for m in M))
for f in sorted(glob.glob("scratchpad/audit5/eval/a5*_s9?__ep*.csv"))+ \
         sorted(glob.glob("results/eval/task4_tournament/benedict_q_e31_S0_s8?__ep*__task4_rb_val550731.csv")):
    rows=[r for r in csv.DictReader(open(f)) if r["agent"]=="benedict_task4"]
    rows=[r for r in rows if int(r["round"])<300]
    g=lambda k: np.mean([float(r[k]) for r in rows])
    n=os.path.basename(f).replace("__task4_rb_val550731.csv","").replace(".csv","").replace("benedict_q_","")
    print(f"{n:>26}"+"".join(f"{g(m):10.3f}" for m in M))
