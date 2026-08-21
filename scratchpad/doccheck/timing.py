import csv, glob, statistics as st
rows_n=0; mx=0.0; means=[]; agents=set(); files=0; mxfile=None
tot_steps=0; wsum=0.0
for p in glob.glob("results/eval/**/*.csv", recursive=True):
    with open(p) as f:
        rd=csv.DictReader(f)
        if "think_max_ms" not in (rd.fieldnames or []): continue
        files+=1
        for r in rd:
            if not r["agent"].startswith("benedict"): continue
            rows_n+=1
            v=float(r["think_max_ms"])
            if v>mx: mx=v; mxfile=p
            steps=float(r["steps"]) if r["steps"] else 0
            means.append(float(r["think_mean_ms"]))
            tot_steps+=steps; wsum+=float(r["think_mean_ms"])*steps
print("csv files scanned:",files)
print("benedict agent-rounds:",rows_n)
print("max think_max_ms:",mx,"in",mxfile)
print("unweighted mean of think_mean_ms:",round(st.mean(means),4))
print("step-weighted mean think_ms:",round(wsum/tot_steps,4))
