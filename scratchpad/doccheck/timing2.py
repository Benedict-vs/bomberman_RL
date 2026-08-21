import csv, glob, statistics as st, os
def scan(pat, agent_pref="benedict"):
    rows_n=0; mx=0.0; means=[]; mxfile=None; tot=0; wsum=0.0
    for p in sorted(glob.glob(pat)):
        with open(p) as f:
            rd=csv.DictReader(f)
            if "think_max_ms" not in (rd.fieldnames or []): continue
            for r in rd:
                if not r["agent"].startswith(agent_pref): continue
                rows_n+=1
                v=float(r["think_max_ms"])
                if v>mx: mx=v; mxfile=p
                means.append(float(r["think_mean_ms"]))
                s=float(r["steps"] or 0); tot+=s; wsum+=float(r["think_mean_ms"])*s
    print(f"{pat}: rounds={rows_n} max={mx:.4f} ({os.path.basename(mxfile) if mxfile else None}) meanOfMeans={st.mean(means):.4f} stepWeighted={wsum/tot:.4f}")
scan("results/eval/task4_tournament/*.csv")
scan("results/eval/task3_opponents/*.csv")
scan("results/eval/task2_crates/*.csv")
scan("results/eval/task1_coin_collectors/*.csv")
