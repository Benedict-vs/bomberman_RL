import csv, sys, statistics as st
from collections import defaultdict

def load(p):
    with open(p) as f:
        return list(csv.DictReader(f))

def summarize(path, cols=("score","won","kills","suicides","killed_by_opponent","crates","coins","survived","bombs","think_max_ms","think_mean_ms","think_over_limit","steps")):
    rows = load(path)
    by = defaultdict(list)
    for r in rows:
        by[r["agent"]].append(r)
    n_rounds = len({r["round"] for r in rows})
    print(f"== {path.split('/')[-1]}  rounds={n_rounds}")
    for a, rs in sorted(by.items()):
        out = []
        for c in cols:
            if c not in rs[0]: continue
            vals = [float(x[c]) for x in rs]
            out.append(f"{c}={st.mean(vals):.4f}")
        print("   ", a, " ".join(out))
    return by

if __name__ == "__main__":
    for p in sys.argv[1:]:
        summarize(p)
