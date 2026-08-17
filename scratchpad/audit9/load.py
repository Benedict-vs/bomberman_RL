"""Load all E37 evaluation CSVs. Read-only. numpy + csv only (no pandas in this venv)."""
from __future__ import annotations
import csv, glob, os, re, pickle
import numpy as np

ROOT = "/Users/benedictvonschubert/Projects/bomberman_RL"
PAT = os.path.join(ROOT, "results/eval/task4_tournament/benedict_q_e37_*__task4_rb_val550731.csv")
RX = re.compile(r"benedict_q_e37_(?P<arm>\w+?)_s(?P<seed>\d+)__ep(?P<ep>\d+)__")

NUM = ["survived", "round_steps", "score", "coins", "kills", "suicides", "crates",
       "bombs", "moves", "invalid", "steps", "think_mean_ms", "think_max_ms",
       "think_over_limit", "died", "killed_by_opponent", "rank", "won"]


def load(agent="benedict_task4"):
    """-> dict[(arm, seed, ep)] = dict[col] = np.array over 1000 rounds (our agent)."""
    out = {}
    for f in sorted(glob.glob(PAT)):
        m = RX.search(os.path.basename(f))
        key = (m.group("arm"), int(m.group("seed")), int(m.group("ep")))
        cols = {c: [] for c in NUM}
        rounds = []
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r["agent"] != agent:
                    continue
                rounds.append(int(r["round"]))
                for c in NUM:
                    cols[c].append(float(r[c]))
        d = {c: np.array(v) for c, v in cols.items()}
        d["round"] = np.array(rounds)
        out[key] = d
    return out


if __name__ == "__main__":
    d = load()
    print(len(d), "runs")
    with open(os.path.join(ROOT, "scratchpad/audit9/e37.pkl"), "wb") as fh:
        pickle.dump(d, fh)
    arms = sorted({k[0] for k in d})
    eps = sorted({k[2] for k in d})
    for a in arms:
        print(a, [sum(1 for k in d if k[0] == a and k[2] == e) for e in eps],
              "rounds:", sorted({len(d[k]["score"]) for k in d if k[0] == a}))
