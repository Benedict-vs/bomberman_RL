"""Visitation-weighted value statistics: what the policy actually reads.

E25 reported "median decision margin 0.1156 -> 0.0003" from a rollout probe.
This recomputes it independently and adds the row-concentration figure, which
says how much of the game is decided in how few rows.
"""
from __future__ import annotations
import argparse, os, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="_rung2ship")
    ap.add_argument("--hunt", default="1")
    ap.add_argument("--n-rounds", type=int, default=40)
    ap.add_argument("--seed", type=int, default=550731)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    os.environ["BM_MODEL_SUFFIX"] = a.suffix
    os.environ["BM_HUNT"] = a.hunt
    os.environ["BM_QUIET_LOGS"] = "1"

    import numpy as np, importlib
    from environment import BombeRLeWorld, WorldArgs
    cb = importlib.import_module("agent_code.benedict_task3.callbacks")

    rows, margins, chosen_q, is_bomb = [], [], [], []
    orig_act = cb.act

    def probe_act(self, gs):
        s = cb.state_to_features(gs)
        q = self.q[s]
        srt = np.sort(q)
        rows.append(s); margins.append(srt[-1] - srt[-2]); chosen_q.append(srt[-1])
        is_bomb.append(int(np.argmax(q) == 5))
        return orig_act(self, gs)
    cb.act = probe_act

    log_dir = ROOT / "logs" / "audit2"; log_dir.mkdir(parents=True, exist_ok=True)
    wargs = WorldArgs(no_gui=True, fps=1000, turn_based=False, update_interval=0.1,
                      save_replay=False, replay=None, make_video=False,
                      continue_without_training=True, log_dir=str(log_dir),
                      save_stats=False, match_name="audit2m", seed=a.seed,
                      silence_errors=False, scenario="classic")
    world = BombeRLeWorld(wargs, [(n, False) for n in
                                  ["benedict_task3"] + ["coin_collector_agent"] * 3])
    for r in range(a.n_rounds):
        world.rng = np.random.default_rng(a.seed + r)
        np.random.seed(a.seed + r)
        world.new_round(); world.user_input = None
        while world.running:
            world.do_step()

    m = np.array(margins); cq = np.array(chosen_q)
    c = Counter(rows); tot = len(rows)
    top = c.most_common(10)
    cum5 = sum(n for _, n in c.most_common(5)) / tot
    cum20 = sum(n for _, n in c.most_common(20)) / tot
    print(f"{a.tag or a.suffix:<26} steps {tot:6d}  distinct rows {len(c):4d}  "
          f"top5 {cum5:.2f} top20 {cum20:.2f}  |  Q(chosen) med {np.median(cq):8.4f}  "
          f"margin med {np.median(m):9.5f}  p25 {np.percentile(m,25):9.5f}  "
          f"frac<1e-3 {np.mean(m<1e-3):.3f}  frac<1e-2 {np.mean(m<1e-2):.3f}  "
          f"BOMB argmax {np.mean(is_bomb):.3f}", flush=True)


if __name__ == "__main__":
    main()
