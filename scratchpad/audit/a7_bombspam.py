"""The 25 invalid BOMBs per round: learned argmax, or noise? And which rows?"""
from __future__ import annotations
import argparse, importlib, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from environment import BombeRLeWorld, WorldArgs   # noqa: E402


def decode(cb, idx):
    out = []
    for size in reversed(cb.FEATURE_SIZES):
        out.append(idx % size); idx //= size
    return tuple(reversed(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="benedict_task3")
    ap.add_argument("--opponents", default="coin_collector_agent")
    ap.add_argument("--n-opponents", type=int, default=3)
    ap.add_argument("--n-rounds", type=int, default=60)
    ap.add_argument("--seed", type=int, default=550731)
    a = ap.parse_args()

    cb = importlib.import_module(f"agent_code.{a.agent}.callbacks")
    q = np.load(cb.MODEL_FILE)
    log_dir = ROOT / "logs" / "audit_bs"; log_dir.mkdir(parents=True, exist_ok=True)
    wargs = WorldArgs(no_gui=True, fps=1000, turn_based=False, update_interval=0.1,
                      save_replay=False, replay=None, make_video=False,
                      continue_without_training=True, log_dir=str(log_dir),
                      save_stats=False, match_name="audit_bs", seed=None,
                      silence_errors=False, scenario="classic")
    line_up = [a.agent] + [a.opponents] * a.n_opponents if a.n_opponents else [a.agent]
    world = BombeRLeWorld(wargs, [(n, False) for n in line_up])
    me = world.agents[0]
    rng = np.random.default_rng(cb.POLICY_SEED)

    margins = []; rows = Counter(); danger5 = Counter()
    n_have_bomb_steps = 0; n_no_bomb_steps = 0
    bomb_when_have = 0; bomb_when_not = 0
    for r in range(a.n_rounds):
        world.rng = np.random.default_rng(a.seed + r); np.random.seed(a.seed + r)
        world.new_round(); world.user_input = None
        while world.running:
            if not me.dead:
                gs = world.get_state_for_agent(me)
                idx = cb.state_to_features(gs)
                qr = q[idx]
                best = np.flatnonzero(qr >= qr.max() - cb.TIE_TOL)
                ai = int(best[0] if best.size == 1 else rng.choice(best))
                have = bool(gs["self"][2])
                n_have_bomb_steps += have; n_no_bomb_steps += (not have)
                if cb.ACTIONS[ai] == "BOMB":
                    if have: bomb_when_have += 1
                    else:
                        bomb_when_not += 1
                        margins.append(float(qr.max() - np.partition(qr, -2)[-2]))
                        d = decode(cb, idx)
                        rows[d] += 1
                        danger5[d[4]] += 1
            world.do_step()

    n = a.n_rounds
    print(f"steps with a bomb available : {n_have_bomb_steps/n:7.1f}/ep  -> BOMB chosen {bomb_when_have/n:6.1f}")
    print(f"steps with NO bomb available: {n_no_bomb_steps/n:7.1f}/ep  -> BOMB chosen {bomb_when_not/n:6.1f} "
          f"({100*bomb_when_not/max(n_no_bomb_steps,1):.1f} % of them)")
    m = np.array(margins)
    print(f"\nQ(BOMB)-Q(2nd) on those invalid BOMBs: median {np.median(m):.4f}  "
          f"10th {np.percentile(m,10):.4f}  90th {np.percentile(m,90):.4f}  "
          f"below 0.01: {100*(m<0.01).mean():.1f} %  exactly 0 (tie): {100*(m==0).mean():.1f} %")
    print(f"\ndigit 5 (own danger) on those steps: {dict(sorted(danger5.items()))}")
    print("\ntop rows choosing an invalid BOMB (digits 1-4 nb, 5 danger, 6 target, 7 bombuseful, 8 dist):")
    for d, c in rows.most_common(10):
        idx = 0
        for v, s_ in zip(d, cb.FEATURE_SIZES): idx = idx*s_ + v
        print(f"  {d}  n={c:5d}  Q={np.round(q[idx],3)}")


if __name__ == "__main__":
    main()
