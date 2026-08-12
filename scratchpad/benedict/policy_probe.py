"""What is the policy actually doing? Action distribution against table state.

E25 produced an agent that drops ~24 bombs a round and destroys 0.14 crates. A
bomb with a crate in range always destroys it, so those bombs are being placed in
states where digit 7 (`bomb_useful`) is 0. Two very different explanations:

  learned    BOMB genuinely has the highest Q in those rows -- the reward table
             is being exploited and the fix is the reward table
  degenerate the row is all zeros (or a near-tie), so `act` falls through to
             `policy_rng.choice` and BOMB is one of six coin flips -- the fix is
             coverage, and the bombs are noise rather than strategy

This replicates `act`'s choice rule exactly and reports which it is.

Run:  BM_MODEL_SUFFIX=_e25_B_s10__ep40000 uv run python scratchpad/benedict/policy_probe.py \
        --agent benedict_task3 --opponents coin_collector_agent --n-rounds 30
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from environment import BombeRLeWorld, WorldArgs        # noqa: E402


def load_callbacks(agent: str):
    path = ROOT / "agent_code" / agent / "callbacks.py"
    spec = importlib.util.spec_from_file_location(f"{agent}_cb", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agent", default="benedict_task3")
    ap.add_argument("--opponents", default="coin_collector_agent")
    ap.add_argument("--n-opponents", type=int, default=3)
    ap.add_argument("--n-rounds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=550731)
    args = ap.parse_args()

    cb = load_callbacks(args.agent)
    q = np.load(cb.MODEL_FILE)
    print(f"table {Path(cb.MODEL_FILE).name}  rows with value: "
          f"{int((np.abs(q).sum(1) > 0).sum())} / {len(q)}")

    log_dir = ROOT / "logs" / "policy_probe"
    log_dir.mkdir(parents=True, exist_ok=True)
    wargs = WorldArgs(
        no_gui=True, fps=1000, turn_based=False, update_interval=0.1,
        save_replay=False, replay=None, make_video=False, continue_without_training=True,
        log_dir=str(log_dir), save_stats=False, match_name="policy_probe",
        seed=None, silence_errors=True, scenario="classic",
    )
    line_up = [args.agent] + [args.opponents] * args.n_opponents
    world = BombeRLeWorld(wargs, [(n, False) for n in line_up])

    actions = Counter()
    bomb_useful_when_bombing = Counter()
    tie_size = Counter()
    zero_rows = 0
    steps = 0
    bomb_from_tie = 0
    bomb_total = 0
    useful_available = 0            # steps where a bomb here *would* hit a crate
    bomb_margins: list[float] = []   # Q(BOMB) - Q(second best), when BOMB wins
    chosen_q: list[float] = []

    for r in range(args.n_rounds):
        world.rng = np.random.default_rng(args.seed + r)
        np.random.seed(args.seed + r)
        world.new_round()
        world.user_input = None
        me = world.agents[0]

        while world.running:
            if not me.dead:
                gs = world.get_state_for_agent(me)
                row = cb.state_to_features(gs)
                qr = q[row]
                # Exactly callbacks.act's rule.
                best = np.flatnonzero(qr >= qr.max() - cb.TIE_TOL)
                chosen = cb.ACTIONS[int(best[0])] if best.size == 1 else "TIE"
                steps += 1
                if not np.abs(qr).sum():
                    zero_rows += 1
                tie_size[int(best.size)] += 1

                # Digit 7 is `bomb_useful`; recompute it rather than decode the row.
                x, y = gs["self"][3]
                useful = int(gs["self"][2] and cb.bomb_hits_crate(x, y, gs["field"]))
                useful_available += useful
                chosen_q.append(float(qr.max()))

                if chosen == "TIE":
                    actions["<tie -> random>"] += 1
                    if cb.ACTIONS.index("BOMB") in best:
                        bomb_from_tie += 1
                else:
                    actions[chosen] += 1
                    if chosen == "BOMB":
                        bomb_total += 1
                        bomb_useful_when_bombing[useful] += 1
                        runner_up = np.partition(qr, -2)[-2]
                        bomb_margins.append(float(qr.max() - runner_up))
            world.do_step()

    print(f"\nsteps observed: {steps}")
    print(f"all-zero rows : {100 * zero_rows / steps:.2f} % of steps")
    print("\ngreedy action (ties reported separately, since act() randomises them):")
    for a, c in actions.most_common():
        print(f"  {a:<18}{100 * c / steps:6.2f} %")
    print(f"\ntie width distribution: {dict(sorted(tie_size.items()))}")
    print(f"BOMB is among the tied actions in {100 * bomb_from_tie / max(steps,1):.2f} % of steps")
    if bomb_total:
        u = bomb_useful_when_bombing
        tot = sum(u.values())
        print(f"\nwhen BOMB is the *unique* argmax ({bomb_total} times): "
              f"crate in range {100 * u.get(1,0) / tot:.1f} %, "
              f"nothing in range {100 * u.get(0,0) / tot:.1f} %")
    else:
        print("\nBOMB was never the unique argmax -- every bomb came from a tie.")

    print(f"\na bomb dropped here would hit a crate in {100 * useful_available / steps:.2f} % "
          f"of steps (digit 7 = 1) -- i.e. how often the agent is even next to a crate")
    if bomb_margins:
        mg = np.array(bomb_margins)
        print(f"Q(BOMB) - Q(second best) when BOMB wins: median {np.median(mg):.4f}, "
              f"mean {mg.mean():.4f}, 10th pct {np.percentile(mg,10):.4f}, max {mg.max():.4f}")
        print(f"  below 0.01 (the E23 tie tolerance): {100 * (mg < 0.01).mean():.1f} % of them")
    cq = np.array(chosen_q)
    print(f"Q of the chosen action: median {np.median(cq):.3f}, "
          f"min {cq.min():.3f}, max {cq.max():.3f}")


if __name__ == "__main__":
    main()
