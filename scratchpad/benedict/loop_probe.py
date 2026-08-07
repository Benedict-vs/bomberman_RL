"""Does the greedy policy actually go anywhere? A rollout, not a table read.

`table_check.py` asks which rows are traps by reading the Q-table. It found
nothing wrong with E10 -- and E10's agent stood on two tiles for 400 steps in
every round of all five seeds. Two reasons it could not see that:

1. It detects **period-1** fixed points (an argmax pointing into a wall, so the
   invalid action leaves the state unchanged). The recurring failure on this
   project is a **period-2** cycle built from perfectly legal moves: step left,
   the mirror row says step right, and the state after two steps is the one you
   started from. E04, E06 arm A and E10 are all this.
2. More generally, a count of rows in a table is not a visitation distribution.
   Every E10 seed had 17-34 rows whose argmax is `BOMB` while dropping ~0 bombs
   at eps = 0, because the greedy trajectory never reached those rows.

So this file rolls the policy out and measures where it goes. It drives
``BombeRLeWorld`` directly, like ``tools/evaluate.py``, and touches no framework
file. 20 rounds is enough to separate a healthy agent from a stuck one, which
makes it cheap enough to run as a gate *before* a 300-round evaluation:

    uv run python -m scratchpad.benedict.loop_probe --agent benedict_task2
    BM_MODEL_SUFFIX=_e11_s0 uv run python -m scratchpad.benedict.loop_probe \
        --agent benedict_task2 --rounds 40

Exit code 1 when the median round ends up confined to `--min-tiles` tiles or
fewer, which is a provable dead policy regardless of what the table looks like.

Deliberately *not* seeded with the evaluation seed 20260731: this is a
diagnostic run before the measurement, and it should not be the arenas the
result is reported on.
"""

import argparse
import collections
import sys

import numpy as np

from environment import BombeRLeWorld, WorldArgs

# Rounds are played at the agent's own eps -- which is 0 outside training, since
# `act` only consults self.eps when self.train is set. That is the policy that
# gets measured, so it is the policy that gets probed.
DEFAULT_SEED = 810731


def build_args(scenario: str) -> WorldArgs:
    """Only the fields the environment actually reads; mirrors evaluate.py."""

    return WorldArgs(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1,
        save_replay=False, replay=None, make_video=False,
        # Rounds run to their natural end rather than stopping when a training
        # agent dies -- the same reason evaluate.py sets it.
        continue_without_training=True,
        log_dir="logs", save_stats=False, match_name="loop_probe",
        seed=None, silence_errors=False, scenario=scenario,
    )


def play_round(args: WorldArgs, agent: str, seed: int):
    """One round. Returns the visited tiles in order and the actions taken."""

    world = BombeRLeWorld(args._replace(seed=seed), [(agent, False)])
    world.new_round()
    # BombeRLeWorld.do_step sets this, but get_state_for_agent reads it before
    # the first do_step ever runs.
    world.user_input = None

    tiles, actions = [], []
    while world.running:
        for a in world.active_agents:
            if world.get_state_for_agent(a) is not None:
                tiles.append((a.x, a.y))
        world.do_step("WAIT")
        for a in world.agents:
            if a.last_action is not None:
                actions.append(a.last_action)
    return tiles, actions


def cycle_entry(tiles: list, width: int) -> int | None:
    """First step after which the agent only ever revisits `width` tiles.

    `None` when it never confines itself, which is what a healthy round looks
    like. Reported as a step number rather than a yes/no because *when* the
    policy gives up is the useful number -- E10's seeds gave up at step 0, E04's
    around step 37.
    """

    for k in range(len(tiles)):
        if len(set(tiles[k:])) <= width:
            return k
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", required=True,
                        help="folder under agent_code/, e.g. benedict_task2")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--scenario", default="classic")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cycle-width", type=int, default=2,
                        help="tiles a 'cycle' may span (default: 2)")
    parser.add_argument("--stuck-before", type=float, default=0.9,
                        help="a cycle only counts as a failure if it is entered "
                             "within this fraction of the round (default: 0.9)")
    parser.add_argument("--min-tiles", type=int, default=2,
                        help="fail if the median round visits this many tiles "
                             "or fewer (default: 2)")
    args = parser.parse_args()

    world_args = build_args(args.scenario)
    per_round_tiles, entries, all_actions = [], [], collections.Counter()
    steps = []

    for r in range(args.rounds):
        tiles, actions = play_round(world_args, args.agent, args.seed + r)
        per_round_tiles.append(len(set(tiles)))
        entries.append(cycle_entry(tiles, args.cycle_width))
        all_actions.update(actions)
        steps.append(len(tiles))

    # A round that settles onto two tiles at step 396 of 400 has not given up --
    # it has finished. Only a cycle entered with real time left is a failure, so
    # the count is restricted to the first `--stuck-before` fraction of the
    # round. Without this the summary read "20/20 confined" for E13 tables that
    # were in fact playing the whole round.
    stuck = [e for e, n in zip(entries, steps)
             if e is not None and e < args.stuck_before * n]
    median_tiles = int(np.median(per_round_tiles))
    total = sum(all_actions.values())

    print(f"agent   : {args.agent}   scenario: {args.scenario}   "
          f"rounds: {args.rounds}   seeds: {args.seed}..{args.seed + args.rounds - 1}")
    print(f"steps per round      : median {int(np.median(steps))}")
    print(f"distinct tiles/round : median {median_tiles}  "
          f"(min {min(per_round_tiles)}, max {max(per_round_tiles)})")
    print(f"rounds confined to <={args.cycle_width} tiles before "
          f"{args.stuck_before:.0%} of the round : {len(stuck)}/{args.rounds}", end="")
    if stuck:
        print(f", entering at step median {int(np.median(stuck))} "
              f"(min {min(stuck)}, max {max(stuck)})")
    else:
        print()
    print("action mix           : " +
          "  ".join(f"{k} {v / total:.1%}" for k, v in all_actions.most_common()))
    print()

    if median_tiles <= args.min_tiles:
        print(f"FAIL: the median round never leaves {median_tiles} tiles. Whatever "
              "the table says, this policy does not play.")
        return 1
    if len(stuck) > args.rounds // 2:
        print(f"WARN: {len(stuck)}/{args.rounds} rounds end confined to "
              f"<={args.cycle_width} tiles. Productive early, absorbing later.")
        return 0
    print("OK: the policy keeps moving.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
