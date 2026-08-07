"""Open up one absorbing 2-cycle and say *why* it closes.

`loop_probe.py` counts how often the greedy policy gets stuck and when. It does
not say what the two rows involved actually contain, and three experiments in a
row (E10, E12, E13) named a mechanism for the cycle that later turned out to be
partly wrong. This looks at one directly.

For the first round that collapses early it prints, for both tiles of the cycle:
the row index, the decoded digits, the whole Q-row, and the action taken. Then it
runs the test that decides the next experiment -- **are the two rows related by a
symmetry of the board?**

    BM_MODEL_SUFFIX=_e13_s1__ep40000 \
        uv run python -m scratchpad.benedict.cycle_dump --agent benedict_task2

Three outcomes, three different next steps:

* **digit 6 flips** between the two tiles -- the target direction is unstable
  between two near-equidistant crates, and the fix is the targeting rule.
* **the rows are D4 images of each other but their Q-values are not** -- they
  were learned independently from too little data, and canonicalising the table
  by its symmetry group is the fix.
* **rows unrelated, values consistent** -- the cycle is what the reward function
  actually asks for, and the fix is in the rewards.

The D4 part matters because the obvious argument cuts both ways: merging mirrored
rows forces mirrored actions, which in a corridor is *exactly* "go back where you
came from". Canonicalisation might enforce this cycle rather than break it, and
that is decided by whether the two rows are mirror images or not.
"""

import argparse
import importlib
import itertools
import os
import sys

import numpy as np

from environment import BombeRLeWorld, WorldArgs

DEFAULT_SEED = 810731

# The dihedral group D4 as permutations of the four directions in DELTAS order
# (up, right, down, left), which is a clockwise cycle. `perm[i] = j` reads "what
# is direction i in the transformed frame was direction j in the original".
# Rotations are cyclic shifts; reflections reverse the cycle.
D4 = [tuple((i + k) % 4 for i in range(4)) for k in range(4)] + \
     [tuple((k - i) % 4 for i in range(4)) for k in range(4)]


def decode(index: int, sizes: tuple[int, ...]) -> tuple[int, ...]:
    digits = []
    for size in reversed(sizes):
        digits.append(index % size)
        index //= size
    return tuple(reversed(digits))


def transform_digits(digits: tuple[int, ...], perm: tuple[int, ...]) -> tuple[int, ...]:
    """Apply a D4 element to a feature tuple.

    Neighbour digits permute with the group. Digit 6 is a *direction* (0 means
    "no target" and is fixed), so it permutes too -- but inversely, because it
    names a direction rather than indexing a slot. Grace and bomb-payoff are
    scalars and invariant.
    """

    neighbours = tuple(digits[perm[i]] for i in range(4))
    target = digits[5]
    if target != 0:
        target = perm.index(target - 1) + 1
    return neighbours + (digits[4], target) + digits[6:]


def transform_q(q_row: np.ndarray, perm: tuple[int, ...]) -> np.ndarray:
    """Same relabelling applied to the six action-values. WAIT and BOMB are fixed."""

    return np.concatenate([q_row[list(perm)], q_row[4:]])


def play_until_stuck(agent: str, scenario: str, seed: int, rounds: int, width: int):
    """First round whose trajectory collapses onto `width` tiles early on."""

    args = WorldArgs(
        no_gui=True, fps=15, turn_based=False, update_interval=0.1,
        save_replay=False, replay=None, make_video=False,
        continue_without_training=True, log_dir="logs", save_stats=False,
        match_name="cycle_dump", seed=None, silence_errors=False, scenario=scenario)

    for r in range(rounds):
        world = BombeRLeWorld(args._replace(seed=seed + r), [(agent, False)])
        world.new_round()
        world.user_input = None
        tiles, states, actions = [], [], []
        cb = importlib.import_module(f"agent_code.{agent}.callbacks")
        while world.running:
            for a in world.active_agents:
                st = world.get_state_for_agent(a)
                if st is not None:
                    tiles.append((a.x, a.y))
                    states.append(cb.state_to_features(st))
            world.do_step("WAIT")
            for a in world.agents:
                if a.last_action is not None:
                    actions.append(a.last_action)

        for k in range(len(tiles)):
            if len(set(tiles[k:])) <= width and k < 0.5 * len(tiles):
                return seed + r, k, tiles, states, actions
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--scenario", default="classic")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--cycle-width", type=int, default=2)
    args = parser.parse_args()

    cb = importlib.import_module(f"agent_code.{args.agent}.callbacks")
    sizes = tuple(cb.FEATURE_SIZES)
    actions = list(cb.ACTIONS)
    if not os.path.isfile(cb.MODEL_FILE):
        print(f"No table at {cb.MODEL_FILE}", file=sys.stderr)
        return 2
    q = np.load(cb.MODEL_FILE)

    found = play_until_stuck(args.agent, args.scenario, args.seed,
                             args.rounds, args.cycle_width)
    if found is None:
        print(f"No round out of {args.rounds} collapses in its first half. "
              "Nothing to dump -- this table does not have the failure.")
        return 0

    seed, k, tiles, states, taken = found
    print(f"table : {os.path.basename(cb.MODEL_FILE)}")
    print(f"round : world seed {seed}, collapses at step {k} of {len(tiles)}")
    print(f"cycle : {sorted(set(tiles[k:]))}")
    print()

    # The distinct (tile, row) pairs the agent alternates between, in the order
    # it first meets them.
    seen, cycle = [], []
    for t, s in zip(tiles[k:], states[k:]):
        if (t, s) not in seen:
            seen.append((t, s))
            cycle.append((t, s))
    for tile, row in cycle:
        digits = decode(row, sizes)
        best = int(np.argmax(q[row]))
        print(f"tile {tile}  row {row}")
        print(f"  digits      {digits}   "
              f"(neighbours {digits[:4]}, grace {digits[4]}, "
              f"target {'none' if digits[5] == 0 else actions[digits[5] - 1]}, "
              f"bomb pays {digits[6]})")
        print(f"  Q           {np.round(q[row], 3)}")
        print(f"  argmax      {actions[best]}   spread {np.ptp(q[row]):.3f}")
        print()

    if len(cycle) != 2:
        print(f"Cycle has {len(cycle)} distinct states, not 2 -- the pairwise "
              "symmetry test below assumes two.")
        return 0

    (t_a, row_a), (t_b, row_b) = cycle
    d_a, d_b = decode(row_a, sizes), decode(row_b, sizes)

    print("--- does digit 6 flip between the two tiles?")
    tgt = [('none' if d[5] == 0 else actions[d[5] - 1]) for d in (d_a, d_b)]
    print(f"    {t_a}: {tgt[0]}      {t_b}: {tgt[1]}")
    if tgt[0] != tgt[1] and 'none' not in tgt:
        print("    -> FLIPS. The target direction is unstable between the two")
        print("       tiles, so the agent is chasing two different crates.")
        print("       Next experiment is the targeting rule, not the symmetry.")
    print()

    print("--- are the two rows related by a symmetry of the board?")
    match = None
    for perm in D4:
        if transform_digits(d_a, perm) == d_b:
            match = perm
            break
    if match is None:
        print("    No D4 element maps one row onto the other. They are genuinely")
        print("    different states, so canonicalising by symmetry would neither")
        print("    merge them nor change this cycle. Look at the rewards instead.")
        return 0

    print(f"    Yes, under permutation {match}.")
    mapped = transform_q(q[row_a], match)
    delta = mapped - q[row_b]
    print(f"    Q(A) mapped   {np.round(mapped, 3)}")
    print(f"    Q(B)          {np.round(q[row_b], 3)}")
    print(f"    difference    {np.round(delta, 3)}   max |diff| {np.abs(delta).max():.3f}")
    if np.abs(delta).max() > 0.1:
        print("    -> The two rows describe the same situation and disagree. They")
        print("       were learned independently; canonicalising the table would")
        print("       merge them and share the data. That is the D4 experiment.")
    else:
        print("    -> The rows already agree, so canonicalisation would not change")
        print("       the values -- and merging them forces mirrored actions,")
        print("       which is this cycle. D4 would entrench it, not fix it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
