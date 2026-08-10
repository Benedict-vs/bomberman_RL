"""Does the greedy action follow digit 6, and by how much? A table read.

`loop_probe.py` rolls the policy out because a count of rows is not a
visitation distribution. This file is the complementary, deliberately *static*
measurement E19 pre-registers: it does not ask where the policy goes, it asks
whether the table has made up its mind at all.

The question is only well posed in rows where all three hold:

- the agent is **safe** (`own_danger == 0`), so digit 6 is a target direction
  and not an escape direction,
- digit 6 points somewhere (`target != NO_TARGET`),
- the neighbour in that direction reads `NB_CLEAR`, so the move is legal and
  not a step into a blast.

512 of the 12 800 rows qualify; a 100 000-episode run touches ~115 of them.
For each, the **margin** is Q(target move) − max Q(anything else): positive
means the greedy action follows the feature, negative means the table overrides
its own BFS. E18 found the corner-spawn row 3007 doing exactly that at margins
of 0.127 and 0.381 -- inside the noise the per-cell alpha leaves behind -- and
E19 shapes the reward to widen them.

    uv run python -m scratchpad.benedict.target_follow --baseline
    uv run python -m scratchpad.benedict.target_follow \
        agent_code/benedict_task2/q_table_e19_shape02_s*__ep100000.npy

Untouched rows (all-zero) are excluded: a row the run never reached has no
opinion to override, and counting it either way would say more about the arena
than about the table.
"""

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent_code.benedict_task2.callbacks import (  # noqa: E402
    ACTIONS, FEATURE_SIZES, NB_BLOCKED, NB_CLEAR, NO_TARGET,
)

# The state E18 pinned: top-left corner spawn (UP and LEFT walled), safe, a
# crate in bomb range, digit 6 pointing DOWN. Entered in ~21 % of rounds. Given
# as digits rather than as the index 3007 it had before E20 -- appending the
# distance digit multiplied every index by 5, and a hardcoded number would have
# gone on decoding to something plausible instead of failing.
CORNER_DIGITS = (NB_BLOCKED, NB_CLEAR, NB_CLEAR, NB_BLOCKED, 0, 3, 1)

BASELINE = "agent_code/benedict_task2/q_table_e16_c5_k03_s{i}__ep100000.npy"


def decode(index: int) -> tuple[int, ...]:
    """Inverse of `callbacks.encode` -- mixed radix, most significant first."""

    digits = []
    for size in reversed(FEATURE_SIZES):
        digits.append(index % size)
        index //= size
    return tuple(reversed(digits))


def well_posed_rows() -> np.ndarray:
    """Rows where "does the greedy action follow digit 6" has an answer.

    Returns an (n, 2) array of (row index, action index of the target move).
    """

    rows = []
    for index in range(int(np.prod(FEATURE_SIZES))):
        digits = decode(index)
        neighbours, danger, target = digits[:4], digits[4], digits[5]
        if danger != 0 or target == NO_TARGET:
            continue
        if neighbours[target - 1] != NB_CLEAR:
            continue
        # E20 appended a distance digit; 0 there means "no target", which
        # contradicts target != NO_TARGET and marks an unreachable row.
        if len(FEATURE_SIZES) > 7 and digits[7] == 0:
            continue
        # `bfs_first_step` returns action_idx + 1, so target - 1 indexes both
        # the neighbour digits and ACTIONS -- DELTAS and ACTIONS share an order.
        rows.append((index, target - 1))
    return np.array(rows)


def report(path: str, rows: np.ndarray) -> dict:
    q = np.load(path)
    indices, targets = rows[:, 0], rows[:, 1]

    touched = np.abs(q[indices]).sum(axis=1) > 0
    indices, targets = indices[touched], targets[touched]

    values = q[indices]
    target_q = values[np.arange(len(indices)), targets]
    others = values.copy()
    others[np.arange(len(indices)), targets] = -np.inf
    margin = target_q - others.max(axis=1)

    corner_rows = [i for i in range(int(np.prod(FEATURE_SIZES)))
                   if decode(i)[:7] == CORNER_DIGITS
                   and (len(FEATURE_SIZES) == 7 or decode(i)[7] != 0)]
    # Pick the busiest variant: before E20 there is one row, after it there is
    # one per distance bucket and the agent is at d = 2 in ~78 % of visits.
    corner_row = max(corner_rows, key=lambda i: np.abs(q[i]).sum())
    corner = q[corner_row]
    corner_target = CORNER_DIGITS[5] - 1
    corner_rest = corner.copy()
    corner_rest[corner_target] = -np.inf

    return {
        "name": Path(path).stem.replace("q_table_", ""),
        "corner_row": corner_row,
        "touched": len(indices),
        "follows": float((margin > 0).mean() * 100),
        "mean_margin": float(margin.mean()),
        "median_margin": float(np.median(margin)),
        "thin": float((np.abs(margin) < 0.2).mean() * 100),
        "corner_action": ACTIONS[int(np.argmax(corner))],
        "corner_margin": float(corner[corner_target] - corner_rest.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tables", nargs="*", help="paths to .npy Q-tables")
    parser.add_argument("--baseline", action="store_true",
                        help="also read the five e16_c5_k03 @100k tables")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args()

    paths = []
    if args.baseline:
        paths += [BASELINE.format(i=i) for i in range(5)]
    for pattern in args.tables:
        paths += sorted(glob.glob(pattern)) or [pattern]
    if not paths:
        parser.error("give at least one table, or --baseline")

    rows = well_posed_rows()
    print(f"well-posed rows: {len(rows)} of {int(np.prod(FEATURE_SIZES))}\n")

    results = [report(p, rows) for p in paths]
    if args.markdown:
        print("| table | touched | follows | mean margin | thin (<0.2) | row 3007 |")
        print("|---|---|---|---|---|---|")
        for r in results:
            print(f"| `{r['name']}` | {r['touched']} | {r['follows']:.1f} % | "
                  f"{r['mean_margin']:+.3f} | {r['thin']:.1f} % | "
                  f"{r['corner_action']} ({r['corner_margin']:+.3f}) |")
    else:
        for r in results:
            print(f"{r['name']:34s} touched {r['touched']:4d}  "
                  f"follows {r['follows']:5.1f} %  "
                  f"mean margin {r['mean_margin']:+7.3f}  "
                  f"median {r['median_margin']:+7.3f}  "
                  f"thin {r['thin']:5.1f} %  "
                  f"corner[{r['corner_row']:5d}] {r['corner_action']:5s} "
                  f"({r['corner_margin']:+.3f})")

    follows = [r["follows"] for r in results]
    print(f"\nfollow rate: min {min(follows):.1f} %  max {max(follows):.1f} %  "
          f"mean {np.mean(follows):.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
