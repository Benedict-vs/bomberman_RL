"""Static traps in a Q-table, found without starting the game engine.

`loop_check.py` replays the greedy policy and asks *when* the agent stops making
progress. This asks a cheaper question first: **which rows are traps at all?**
It reads only the table and the feature layout, so it runs in under a second and
can be used as a gate before committing to a 300-round evaluation.

The check that motivates the file is E09. The rung-1 model froze on its starting
tile in all 300 rounds on `classic`, because three of the four corner rows have
an argmax pointing at a blocked neighbour. An invalid action leaves the state
unchanged, so that argmax is not a small error -- it is a fixed point, and the
round is over at step 1. All of it was visible in the shipped table by
inspection; the 300-round run only confirmed it eight minutes later.

The other half of E09's finding is that *which* rows matter depends on the
scenario, not on training: `blocked` uses `field != 0`, so crates count as walls
and `classic` pushes 79 % of free tiles into wall patterns that `coin-heaven`
essentially never produces. So the report weights every finding by how often its
wall pattern actually occurs on the target scenario.

    uv run python -m scratchpad.benedict.table_check --agent benedict_task2
    uv run python -m scratchpad.benedict.table_check --agent tabular_q_task1 \
        --scenario coin-heaven

Exit code is 1 only for a *provable* failure -- a starting corner that freezes on
step 1, which ends every round before it begins -- so it can gate a sweep:

    uv run python -m scratchpad.benedict.table_check --agent X && ./train.sh

Everything else is printed as a warning and exits 0. The frequency weighting
covers the wall digits only, so a common wall pattern does not prove the *row*
is reachable; that judgement stays with the reader.

**Contract.** The agent's `FEATURE_SIZES` must start with the four wall bits in
`DELTAS` order (up, right, down, left). Both current agents satisfy this; the
script refuses to guess if a future one does not.
"""

import argparse
import importlib
import os
import sys

import numpy as np

import settings as s

# Below this share of free tiles a wall pattern is reported but not treated as a
# reason to fail: it exists on the board, but an agent may never stand there.
NEGLIGIBLE = 0.005


def decode(index: int, sizes: tuple[int, ...]) -> tuple[int, ...]:
    """Inverse of the agent's `encode` -- mixed radix, most significant first."""

    digits = []
    for size in reversed(sizes):
        digits.append(index % size)
        index //= size
    return tuple(reversed(digits))


def wall_pattern_frequency(scenario: str, n_arenas: int = 300,
                           seed: int = 20260731) -> tuple[np.ndarray, float]:
    """Share of free tiles carrying each of the 16 wall patterns, by scenario.

    Replicates the crate/wall half of `environment.py:build_arena` rather than
    driving the engine: only the layout matters here, and this keeps the script
    free of the world, the agents and the 0.5 s step budget.

    Returns (frequency by wall code, mean crates per arena). The crate count is
    what decides whether coins are visible at step 1: `build_arena` fills coin
    positions from shuffled *crate* tiles first, so a scenario with more crates
    than coins starts with every coin hidden.

    Returned index is the wall-bit tuple read as a binary number, U R D L.
    """

    rng = np.random.default_rng(seed)
    density = s.SCENARIOS[scenario]["CRATE_DENSITY"]
    counts = np.zeros(16)
    crates = 0
    deltas = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    for _ in range(n_arenas):
        arena = np.zeros((s.COLS, s.ROWS), int)
        arena[rng.random((s.COLS, s.ROWS)) < density] = 1
        arena[:1, :] = arena[-1:, :] = -1
        arena[:, :1] = arena[:, -1:] = -1
        for x in range(s.COLS):
            for y in range(s.ROWS):
                if (x + 1) * (y + 1) % 2 == 1:
                    arena[x, y] = -1
        # Start positions and their four neighbours are cleared of crates. This
        # is what gives every corner exactly two blocked neighbours, which is
        # why E09's four poisoned rows were reachable on step 1 of every round.
        for (x, y) in start_positions():
            for (xx, yy) in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
                if arena[xx, yy] == 1:
                    arena[xx, yy] = 0

        crates += int((arena == 1).sum())
        for x in range(1, s.COLS - 1):
            for y in range(1, s.ROWS - 1):
                if arena[x, y] != 0:
                    continue
                bits = [int(arena[x + dx, y + dy] != 0) for dx, dy in deltas]
                counts[bits_to_code(bits)] += 1

    return counts / counts.sum(), crates / n_arenas


def start_positions() -> list[tuple[int, int]]:
    return [(1, 1), (1, s.ROWS - 2), (s.COLS - 2, 1), (s.COLS - 2, s.ROWS - 2)]


def bits_to_code(bits) -> int:
    return (bits[0] << 3) | (bits[1] << 2) | (bits[2] << 1) | bits[3]


def corner_wall_patterns() -> dict[tuple[int, int], int]:
    """Wall code of each starting corner. Crates there are cleared, so the only
    blocked neighbours are the two board walls -- fixed, on every scenario."""

    codes = {}
    for (x, y) in start_positions():
        bits = [int(y - 1 < 1), int(x + 1 > s.COLS - 2),
                int(y + 1 > s.ROWS - 2), int(x - 1 < 1)]
        codes[(x, y)] = bits_to_code(bits)
    return codes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", required=True,
                        help="folder under agent_code/, e.g. benedict_task2")
    parser.add_argument("--scenario", default="classic", choices=sorted(s.SCENARIOS),
                        help="scenario the table will be deployed on (default: classic)")
    parser.add_argument("--top", type=int, default=12,
                        help="how many rows to list per finding (default: 12)")
    args = parser.parse_args()

    cb = importlib.import_module(f"agent_code.{args.agent}.callbacks")
    sizes = tuple(cb.FEATURE_SIZES)
    actions = list(cb.ACTIONS)

    if len(sizes) < 4 or any(size != 2 for size in sizes[:4]):
        print(f"FEATURE_SIZES={sizes} does not start with four wall bits. "
              "This script cannot locate them; teach it the new layout rather "
              "than guessing.", file=sys.stderr)
        return 2

    if not os.path.isfile(cb.MODEL_FILE):
        print(f"No table at {cb.MODEL_FILE} -- train first, or set BM_MODEL_SUFFIX "
              "to point at one of a parallel sweep's tables.", file=sys.stderr)
        return 2

    q = np.load(cb.MODEL_FILE)
    n_states = int(np.prod(sizes))
    if q.shape != (n_states, len(actions)):
        print(f"Table shape {q.shape} does not match FEATURE_SIZES {sizes} "
              f"-> expected {(n_states, len(actions))}. Wrong model file?",
              file=sys.stderr)
        return 2

    freq, mean_crates = wall_pattern_frequency(args.scenario)
    coins_hidden = mean_crates >= s.SCENARIOS[args.scenario]["COIN_COUNT"]
    corners = corner_wall_patterns()
    rows_per_pattern = n_states // 16      # rows sharing one wall pattern

    print(f"table   : {cb.MODEL_FILE}")
    print(f"agent   : {args.agent}   scenario: {args.scenario}")
    print(f"layout  : FEATURE_SIZES={sizes} -> {n_states} rows x {len(actions)} actions")
    print()

    trained = np.array([not np.allclose(q[i], 0) for i in range(n_states)])
    print(f"coverage: {trained.sum()} of {n_states} rows carry a non-zero value "
          f"({trained.sum() / n_states:.1%})")

    # An untouched row ties across every action, so `act` tie-breaks uniformly
    # -- including BOMB. On a rung where BOMB is fatal, an untouched row the
    # agent can reach is a death sentence, not a missing optimisation.
    untouched_mass = sum(freq[code] for code in range(16)
                         if not trained[code * rows_per_pattern:
                                        (code + 1) * rows_per_pattern].any())
    print(f"          wall patterns with no trained row at all cover "
          f"{untouched_mass:.1%} of free tiles on {args.scenario}")
    print()

    # --- Finding 1: argmax into a blocked neighbour (the E09 failure) --------
    # Only trained rows: on an all-zero row np.argmax returns index 0, but `act`
    # tie-breaks uniformly, so the deterministic-argmax argument does not apply.
    # Untouched rows are a hazard of their own and are counted above.
    blocked_argmax = []
    for i in range(n_states):
        if not trained[i]:
            continue
        digits = decode(i, sizes)
        best = int(np.argmax(q[i]))
        if best < 4 and digits[best] == 1:
            blocked_argmax.append((i, digits, best))

    print(f"[1] argmax points at a blocked neighbour : {len(blocked_argmax)} "
          f"of {trained.sum()} trained rows")
    print("    An invalid action leaves the state unchanged, so such a row is")
    print("    absorbing whenever the rest of the state is static (no live bomb).")
    if blocked_argmax:
        reachable = [(i, d, a) for i, d, a in blocked_argmax
                     if freq[bits_to_code(d[:4])] > NEGLIGIBLE]
        print(f"    of which reachable on {args.scenario}: {len(reachable)}")
        for i, digits, best in blocked_argmax[:args.top]:
            # Share of free tiles carrying this wall pattern. The row is one of
            # several sharing it, so this is an upper bound on the row itself.
            share = freq[bits_to_code(digits[:4])]
            mark = "  <-- START CORNER" if bits_to_code(digits[:4]) in corners.values() else ""
            print(f"      row {i:5d}  {digits}  argmax={actions[best]:<5s} "
                  f"pattern<={share:6.2%}{mark}")
        if len(blocked_argmax) > args.top:
            print(f"      ... and {len(blocked_argmax) - args.top} more")
    print()

    # --- Finding 2: the four rows that decide step 1 of every round ----------
    # E09: the agent never left its spawn tile, so these four rows alone
    # determined 300 of 300 rounds. They are worth their own section.
    print("[2] starting-corner rows, all non-wall digits at 0")
    if coins_hidden:
        print(f"    {mean_crates:.0f} crates vs {s.SCENARIOS[args.scenario]['COIN_COUNT']} "
              f"coins on {args.scenario}: every coin starts under a crate, so this")
        print("    IS the state on step 1 of every round. A frozen row here ends")
        print("    300 rounds out of 300.")
    else:
        print(f"    {mean_crates:.0f} crates vs {s.SCENARIOS[args.scenario]['COIN_COUNT']} "
              f"coins on {args.scenario}: coins are visible from step 1, so the")
        print("    spawn state is NOT this row. Informational only here.")
    for pos, code in corners.items():
        row = code * rows_per_pattern       # all non-wall digits zero
        digits = decode(row, sizes)
        if np.allclose(q[row], 0):
            print(f"      {str(pos):9s} row {row:5d}  UNTRAINED -> uniform over "
                  f"{len(actions)} actions incl. BOMB")
            continue
        best = int(np.argmax(q[row]))
        stuck = best < 4 and digits[best] == 1
        note = "FROZEN (invalid, forever)" if stuck else \
               "waits forever" if actions[best] == "WAIT" else "moves"
        print(f"      {str(pos):9s} row {row:5d}  argmax={actions[best]:<5s} -> {note}")
    print()

    # --- Finding 3: ties, i.e. rows decided by the RNG ----------------------
    ties = [i for i in range(n_states)
            if trained[i] and (q[i] == q[i].max()).sum() > 1]
    print(f"[3] trained rows with a tied argmax      : {len(ties)}")
    print("    Decided by policy_rng, so the measured policy is not the argmax")
    print("    policy. Fine in small numbers; a symptom if the count is large.")
    for i in ties[:args.top]:
        tied = [actions[a] for a in np.flatnonzero(q[i] == q[i].max())]
        print(f"      row {i:5d}  {decode(i, sizes)}  {'/'.join(tied)}")
    print()

    # --- Finding 4: where BOMB wins -----------------------------------------
    bomb_rows = [i for i in range(n_states)
                 if trained[i] and actions[int(np.argmax(q[i]))] == "BOMB"]
    print(f"[4] trained rows preferring BOMB         : {len(bomb_rows)}")
    print("    Expected 0 on rung 1 and >0 on rung 2. Read against the rung:")
    print("    a rung-2 agent with 0 here cannot open a crate.")
    for i in bomb_rows[:args.top]:
        print(f"      row {i:5d}  {decode(i, sizes)}  q={np.round(q[i], 2)}")
    print()

    # --- Finding 5: value spread --------------------------------------------
    # A row whose actions are within noise of each other is one update away
    # from flipping its argmax. Maxi's E01/E02 diagnostic, kept because it
    # caught a broken policy that every aggregate number called healthy.
    spreads = np.array([np.ptp(q[i]) for i in range(n_states) if trained[i]])
    print(f"[5] value spread over trained rows       : "
          f"median {np.median(spreads):.3f}, min {spreads.min():.3f}, "
          f"max {spreads.max():.3f}")
    flat = int((spreads < 0.1).sum())
    print(f"    rows with spread < 0.1 (argmax is noise): {flat}")
    print()

    reachable_traps = [i for i, d, _ in blocked_argmax
                       if freq[bits_to_code(d[:4])] > NEGLIGIBLE]
    corner_traps = [pos for pos, code in corners.items()
                    if (lambda r: not np.allclose(q[r], 0)
                        and int(np.argmax(q[r])) < 4
                        and decode(r, sizes)[int(np.argmax(q[r]))] == 1)(code * rows_per_pattern)]

    if corner_traps and coins_hidden:
        print(f"FAIL: {len(corner_traps)} starting corner(s) freeze on step 1: "
              f"{corner_traps}")
        return 1
    if reachable_traps:
        # Not fatal: the wall pattern occurring often does not mean the *row*
        # does, because the non-wall digits are not weighted. On coin-heaven
        # these same three rows are only entered while standing on a coin, which
        # resolves the same step -- and that table scored 50/50.
        print(f"WARN: {len(reachable_traps)} rows have an argmax into a wall on a "
              f"wall pattern that is common on {args.scenario}. Check whether the "
              "remaining digits are reachable there.")
        return 0
    print("OK: no trained row has an argmax into a blocked neighbour.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
