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
scenario, not on training: on `classic` roughly four free tiles in five sit in a
wall pattern that `coin-heaven` essentially never produces. So every finding is
weighted by how often its wall pattern actually occurs on the target scenario.

    uv run python -m scratchpad.benedict.table_check --agent benedict_task2
    uv run python -m scratchpad.benedict.table_check --agent tabular_q_task1 \
        --scenario coin-heaven

Exit code is 1 only for a *provable* failure -- a starting corner whose row is
absorbing, which ends every round before it begins -- so it can gate a sweep:

    uv run python -m scratchpad.benedict.table_check --agent X && ./train.sh

Everything else is printed as a warning and exits 0.

**Contract.** The first four entries of the agent's `FEATURE_SIZES` are the four
neighbour directions in `DELTAS` order (up, right, down, left). What a *value*
means changed between the rungs and is listed in `LAYOUTS`: rung 1 encodes
"blocked" as 1, rung 2 as 0. Getting that backwards silently inverts finding 1,
which is why the layout is looked up rather than assumed.
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

# Feature layouts this script understands, keyed by the first four radices.
#   blocked / clear -- digit values for an impassable and for a plain free tile
#   lethal          -- digit value for "free, but the blast lands here" (rung 2)
#   danger_digit    -- index of the own-tile grace-period digit, if the layout
#                      has one. Enables the "in danger and not leaving" check.
LAYOUTS = {
    (2, 2, 2, 2): dict(name="rung-1 wall bits", blocked=1, clear=0,
                       lethal=None, danger_digit=None),
    (4, 4, 4, 4): dict(name="rung-2 neighbour states", blocked=0, clear=3,
                       lethal=1, danger_digit=4),
}

DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


def decode(index: int, sizes: tuple[int, ...]) -> tuple[int, ...]:
    """Inverse of the agent's `encode` -- mixed radix, most significant first."""

    digits = []
    for size in reversed(sizes):
        digits.append(index % size)
        index //= size
    return tuple(reversed(digits))


def build_arena(rng, density: float) -> np.ndarray:
    """The crate/wall half of `environment.py:build_arena`.

    Replicated rather than driven, because only the layout matters here and this
    keeps the script free of the world, the agents and the 0.5 s step budget.
    """

    arena = np.zeros((s.COLS, s.ROWS), int)
    arena[rng.random((s.COLS, s.ROWS)) < density] = 1
    arena[:1, :] = arena[-1:, :] = -1
    arena[:, :1] = arena[:, -1:] = -1
    for x in range(s.COLS):
        for y in range(s.ROWS):
            if (x + 1) * (y + 1) % 2 == 1:
                arena[x, y] = -1

    # Start positions and their four neighbours are cleared of crates. This is
    # what gives every corner exactly two blocked neighbours, and it is why
    # E09's four poisoned rows were reachable on step 1 of every round.
    for (x, y) in start_positions():
        for (xx, yy) in [(x, y), (x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]:
            if arena[xx, yy] == 1:
                arena[xx, yy] = 0
    return arena


def visible_coins(rng, arena: np.ndarray, n_coins: int) -> list[tuple[int, int]]:
    """Coins collectable on step 1.

    `environment.py:377-386` fills coin positions from shuffled *crate* tiles
    first and only falls back to free tiles. A coin under a crate is not
    collectable, so on any scenario with more crates than coins the agent's
    coin list is **empty** at step 1 -- the fact E09 and E10 both turn on.
    """

    n_crates = int((arena == 1).sum())
    n_visible = max(0, n_coins - n_crates)
    if n_visible == 0:
        return []
    free = np.argwhere(arena == 0)
    picked = rng.choice(len(free), size=min(n_visible, len(free)), replace=False)
    return [tuple(int(v) for v in free[i]) for i in np.atleast_1d(picked)]


def wall_pattern_frequency(scenario: str, n_arenas: int = 300,
                           seed: int = 20260731):
    """Share of free tiles carrying each of the 16 blocked/free neighbour
    patterns, plus the mean crate count and the sampled arenas themselves.

    Returned index is the blocked-bit tuple read as a binary number, U R D L.
    The crate count decides whether coins are visible at step 1, so it is
    reported alongside.
    """

    rng = np.random.default_rng(seed)
    density = s.SCENARIOS[scenario]["CRATE_DENSITY"]
    counts = np.zeros(16)
    crates = 0
    arenas = []

    for _ in range(n_arenas):
        arena = build_arena(rng, density)
        arenas.append((arena, visible_coins(rng, arena,
                                            s.SCENARIOS[scenario]["COIN_COUNT"])))
        crates += int((arena == 1).sum())
        for x in range(1, s.COLS - 1):
            for y in range(1, s.ROWS - 1):
                if arena[x, y] != 0:
                    continue
                bits = [int(arena[x + dx, y + dy] != 0) for dx, dy in DELTAS]
                counts[bits_to_code(bits)] += 1

    return counts / counts.sum(), crates / n_arenas, arenas


def start_positions() -> list[tuple[int, int]]:
    return [(1, 1), (1, s.ROWS - 2), (s.COLS - 2, 1), (s.COLS - 2, s.ROWS - 2)]


def bits_to_code(bits) -> int:
    return (bits[0] << 3) | (bits[1] << 2) | (bits[2] << 1) | bits[3]


def blocked_bits(digits, layout) -> list[int]:
    """The four neighbour digits reduced to plain blocked/free bits.

    Every layout value other than `blocked` describes a *free* tile -- on rung 2
    a lethal or blast-covered neighbour is somewhere the agent can walk, just
    somewhere it should not. So the arena statistics, which know nothing about
    bombs, stay comparable across rungs.
    """

    return [int(d == layout["blocked"]) for d in digits[:4]]


def spawn_state(arena: np.ndarray, pos, coins) -> dict:
    """The game state as the agent sees it on step 1 of a round."""

    return {
        "round": 1, "step": 1, "field": arena,
        "self": ("probe", 0, True, pos), "others": [], "bombs": [],
        "coins": list(coins), "explosion_map": np.zeros(arena.shape),
        "user_input": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", required=True,
                        help="folder under agent_code/, e.g. benedict_task2")
    parser.add_argument("--scenario", default="classic", choices=sorted(s.SCENARIOS),
                        help="scenario the table will be deployed on (default: classic)")
    parser.add_argument("--top", type=int, default=12,
                        help="how many rows to list per finding (default: 12)")
    parser.add_argument("--arenas", type=int, default=300,
                        help="arenas to sample for the spawn probe (default: 300)")
    args = parser.parse_args()

    cb = importlib.import_module(f"agent_code.{args.agent}.callbacks")
    sizes = tuple(cb.FEATURE_SIZES)
    actions = list(cb.ACTIONS)

    layout = LAYOUTS.get(sizes[:4])
    if layout is None:
        print(f"FEATURE_SIZES={sizes} starts with {sizes[:4]}, which is not a "
              f"layout this script knows ({sorted(LAYOUTS)}). Teach it the new "
              "meaning of the neighbour digits rather than guessing -- getting "
              "'blocked' backwards inverts finding 1 silently.", file=sys.stderr)
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

    freq, mean_crates, arenas = wall_pattern_frequency(args.scenario, args.arenas)
    coins_hidden = mean_crates >= s.SCENARIOS[args.scenario]["COIN_COUNT"]

    print(f"table   : {cb.MODEL_FILE}")
    print(f"agent   : {args.agent}   scenario: {args.scenario}")
    print(f"layout  : {layout['name']}, FEATURE_SIZES={sizes} "
          f"-> {n_states} rows x {len(actions)} actions")
    print()

    trained = np.array([not np.allclose(q[i], 0) for i in range(n_states)])
    print(f"coverage: {trained.sum()} of {n_states} rows carry a non-zero value "
          f"({trained.sum() / n_states:.1%})")
    print("          A row's size is not its cost: only a few hundred rows carry")
    print("          the traffic, so low coverage is expected, not a fault.")
    print()

    # --- Finding 1: argmax into a blocked neighbour (the E09 failure) --------
    # Only trained rows: on an all-zero row np.argmax returns index 0, but `act`
    # tie-breaks uniformly, so the deterministic-argmax argument does not apply.
    # Sealed-in rows are excluded throughout: with all four neighbours blocked
    # the agent has no legal move, so its argmax says nothing about the policy.
    # They are common (E09: 36 % of free tiles on `classic` have four blocked
    # neighbours) and the agent does reach them -- its own dropped bomb turns a
    # neighbour into a blocked tile -- so leaving them in buries the real
    # findings under rows where nothing could have been done.
    sealed = [i for i in range(n_states) if trained[i]
              and all(d == layout["blocked"] for d in decode(i, sizes)[:4])]

    blocked_argmax = []
    for i in range(n_states):
        if not trained[i] or i in set(sealed):
            continue
        digits = decode(i, sizes)
        best = int(np.argmax(q[i]))
        if best < 4 and digits[best] == layout["blocked"]:
            blocked_argmax.append((i, digits, best))

    print(f"[1] argmax points at a blocked neighbour : {len(blocked_argmax)} "
          f"of {trained.sum() - len(sealed)} trained rows with a legal move "
          f"({len(sealed)} sealed-in rows excluded)")
    print("    An invalid action leaves the state unchanged, so such a row is")
    print("    absorbing whenever the rest of the state is static (no live bomb).")
    reachable_traps = [i for i, d, _ in blocked_argmax
                       if freq[bits_to_code(blocked_bits(d, layout))] > NEGLIGIBLE]
    if blocked_argmax:
        print(f"    of which on a wall pattern common on {args.scenario}: "
              f"{len(reachable_traps)}")
        for i, digits, best in blocked_argmax[:args.top]:
            share = freq[bits_to_code(blocked_bits(digits, layout))]
            print(f"      row {i:5d}  {digits}  argmax={actions[best]:<5s} "
                  f"pattern<={share:6.2%}")
        if len(blocked_argmax) > args.top:
            print(f"      ... and {len(blocked_argmax) - args.top} more")
    print()

    # --- Finding 2: the rows that decide step 1 of every round ---------------
    # E09: the agent never left its spawn tile, so a handful of rows alone
    # decided 300 of 300 rounds. Rather than guessing which row a corner maps to
    # -- which stopped being possible once the layout gained digits that depend
    # on crates and coins -- the state is synthesised and handed to the agent's
    # own feature function.
    print(f"[2] spawn rows, probed on {len(arenas)} generated {args.scenario} arenas")
    if coins_hidden:
        print(f"    {mean_crates:.0f} crates vs {s.SCENARIOS[args.scenario]['COIN_COUNT']} "
              "coins: every coin starts under a crate, so no coin is visible on")
        print("    step 1. A frozen row here ends 300 rounds out of 300.")
    else:
        print(f"    {mean_crates:.0f} crates vs {s.SCENARIOS[args.scenario]['COIN_COUNT']} "
              "coins: coins are visible from step 1.")

    # Counted per spawn, not per row: a spawn on top of a coin looks identical
    # in feature space to one with no coin in sight, but it is not absorbing --
    # the coin is collected at the end of the step, so the state changes even if
    # the agent does nothing. `environment.py:379-384` does not exclude the
    # start positions from coin placement, so on `coin-heaven` this is 28 % of
    # spawns, and treating them as frozen condemns a table that scores 50/50.
    spawn_rows = {}
    for arena, coins in arenas:
        on_coin = set(coins)
        for pos in start_positions():
            row = cb.state_to_features(spawn_state(arena, pos, coins))
            entry = spawn_rows.setdefault(row, [0, 0])
            entry[0] += 1
            entry[1] += int(pos in on_coin)
    n_spawns = sum(count for count, _ in spawn_rows.values())

    frozen_spawns = 0
    for row, (count, resolving) in sorted(spawn_rows.items(), key=lambda kv: -kv[1][0]):
        digits = decode(row, sizes)
        share = count / n_spawns
        if not trained[row]:
            print(f"      row {row:5d} {digits} {share:6.1%}  UNTRAINED -> uniform "
                  f"over {len(actions)} actions incl. BOMB")
            continue
        best = int(np.argmax(q[row]))
        into_wall = best < 4 and digits[best] == layout["blocked"]
        # With no opponents and no bomb on the board, nothing but the agent can
        # change the state -- so WAIT at spawn is just as absorbing as an
        # invalid move. E09 measured both: 0.727 frozen, 0.273 waiting.
        waits = actions[best] == "WAIT"
        stuck = (into_wall or waits) and count > resolving
        frozen_spawns += (count - resolving) if (into_wall or waits) else 0
        note = ("FROZEN (invalid, forever)" if into_wall else
                "WAITS FOREVER" if waits else "acts")
        if not stuck and (into_wall or waits):
            note += ", but every such spawn stands on a coin -> resolves"
        elif resolving:
            note += f" ({resolving / count:.0%} of these spawns stand on a coin)"
        print(f"      row {row:5d} {digits} {share:6.1%}  "
              f"argmax={actions[best]:<5s} -> {note}")
    print()

    # --- Finding 3: in danger and not getting out (rung 2 and later) ---------
    if layout["danger_digit"] is not None:
        fatal = (layout["blocked"], layout["lethal"])
        dying, escapable = [], 0
        for i in range(n_states):
            if not trained[i]:
                continue
            digits = decode(i, sizes)
            if digits[layout["danger_digit"]] == 0:
                continue                    # not in a blast: nothing to escape
            if all(d in fatal for d in digits[:4]):
                continue                    # no way out exists; not a policy fault
            escapable += 1
            best = int(np.argmax(q[i]))
            if actions[best] in ("WAIT", "BOMB"):
                dying.append((i, digits, best))
            elif best < 4 and digits[best] in fatal:
                dying.append((i, digits, best))

        print(f"[3] in a blast with a way out, and the argmax does not take it : "
              f"{len(dying)} of {escapable} trained rows")
        print("    The rung-2 analogue of finding 1: the row is not absorbing --")
        print("    the timer keeps moving -- but it ends the round all the same.")
        print("    Rows with no safe neighbour at all are excluded: there the")
        print("    mistake was made earlier, and finding 5 is where it shows.")
        for i, digits, best in dying[:args.top]:
            print(f"      row {i:5d}  {digits}  grace={digits[layout['danger_digit']]} "
                  f"argmax={actions[best]}")
        if len(dying) > args.top:
            print(f"      ... and {len(dying) - args.top} more")
        print()

    # --- Finding 4: ties, i.e. rows decided by the RNG ----------------------
    ties = [i for i in range(n_states)
            if trained[i] and (q[i] == q[i].max()).sum() > 1]
    print(f"[4] trained rows with a tied argmax      : {len(ties)}")
    print("    Decided by policy_rng, so the measured policy is not the argmax")
    print("    policy. Fine in small numbers; a symptom if the count is large.")
    for i in ties[:args.top]:
        tied = [actions[a] for a in np.flatnonzero(q[i] == q[i].max())]
        print(f"      row {i:5d}  {decode(i, sizes)}  {'/'.join(tied)}")
    print()

    # --- Finding 5: where BOMB wins -----------------------------------------
    bomb_rows = [i for i in range(n_states)
                 if trained[i] and actions[int(np.argmax(q[i]))] == "BOMB"]
    print(f"[5] trained rows preferring BOMB         : {len(bomb_rows)}")
    print("    Expected 0 on rung 1 and >0 on rung 2. Read against the rung:")
    print("    a rung-2 agent with 0 here cannot open a crate.")
    for i in bomb_rows[:args.top]:
        print(f"      row {i:5d}  {decode(i, sizes)}  q={np.round(q[i], 2)}")
    print()

    # --- Finding 6: value spread --------------------------------------------
    # A row whose actions are within noise of each other is one update away from
    # flipping its argmax. Maxi's E01/E02 diagnostic, kept because it caught a
    # broken policy that every aggregate number called healthy.
    spreads = np.array([np.ptp(q[i]) for i in range(n_states) if trained[i]])
    print(f"[6] value spread over trained rows       : "
          f"median {np.median(spreads):.3f}, min {spreads.min():.3f}, "
          f"max {spreads.max():.3f}")
    print(f"    rows with spread < 0.1 (argmax is noise): {int((spreads < 0.1).sum())}")
    print()

    if frozen_spawns:
        print(f"FAIL: {frozen_spawns / n_spawns:.1%} of spawns sit on an absorbing "
              "row -- those rounds are over at step 1.")
        return 1
    if reachable_traps:
        # Not fatal: a common wall pattern does not mean the *row* is common,
        # because the remaining digits are not weighted. On coin-heaven the
        # equivalent rows are only entered while standing on a coin, which
        # resolves the same step -- and that table scored 50/50.
        print(f"WARN: {len(reachable_traps)} rows have an argmax into a wall on a "
              f"wall pattern that is common on {args.scenario}. Check whether the "
              "remaining digits are reachable there.")
        return 0
    print("OK: no trained row has an argmax into a blocked neighbour, and no "
          "spawn row is absorbing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
