"""Tabular Q-learning agent for `classic` against three opponents.

The state is a single mixed-radix row index over eight digits, sizes in
`FEATURE_SIZES`:

    1-4  each neighbour: blocked / lethal this step / in a blast / clear
     5   moves of grace left on my own tile, 0 = safe
     6   BFS first step to the objective. While a bomb covers my tile this is
         the way out; otherwise the nearest coin, else the nearest crate,
         else the nearest opponent.
     7   a bomb here would open a crate or catch an opponent, and I have one
     8   two meanings, selected by digit 5:
           safe rows   how far digit 6's target is: 1 / 2 / 3-4 / 5+
           danger rows (x + y) % 4, see below

4^4 x 5 x 5 x 2 x 5 = 64 000 rows x 6 actions. The table is sparse in practice:
the greedy policy visits a few hundred rows per round, and most of the state
space cannot be reached at all.

Digit 8 in the danger rows: digit 6 holds the escape direction there, so there
is no target distance to encode. A constant digit would make two thirds of the
table unreachable, so `(x + y) % 4` is used instead. Its low bit matches the
wall lattice. Stone pillars sit at (even, even), so a free tile with `x + y` even
has both coordinates odd and is a crossing (four structural exits, a bomb there
clears twelve tiles), while `x + y` odd is a corridor (two exits, six tiles).
Digits 1-4 almost contain this information, but a blocked neighbour does not
distinguish wall from crate, so a corridor's two permanent walls look the same
as two crates.

The learned values depend strongly on this digit, and the direction depends on
the grace left. With one move of grace a crossing is worth more, because exits
matter. With two or more it is worth less, because the blast to get out of
covers twice as many tiles. Encoding it gives +0.20 score against a matched
control. The derivation and the arms that rule out the alternatives are in
`experiments/benedict_task4.md`.

This file runs in the tournament, so it imports only numpy and `settings`, uses
paths relative to `__file__`, and does not use `tools/`.
"""

import os

import numpy as np

import settings as s    # BOMB_POWER / BOMB_TIMER

# Training only: without a suffix, parallel training runs would all write the
# same file. When the variable is unset (normal games and the tournament) the
# model is "q_table.npy" next to this file. The path is relative to this file.
#
# When it is set, the table is stored in checkpoints/<agent>/ instead. The
# submission is a zip of this folder, so per-run tables must not end up next to
# callbacks.py. The tournament never sets the variable, so it always takes the
# first branch.
_SUFFIX = os.environ.get("BM_MODEL_SUFFIX", "")
_AGENT_DIR = os.path.dirname(__file__)
MODEL_FILE = os.path.join(_AGENT_DIR, "q_table.npy") if not _SUFFIX else os.path.join(
    _AGENT_DIR, os.pardir, os.pardir, "checkpoints",
    os.path.basename(_AGENT_DIR), f"q_table{_SUFFIX}.npy",
)

# How close two actions must be to count as tied in `act`. We use exact
# equality: the rows that can absorb a collapsed policy have margins of 1e-4 to
# 1e-2, never 0, so a wider band would change behaviour without preventing them.
TIE_TOL = 0.0

# E51: the certain-death move filter. It is on by default, since it is part of
# the shipped policy and the tournament sets no environment variables.
# `BM_DEATH_FILTER=0` reproduces the pre-E51 agent and is used for the control
# arm of any comparison against it.
#
#   unset / "1"   veto moves after which no continuation survives the bombs
#                 already on the board (shipped)
#   "step"        veto moves that are lethal at the end of this step only.
#                 Shallow arm: digits 1-4 already encode this as NB_LETHAL, and
#                 it gives a third of the full effect (E51 P3).
#   "0"           off, identical to the pre-E51 policy
#
# Worth +0.099 [+0.067, +0.132] score against 3 x binary_v6 and
# +0.121 [+0.007, +0.236] against 3 x rule_based, n = 4000 each, on the frozen
# table with no retraining. It changes the decision on only 0.09 % of steps (the
# table already picks a non-vetoed action 99.91 % of the time). The gain comes
# from `killed_by` (-0.022); the change in suicides (-0.008) is not demonstrated.
# The agent survives longer, so it bombs more and collects more coins and kills:
# +0.043 coins + 5 x 0.011 kills = +0.098 of the +0.099.
#
# BOMB is never vetoed, in any mode. E46 gated bomb placement on escape slack;
# this cut suicides by -0.131 as intended but cost -0.283 score. The zero-slack
# bombs are both the most lethal and the most productive, because a bomb in a
# dense pocket has a small blast and a tight escape for the same geometric
# reason. This filter leaves bomb placement alone and corrects the escape
# instead. Keeping BOMB out of the mask is the only difference between the two
# approaches, so it must stay that way.
#
# Only bombs and explosions visible in the current state are modelled. E43
# traced every death back to the last step at which some action still survived:
# enemy bombs placed after we committed account for 2.3 % of deaths. Assuming
# that opponents may bomb would gain at most 2.3 % and make every step more
# conservative.
DEATH_FILTER_OFF = 0
DEATH_FILTER_STEP = 1
DEATH_FILTER_FULL = 2

_FILTER_MODES = {"": DEATH_FILTER_OFF, "0": DEATH_FILTER_OFF,
                 "step": DEATH_FILTER_STEP, "1": DEATH_FILTER_FULL}
_filter_env = os.environ.get("BM_DEATH_FILTER", "1").strip().lower()
if _filter_env not in _FILTER_MODES:
    # Raise instead of silently playing a different policy than the label says
    # (the E18 mistake, which cost ten evaluations). The tournament sets
    # nothing, so this cannot happen there.
    raise ValueError(
        f"BM_DEATH_FILTER={_filter_env!r} is not one of {sorted(_FILTER_MODES)}"
    )
DEATH_FILTER = _FILTER_MODES[_filter_env]


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# (dx, dy) for UP, RIGHT, DOWN, LEFT in image coords, y grows downwards
DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
# The actions the filter may veto: everything except BOMB. `veto.all()` looks
# like the right test for "no action survives", but it is wrong. BOMB is never
# vetoed, so `veto.all()` is always False, the filter is never skipped, and in a
# hopeless position BOMB is the only finite entry left in the row. The filter
# then forces a bomb: a suicide if the agent has one, otherwise an
# INVALID_ACTION that stands still and dies. The first run of E51 showed this
# (suicides +0.019, invalid +1.667).
FILTERABLE = np.array([a != ACTIONS.index('BOMB') for a in range(len(ACTIONS))])

# Digits 1-4, one per direction. Ordered so a larger value is never a worse tile
# to step onto, which makes a printed row readable without decoding it.
NB_BLOCKED = 0      # wall, crate, bomb or other agent: invalid move
NB_LETHAL = 1       # free, but blast lands here at the end of the step
NB_IN_BLAST = 2     # free and survivable this step, but a live bomb covers it
NB_CLEAR = 3        # free and outside every blast

# Digit 6: 0 = nothing reachable, 1..4 = ACTIONS[0..3].
NO_TARGET = 0

# 'danger' entry for a tile no bomb reaches.
SAFE = s.BOMB_TIMER + 1

# 4 neighbour states + steps of grace on my own tile + target direction
# + a bomb here would pay off + how far the target is
FEATURE_SIZES = (4, 4, 4, 4, 5, 5, 2, 5)
N_STATES = int(np.prod(FEATURE_SIZES))

POLICY_SEED = 20260731


def blast_coords(x: int, y: int, field: np.ndarray) -> list[tuple[int, int]]:
    """Tiles a bomb at (x, y) covers. Mirrors `items.py:Bomb.get_blast_coords`.

    Stone walls stop the blast, crates do not (`items.py:56` only breaks on -1),
    which is why one bomb in a dense corridor clears several crates.
    The blast does not turn corners. No bounds check is needed: the arena is
    walled all round, so the -1 test always fires before an index goes negative.
    """

    coords = [(x, y)]
    for dx, dy in DELTAS:
        for i in range(1, s.BOMB_POWER + 1):
            nx, ny = x + dx * i, y + dy * i
            if field[nx, ny] == -1:
                break
            coords.append((nx, ny))
    return coords


def danger_map(game_state: dict) -> np.ndarray:
    """Steps of grace per tile: 0 = deadly at the end of this step, SAFE = free.

    The timing follows `environment.py:166-173`, which runs the agents first and
    then counts bombs down and evaluates explosions. A bomb the agent sees at
    timer `t` therefore kills at the end of step `now + t`, and a tile with
    `explosion_map > 0` is still burning when this step is evaluated. Both are
    expressed in the same unit, so we can take the minimum over them.
    """

    field = game_state['field']
    danger = np.full(field.shape, SAFE, dtype=np.int8)
    danger[game_state['explosion_map'] > 0] = 0

    for (bx, by), timer in game_state['bombs']:
        for (cx, cy) in blast_coords(bx, by, field):
            if timer < danger[cx, cy]:
                danger[cx, cy] = timer
    return danger


def neighbour_status(x: int, y: int, field: np.ndarray, danger: np.ndarray,
                     occupied: set) -> tuple[int, ...]:
    """Digits 1-4: what happens if I step in each direction."""

    status = []
    for dx, dy in DELTAS:
        nx, ny = x + dx, y + dy
        if field[nx, ny] != 0 or (nx, ny) in occupied:
            status.append(NB_BLOCKED)
        elif danger[nx, ny] == 0:
            status.append(NB_LETHAL)
        elif danger[nx, ny] < SAFE:
            status.append(NB_IN_BLAST)
        else:
            status.append(NB_CLEAR)
    return tuple(status)


def bomb_hits_crate(x: int, y: int, field: np.ndarray,
                    others: list | None = None) -> bool:
    """Would a bomb dropped here destroy something worth destroying?

    An opponent standing in the blast counts as well as a crate; the digit
    itself is still one bit.
    """

    blast = blast_coords(x, y, field)
    if any(field[cx, cy] == 1 for cx, cy in blast):
        return True
    if others:
        return any(pos in blast for pos in others)
    return False


def bfs_first_step(x: int, y: int, field: np.ndarray, is_goal) -> tuple[int, int]:
    """First move of a shortest path to a tile satisfying `is_goal`, and its length.

    Breadth-first over free tiles. Goal tiles are tested but never expanded, so
    the goal itself may be impassable: a crate is a valid destination even
    though the agent cannot stand on it.

    Returns `(direction, distance)`, or `(NO_TARGET, 0)` when nothing is
    reachable. E20 uses the distance as a separate digit. Taking it from the
    same traversal guarantees that both digits describe the same objective.
    """

    queue = [((x, y), None, 0)]
    visited = {(x, y)}
    head = 0
    width, height = field.shape

    order = list(enumerate(DELTAS))

    while head < len(queue):
        (cx, cy), first, depth = queue[head]
        head += 1

        for action_idx, (dx, dy) in order:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if (nx, ny) in visited:
                continue
            # Each queue entry carries the first step that led to it, so the
            # direction falls out of the search without reconstructing a path.
            step = action_idx if first is None else first
            if is_goal((nx, ny)):
                return step + 1, depth + 1      # +1: 0 is reserved for NO_TARGET
            if field[nx, ny] != 0:
                continue            # wall or crate: testable, not passable
            visited.add((nx, ny))
            queue.append(((nx, ny), step, depth + 1))

    return NO_TARGET, 0


def target_direction(x: int, y: int, field: np.ndarray, coins: list,
                     others: list | None = None) -> tuple[int, int]:
    """First step of a shortest path to whatever the agent is currently after.

    Coins while one is reachable, otherwise the nearest crate. The crate itself
    is the goal, not a tile next to it. E10 measured that on a fresh `classic`
    arena 99.7 % of free tiles are already within blast range of some crate, so
    the older rule ("head for a tile a bomb could hit a crate from") was
    satisfied almost anywhere and this digit was 0 almost everywhere. Without a
    gradient in the safe part of the state, two mirror-image rows pointed at
    each other and the agent oscillated between two tiles in 20 of 20 rounds.

    A visible but unreachable coin falls through to the crate branch instead of
    returning NO_TARGET, since a coin enclosed by crates means the agent should
    bomb its way there.
    """

    coin_set = set(coins)
    if (x, y) in coin_set:
        return NO_TARGET, 0            # standing on it; collected this step

    def is_coin(pos: tuple[int, int]) -> bool:
        return pos in coin_set

    def is_crate(pos: tuple[int, int]) -> bool:
        return field[pos] == 1

    if coin_set:
        step, dist = bfs_first_step(x, y, field, is_coin)
        if step != NO_TARGET:
            return step, dist

    step, dist = bfs_first_step(x, y, field, is_crate)
    if step != NO_TARGET or not others:
        return step, dist

    # No crates left, so the objective becomes the nearest opponent. Four agents
    # clear all 122 crates by ~step 140, and without this branch the agent would
    # have no objective for the rest of the round. The table's choice in that
    # row is an invalid BOMB, at -1 each time. As in the crate branch, the goal
    # tile is tested but never expanded, so an occupied tile is a valid
    # destination even though the agent cannot stand on it.
    other_set = set(others)
    return bfs_first_step(x, y, field, lambda pos: pos in other_set)


def escape_direction(x: int, y: int, field: np.ndarray, danger: np.ndarray,
                     occupied: set) -> int:
    """First step of a shortest path out of every blast, or NO_TARGET if none.

    Unlike `target_direction` this search is time-aware: a tile that is safe
    now can be lethal by the time the agent gets there, and a tile inside a
    blast can be crossed if the agent has left it before the timer runs out.
    Both cases are common when escaping from one's own bomb.
    """

    queue = [((x, y), None, 0)]
    visited = {(x, y)}
    head = 0

    while head < len(queue):
        (cx, cy), first, depth = queue[head]
        head += 1

        # Every bomb currently on the board has detonated by then, so walking
        # further cannot help. This bounds the search at ~60 tiles.
        if depth >= SAFE:
            continue

        for action_idx, (dx, dy) in enumerate(DELTAS):
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited:
                continue
            if field[nx, ny] != 0 or (nx, ny) in occupied:
                continue

            # `depth` moves have been made to reach (cx, cy), so stepping here
            # puts the agent on (nx, ny) at the end of step `depth`. It dies if
            # the blast reaches this tile at or before then.
            if danger[nx, ny] <= depth:
                continue

            step = action_idx if first is None else first
            if danger[nx, ny] >= SAFE:
                return step + 1     # +1 because 0 is reserved for NO_TARGET
            visited.add((nx, ny))
            queue.append(((nx, ny), step, depth + 1))

    return NO_TARGET

# Digit 8 (E20): 0 = not applicable (no target, or the danger branch), else the
# bucket from `distance_bucket`.
DIST_NONE = 0


def distance_bucket(distance: int) -> int:
    """How far away digit 6's target is, in four buckets.

    The boundaries come from measurement. Over 40 greedy rounds of the previous
    agent the target distance is 1 in 26.4 % of steps, 2 in 23.2 %, 3-4 in
    28.8 % and 5+ in 21.6 %. Splitting there reduces the within-row spread of
    the distance from 1.28 tiles to 0.48. A uniform {1,2,3,4+} split uses the
    same number of rows and only reaches 0.64, because the distance is not
    concentrated near 1, so a wide top bucket loses little.
    """

    if distance <= 0:
        return DIST_NONE
    if distance <= 2:
        return distance         # 1 and 2 are each their own bucket
    if distance <= 4:
        return 3
    return 4


def lattice_class(x: int, y: int) -> int:
    """Digit 8 in the danger rows: `(x + y) % 4`.

    The low bit matches the wall lattice. Pillars sit at (even, even), so
    `x + y` even means both coordinates are odd, which is a crossing (four
    structural exits, twelve tiles cleared by a bomb), and odd means a corridor
    (two exits, six tiles). The high bit is a diagonal stripe that carries
    position but no structure. We keep it because the four-way split did better
    than the lattice bit alone, and dropping it has not been tested.
    """

    return (x + y) % 4


def state_to_features(game_state: dict) -> int:
    """Map a game state onto a row index of the Q-table."""

    if game_state is None:      # every caller checks this; raise if one stops
        raise ValueError("state_to_features called without a game state")

    field = game_state['field']     # indexing is field[x, y]
    x, y = game_state['self'][3]
    have_bomb = game_state['self'][2]

    danger = danger_map(game_state)

    # environment.py:121-126: bombs and other agents block a move like walls do.
    # They go into "can I go there", while `danger` answers "will I die there".
    occupied = {pos for pos, _ in game_state['bombs']}
    occupied.update(other[3] for other in game_state['others'])

    # Digit 5, in moves rather than timer units: 0 = safe, otherwise the number
    # of moves left including this one. A bomb seen at t leaves t+1 moves.
    own_danger = 0 if danger[x, y] >= SAFE else int(danger[x, y]) + 1

    # Digit 7 includes `bomb_possible`. Otherwise the agent can sit in a "crate
    # in range" row with no bomb left and pick BOMB, which gives INVALID_ACTION.
    # An invalid action leaves the state unchanged, so the policy never leaves
    # that row.
    others = [o[3] for o in game_state['others']]
    bomb_useful = int(have_bomb and bomb_hits_crate(x, y, field, others))

    # Digit 6 is the direction that matters right now. While a bomb covers the
    # agent's tile that is the way out, and nothing else needs encoding: a coin
    # four tiles away is irrelevant if the agent dies in three.
    if own_danger:
        target = escape_direction(x, y, field, danger, occupied)
        # Digit 6 already holds the escape direction here, so digit 8 has no
        # target distance to hold. This branch used to set it to a constant,
        # which left two thirds of the table unreachable and gave the rows where
        # almost all deaths happen no structural information.
        target_dist = lattice_class(x, y)
    else:
        target, distance = target_direction(x, y, field, game_state['coins'], others)
        target_dist = distance_bucket(distance)

    features = neighbour_status(x, y, field, danger, occupied) + (
        own_danger,
        target,
        bomb_useful,
        target_dist,
    )
    return encode(features)


def encode(features: tuple[int, ...]) -> int:
    """Flatten a feature tuple to a single Q-table row index (mixed radix).

    Each feature is one digit whose base is its entry in FEATURE_SIZES, so
    distinct tuples always map to distinct rows in [0, N_STATES). Adding a
    feature only requires extending FEATURE_SIZES.

    Storage is hidden behind this function: if the product of the radices gets
    too large for a dense array, switching to a dict is a two-line change.
    """

    idx = 0
    for value, size in zip(features, FEATURE_SIZES):
        idx = idx * size + value
    return idx


def survives(x: int, y: int, arrival: int, field: np.ndarray,
             danger: np.ndarray, occupied: set, lookahead: bool) -> bool:
    """Can the agent be alive on (x, y) at the end of step `arrival`, and stay alive?

    Same time model and same approximation as `escape_direction`: a node's
    `arrival` is the step at the end of which the agent stands on it, and a tile
    whose blast lands at or before then is fatal. This prunes a blast tile for
    every step from its timer onwards, which also covers the step in which the
    explosion lingers. `EXPLOSION_TIMER = 2` means one lethal step plus one
    more, because `environment.py:200-210` counts the explosion down only after
    the agents have moved, so a tile burns at the end of two consecutive steps.
    The search can over-prune by one step but never under-prunes, which is the
    safe direction for a death filter.

    `lookahead=False` only checks whether the agent survives this step (the
    shallow arm).
    """

    if danger[x, y] <= arrival:
        return False                    # the blast is here at or before arrival
    if not lookahead or danger[x, y] >= SAFE:
        return True                     # no blast reaches this tile at all

    # Standing still never helps here: `danger` only counts down, so a tile that
    # is doomed at step t stays doomed and waiting on it only uses up the grace.
    # The search therefore moves every step and needs no WAIT edge.
    queue = [((x, y), arrival)]
    visited = {(x, y)}
    head = 0

    while head < len(queue):
        (cx, cy), step = queue[head]
        head += 1

        # Every bomb on the board has gone off by then, so walking further
        # cannot help. Same bound as in `escape_direction`, ~60 tiles.
        if step >= SAFE:
            continue

        for dx, dy in DELTAS:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited:
                continue
            if field[nx, ny] != 0 or (nx, ny) in occupied:
                continue
            if danger[nx, ny] <= step + 1:
                continue
            if danger[nx, ny] >= SAFE:
                return True
            visited.add((nx, ny))
            queue.append(((nx, ny), step + 1))

    return False


def death_filter_mask(game_state: dict) -> np.ndarray:
    """Which actions are certain death? One bool per entry of `ACTIONS`.

    `BOMB` is never marked; the `DEATH_FILTER` comment explains why.
    """

    veto = np.zeros(len(ACTIONS), dtype=bool)

    # By far the most common case, which keeps the average cost low: with no
    # bomb and no fire on the board nothing can be vetoed.
    if not game_state['bombs'] and not game_state['explosion_map'].any():
        return veto

    field = game_state['field']
    x, y = game_state['self'][3]
    danger = danger_map(game_state)
    occupied = {pos for pos, _ in game_state['bombs']}
    occupied.update(other[3] for other in game_state['others'])
    lookahead = DEATH_FILTER == DEATH_FILTER_FULL

    for action_idx, (dx, dy) in enumerate(DELTAS):
        nx, ny = x + dx, y + dy
        if field[nx, ny] != 0 or (nx, ny) in occupied:
            # `environment.py:121-126`: a move into a wall, crate, bomb or agent
            # leaves the agent where it is, so its survival is the same as for
            # WAIT. Checking the blocked target tile would veto the wrong action.
            nx, ny = x, y
        veto[action_idx] = not survives(nx, ny, 0, field, danger, occupied, lookahead)

    veto[ACTIONS.index('WAIT')] = not survives(x, y, 0, field, danger, occupied,
                                               lookahead)
    return veto


def setup(self):
    """Called once before a set of games to initialize data structures."""

    self.policy_rng = np.random.default_rng(POLICY_SEED)

    if self.train:
        self.logger.info("Starting from an empty Q-table")
        self.q = np.zeros((N_STATES, len(ACTIONS)))
    elif not os.path.isfile(MODEL_FILE):
        # E18: evaluating a missing table has to raise an error. Otherwise the
        # agent plays the uniform-random policy of an all-zero table, which is
        # what ten evaluations of a missing checkpoint measured (2.82 crates,
        # 1.000 suicides, identical across five "seeds"). In the tournament the
        # table is shipped next to this file, so this only triggers if the
        # submission is broken, and the submission test run should then fail
        # instead of playing a random agent.
        raise FileNotFoundError(f"No Q-table at {MODEL_FILE} and not training.")
    else:
        self.logger.info("Loading Q-table from disk.")
        self.q = np.load(MODEL_FILE)

        # A table from an older FEATURE_SIZES would otherwise only fail later
        # in act(), as an IndexError in the middle of a round.
        expected = (N_STATES, len(ACTIONS))
        if self.q.shape != expected:
            raise ValueError(
                f"Q-table on disk has shape {self.q.shape}, expected {expected}. "
                "It was trained with a different feature layout -- retrain."
            )


def act(self, game_state: dict) -> str:
    """Called each game step to determine the agent's next action."""

    state = state_to_features(game_state)

    # E51. If every action the filter may veto is fatal, the agent dies whatever
    # it does (BOMB leaves it standing where it is, so it dies too). The filter
    # is then skipped and the unfiltered row decides, as in the agent without
    # the filter. The test is over FILTERABLE and not the whole mask; the
    # comment at FILTERABLE explains what the whole-mask version costs.
    veto = death_filter_mask(game_state) if DEATH_FILTER else None
    if veto is not None and veto[FILTERABLE].all():
        veto = None

    # self.eps and self.rng are set in train.py. Outside training the policy is
    # greedy and neither is used, so the agent also works when train.py is not
    # imported. E51 evaluates a frozen table at eps = 0 and never takes this
    # branch. The filter is still applied here, because a later training run
    # should only explore among actions that are not suicide, and keeping this
    # branch consistent with the greedy one avoids an unnoticed mismatch.
    if self.train and self.rng.random() < self.eps:
        legal = np.arange(len(ACTIONS)) if veto is None else np.flatnonzero(~veto)
        return ACTIONS[int(self.rng.choice(legal))]

    # Break ties at random instead of by action order. A converged table rarely
    # has ties, but an argmax that always picks the same action can turn a tie
    # into a loop: an invalid move leaves the state unchanged, so the agent
    # repeats it forever. This is a cheap safeguard and does not replace the
    # learning rate that prevents ties in the first place.
    #
    # TIE_TOL widens "tied" from exact float equality to a band. At the default
    # of 0.0 the behaviour is identical to the one every entry up to E22 was
    # measured with. The audit of E19-E22 found that the random tie-break never
    # triggers in practice: the rows that absorb a collapsed policy have
    # margins of 1e-4 to 1e-2, never 0.
    q_row = self.q[state]
    if veto is not None:
        # -inf rather than a large negative value, so a vetoed action can never
        # fall inside the TIE_TOL band and be picked by the tie-break below.
        q_row = np.where(veto, -np.inf, q_row)
    best = np.flatnonzero(q_row >= q_row.max() - TIE_TOL)
    return ACTIONS[int(best[0] if best.size == 1 else self.policy_rng.choice(best))]
