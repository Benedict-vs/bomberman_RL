"""Tabular Q-learning agent — task 2 (`classic`, no opponents).

Forked from `agent_code/tabular_q_task1/`, the agreed rung-1 baseline, and
rebuilt for rung 2 in E10. The rung-1 feature map transferred *nothing*: on
`classic` it scored 0.000 and never took a single step in 300 rounds, because
"no coin is visible" -- the normal condition on this rung, where all nine coins
start under crates -- was a state it had only ever met at the end of a won
round (`experiments/benedict.md` E09).

Carried over from rung 1: the dense mixed-radix table, random tie-breaking
among equal-value actions, the per-cell learning rate and the seeded RNGs.
What E10 changes:

- crates stop being collapsed into "wall". A neighbour is blocked, lethal
  this step, covered by a live bomb, or clear.
- the target digit falls back to the nearest crate-bombing position when no
  coin is visible, so it carries information for the whole round instead of
  being pinned at 0.
- the own tile carries a grace period, counted in moves rather than in bomb
  timer units.
- one bit for "a bomb dropped here would open a crate, and I have one to drop".

State: 4 neighbour states + grace + target direction + bomb payoff
-> 4^4 x 5 x 5 x 2 = 12 800 rows. Large but cheap to fill: a competent agent
visits 451 of them, and 146 cover 90 % of its steps.
"""

import os

import numpy as np

import settings as s    # BOMB_POWER / BOMB_TIMER

# Training-only escape hatch: parallel training runs would otherwise all write
# the same file. Unset -- every normal game, and the tournament -- this is
# exactly "q_table.npy". Relative to this file, never absolute.
MODEL_FILE = os.path.join(
    os.path.dirname(__file__),
    f"q_table{os.environ.get('BM_MODEL_SUFFIX', '')}.npy",
)

# E17 ablation switch, same environment-variable pattern as BM_MODEL_SUFFIX.
# Each arm pins one feature component to a constant, so the table keeps its
# 12 800 rows and only the information content changes -- rows collapse, they
# do not disappear. Unset -- every normal game, and the tournament -- the full
# map is active. The switch must be set for training AND for evaluating an
# ablated table: the shapes match either way, so a mismatch would not crash,
# it would silently measure a table through features it was never trained on.
#
#   escape       -- digit 6 no longer switches to the way out while in a blast
#   crate_target -- digit 6 goes silent when no coin is visible (the E11 map)
#   danger       -- digits 1-4 collapse to blocked/clear, digit 5 pinned at 0.
#                   Digit 5 = 0 also means the escape branch never fires, so
#                   this arm removes escape AS WELL -- the components nest.
#   bomb_digit   -- digit 7 pinned at 0
#   target_dist  -- E20: digit 8 pinned at 0. Same information as the pre-E20
#                   map in a table of the same 64 000 rows, so it prices the
#                   sample dilution on its own, with no new information at all.
ABLATE = os.environ.get("BM_ABLATE", "")
if ABLATE not in ("", "escape", "crate_target", "danger", "bomb_digit", "target_dist"):
    raise ValueError(
        f"BM_ABLATE={ABLATE!r} is not an ablation arm. A typo here would "
        "silently train the full agent under an arm's label -- fail instead."
    )

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# (dx, dy) for UP, RIGHT, DOWN, LEFT -- image coords, y grows downwards
DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

# Digits 1-4, one per direction. Ordered so a larger value is never a worse tile
# to step onto, which makes a printed row readable without decoding it.
NB_BLOCKED = 0      # wall, crate, bomb or other agent -- invalid move
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

# How close two actions must be to count as tied in `act`. 0.0 reproduces every
# measurement up to E22 exactly. Set as an environment switch rather than an
# edited constant so the default in the tournament is the value in this file.
TIE_TOL = float(os.environ.get("BM_TIE_TOL", 0.0))


def blast_coords(x: int, y: int, field: np.ndarray) -> list[tuple[int, int]]:
    """Tiles a bomb at (x, y) covers. Mirrors `items.py:Bomb.get_blast_coords`.

    Stone walls stop the blast, crates do **not** -- `items.py:56` breaks on -1
    only -- which is why one bomb in a dense corridor clears several crates.
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
    """Steps of grace per tile: 0 = deadly at the end of *this* step, SAFE = free.

    The timing comes straight out of `environment.py:166-173`, which runs the
    agents first and only then counts bombs down and evaluates explosions. So a
    bomb the agent sees at timer `t` kills at the end of step `now + t`, and a
    tile with `explosion_map > 0` is still burning when this step is evaluated.
    Both are therefore expressed in the same unit and can be minimised over.
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
        elif ABLATE == "danger":
            status.append(NB_CLEAR)     # E17: lethal and in-blast read as clear
        elif danger[nx, ny] == 0:
            status.append(NB_LETHAL)
        elif danger[nx, ny] < SAFE:
            status.append(NB_IN_BLAST)
        else:
            status.append(NB_CLEAR)
    return tuple(status)


def bomb_hits_crate(x: int, y: int, field: np.ndarray) -> bool:
    return any(field[cx, cy] == 1 for cx, cy in blast_coords(x, y, field))


def bfs_first_step(x: int, y: int, field: np.ndarray, is_goal) -> tuple[int, int]:
    """First move of a shortest path to a tile satisfying `is_goal`, and its length.

    Breadth-first over free tiles. Goal tiles are *tested but never expanded*,
    so the goal itself may be impassable -- a crate is a legitimate destination
    even though the agent cannot stand on it.

    Returns `(direction, distance)`, `(NO_TARGET, 0)` when nothing is reachable.
    E20 needs the distance as a digit of its own; taking it from this traversal
    rather than a second one keeps the two digits describing one objective by
    construction, so they cannot drift apart.
    """

    queue = [((x, y), None, 0)]
    visited = {(x, y)}
    head = 0
    width, height = field.shape

    while head < len(queue):
        (cx, cy), first, depth = queue[head]
        head += 1

        for action_idx, (dx, dy) in enumerate(DELTAS):
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


def target_direction(x: int, y: int, field: np.ndarray, coins: list) -> tuple[int, int]:
    """First step of a shortest path to whatever the agent is currently after.

    Coins while one is reachable, otherwise the nearest crate. The crate is the
    *goal*, not somewhere to stand: E10 measured that on a fresh `classic` arena
    99.7 % of free tiles are already within blast range of some crate, so the
    older rule -- "head for a tile a bomb could hit a crate from" -- was
    satisfied wherever the agent happened to be, and this digit read 0 almost
    everywhere. With no gradient in the safe part of the state, two mirror-image
    rows pointed at each other and the agent oscillated between two tiles in 20
    rounds out of 20.

    A visible but unreachable coin falls through to the crate branch instead of
    returning NO_TARGET: a coin sealed in a pocket of crates is a reason to go
    bombing, not a reason to have no objective at all.
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

    if ABLATE == "crate_target":
        return NO_TARGET, 0    # E17: the E11 behaviour -- no target without a coin

    return bfs_first_step(x, y, field, is_crate)


def escape_direction(x: int, y: int, field: np.ndarray, danger: np.ndarray,
                     occupied: set) -> int:
    """First step of a shortest path out of every blast, or NO_TARGET if none.

    Time-aware, which is what separates it from `target_direction`: a tile that
    is safe now can be lethal by the time the agent gets there, and a tile that
    is in a blast now can be crossed if the agent is past it before the timer
    runs out. Both cases occur constantly while escaping one's own bomb.
    """

    queue = [((x, y), None, 0)]
    visited = {(x, y)}
    head = 0

    while head < len(queue):
        (cx, cy), first, depth = queue[head]
        head += 1

        # Every bomb currently on the board has detonated by then, so a tile
        # that is still unsafe at this depth cannot be made safe by walking
        # further. Bounds the search at ~60 tiles.
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

    The boundaries are measured rather than guessed. Over 40 greedy rounds of
    the incumbent the target distance spends 26.4 % of its steps at 1, 23.2 % at
    2, 28.8 % at 3-4 and 21.6 % at 5+, and splitting there cuts the within-row
    spread of the distance from 1.28 tiles to 0.48. A uniform {1,2,3,4+} costs
    the same number of rows and only reaches 0.64 -- the distance is not
    concentrated near 1, which is what makes the wide top bucket the cheap one.
    """

    if distance <= 0:
        return DIST_NONE
    if distance <= 2:
        return distance         # 1 and 2 are each their own bucket
    if distance <= 4:
        return 3
    return 4


def state_to_features(game_state: dict) -> int:
    """Map a game state onto a row index of the Q-table."""

    if game_state is None:      # every caller guards; fail loudly if one stops
        raise ValueError("state_to_features called without a game state")

    field = game_state['field']     # indexing is field[x, y]
    x, y = game_state['self'][3]
    have_bomb = game_state['self'][2]

    danger = danger_map(game_state)

    # environment.py:121-126: bombs and other agents block a move exactly like
    # walls do. That belongs in "can I go there", not in "will I die there".
    occupied = {pos for pos, _ in game_state['bombs']}
    occupied.update(other[3] for other in game_state['others'])

    # Digit 5, in moves rather than in timer units: 0 = safe, otherwise how many
    # moves are left *including this one*. A bomb seen at t leaves t+1 moves.
    own_danger = 0 if danger[x, y] >= SAFE else int(danger[x, y]) + 1
    if ABLATE == "danger":
        own_danger = 0

    # Digit 7 folds in `bomb_possible` deliberately. Without it the agent sits
    # in a "crate in range" row with no bomb left, picks BOMB, gets
    # INVALID_ACTION -- and an invalid action leaves the state unchanged, which
    # is the absorbing-row failure of E09 in a new place.
    bomb_useful = int(have_bomb and bomb_hits_crate(x, y, field))
    if ABLATE == "bomb_digit":
        bomb_useful = 0

    # Digit 6 is "the direction that matters right now". While a bomb covers the
    # agent's tile that is the way out, and nothing else is worth encoding --
    # a coin four tiles away is irrelevant if the agent is dead in three.
    if own_danger and ABLATE != "escape":
        target = escape_direction(x, y, field, danger, occupied)
        # E20: digit 8 is the *target* distance, and there is no target here.
        # Digit 5 already carries the scarce resource while escaping.
        target_dist = DIST_NONE
    else:
        target, distance = target_direction(x, y, field, game_state['coins'])
        target_dist = distance_bucket(distance)

    if ABLATE == "target_dist":
        target_dist = DIST_NONE

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

    Storage is an implementation detail behind this function: swapping the
    dense array for a dict is a two-line change if the product of the radices
    ever stops being countable.
    """

    idx = 0
    for value, size in zip(features, FEATURE_SIZES):
        idx = idx * size + value
    return idx


def setup(self):
    """Called once before a set of games to initialize data structures."""

    self.policy_rng = np.random.default_rng(POLICY_SEED)

    if self.train:
        self.logger.info("Starting from an empty Q-table")
        self.q = np.zeros((N_STATES, len(ACTIONS)))
    elif not os.path.isfile(MODEL_FILE):
        # E18 post-mortem: evaluating a table that does not exist must fail, not
        # silently play the uniform-random policy of an all-zero table -- ten
        # evaluations of a missing checkpoint measured exactly that, at 2.82
        # crates and 1.000 suicides, identically across five "seeds". In the
        # tournament the table ships beside this file, so this can only fire
        # when something is genuinely broken -- and the submission pre-run
        # should say so loudly rather than play a random agent.
        raise FileNotFoundError(f"No Q-table at {MODEL_FILE} and not training.")
    else:
        self.logger.info("Loading Q-table from disk.")
        self.q = np.load(MODEL_FILE)

        # A table left over from an older FEATURE_SIZES would not fail here but
        # deep inside act(), as an IndexError in the middle of a round.
        expected = (N_STATES, len(ACTIONS))
        if self.q.shape != expected:
            raise ValueError(
                f"Q-table on disk has shape {self.q.shape}, expected {expected}. "
                "It was trained with a different feature layout -- retrain."
            )


def act(self, game_state: dict) -> str:
    """Called each game step to determine the agent's next action."""

    state = state_to_features(game_state)

    # self.eps and self.rng are set in train.py. Outside training the policy is
    # greedy, so the tournament never reaches either -- which is what keeps the
    # agent working when train.py is not imported at all.
    if self.train and self.rng.random() < self.eps:
        return ACTIONS[int(self.rng.integers(len(ACTIONS)))]

    # Break ties at random rather than by action order. A converged table rarely
    # ties, but an argmax that always resolves to the same action turns a
    # near-tie into an absorbing loop -- an invalid move leaves the state
    # unchanged, so the agent repeats it forever. Cheap insurance; it is not a
    # substitute for the learning rate that stops the ties happening.
    #
    # TIE_TOL widens "tied" from exact float equality to a band. At 0.0 -- the
    # default -- this is byte-for-byte the behaviour every entry up to E22 was
    # measured with. The audit of E19-E22 measured that the insurance above
    # never fires in practice: the rows that absorb a collapsed policy sit at
    # margins of 1e-4 to 1e-2, never at 0.
    q_row = self.q[state]
    best = np.flatnonzero(q_row >= q_row.max() - TIE_TOL)
    return ACTIONS[int(best[0] if best.size == 1 else self.policy_rng.choice(best))]
