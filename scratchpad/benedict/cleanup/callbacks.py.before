"""Tabular Q-learning agent for `classic` against three opponents.

The state is a single mixed-radix row index over eight digits, sizes in
`FEATURE_SIZES`:

    1-4  each neighbour: blocked / lethal this step / in a blast / clear
     5   moves of grace left on my own tile, 0 = safe
     6   BFS first step to the objective. While a bomb covers my tile that is
         the way *out*; otherwise the nearest coin, else the nearest crate,
         else the nearest opponent.
     7   a bomb here would open a crate or catch an opponent, and I have one
     8   two meanings, selected by digit 5:
           safe rows   how far digit 6's target is: 1 / 2 / 3-4 / 5+
           danger rows (x + y) % 4 -- see below

4^4 x 5 x 5 x 2 x 5 = 64 000 rows x 6 actions, and deliberately sparse: the
greedy policy visits a few hundred rows per round, so the table is dense
storage over a state space that is mostly unreachable by construction.

**Digit 8 in the danger rows.** Digit 6 carries the escape direction there, so
there is no "target distance" to encode and the digit would otherwise sit at a
constant, making two thirds of the table unreachable. `(x + y) % 4` fills it,
and its low bit is exactly the wall lattice: stone pillars sit at (even, even),
so a free tile with `x + y` even has both coordinates odd and is a **crossing**
-- four structural exits, and a bomb dropped there clears twelve tiles -- while
`x + y` odd is a **corridor**, two exits and six tiles. Digits 1-4 nearly carry
this already but not quite, because a blocked neighbour merges wall with crate,
so a corridor's two permanent walls look exactly like two crates.

The learned table splits hard on it, and conditionally: a crossing is worth
*more* with one move of grace left, when what matters is having exits, and
*less* with two or more, when what matters is clearing a blast that covers
twice as many tiles. Encoding it is worth +0.20 score against a matched
control. Full derivation and the arms that rule out the alternatives:
`experiments/benedict_task4.md`.

Everything here runs in the tournament, so this file imports numpy and
`settings` only, uses paths relative to `__file__`, and never touches `tools/`.
"""

import os

import numpy as np

import settings as s    # BOMB_POWER / BOMB_TIMER

# Training-only escape hatch: parallel training runs would otherwise all write
# the same file. Unset -- every normal game, and the tournament -- this is
# exactly "q_table.npy" beside this file. Relative to this file, never absolute.
#
# When it IS set, the table lives in checkpoints/<agent>/ instead of here. The
# submission is a zip of this folder, and per-run tables sitting next to
# callbacks.py is a submission accident waiting to happen -- keeping them out by
# construction beats remembering to delete them. The tournament never sets the
# variable, so the branch below is not even taken there.
_SUFFIX = os.environ.get("BM_MODEL_SUFFIX", "")
_AGENT_DIR = os.path.dirname(__file__)
MODEL_FILE = os.path.join(_AGENT_DIR, "q_table.npy") if not _SUFFIX else os.path.join(
    _AGENT_DIR, os.pardir, os.pardir, "checkpoints",
    os.path.basename(_AGENT_DIR), f"q_table{_SUFFIX}.npy",
)

# How close two actions must be to count as tied in `act`. Exact equality: the
# rows that can absorb a collapsed policy sit at margins of 1e-4 to 1e-2, never
# at 0, so a wider band changes behaviour without protecting against anything.
TIE_TOL = 0.0

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

    step, dist = bfs_first_step(x, y, field, is_crate)
    if step != NO_TARGET or not others:
        return step, dist

    # The board is out of crates, so the objective becomes the nearest opponent.
    # Four agents strip all 122 crates by ~step 140, and without this branch the
    # agent has no objective at all for the rest of the round -- the table's
    # answer in that row is an invalid BOMB, at -1 a time. Same rule as the crate
    # branch: the goal tile is tested but never expanded, so an occupied tile is
    # a legal destination even though the agent cannot stand on it.
    other_set = set(others)
    return bfs_first_step(x, y, field, lambda pos: pos in other_set)


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


def lattice_class(x: int, y: int) -> int:
    """Digit 8 in the danger rows: `(x + y) % 4`.

    Its low bit is the wall lattice exactly -- pillars sit at (even, even), so
    `x + y` even means both coordinates are odd, which is a crossing (four
    structural exits, twelve tiles cleared by a bomb) and odd means a corridor
    (two exits, six tiles). The high bit is a diagonal stripe carrying position
    but no structure; it is kept because the four-way split measurably
    outperforms the lattice bit alone, and dropping it would be a change this
    project has not tested.
    """

    return (x + y) % 4


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

    # Digit 7 folds in `bomb_possible` deliberately. Without it the agent sits
    # in a "crate in range" row with no bomb left, picks BOMB, gets
    # INVALID_ACTION -- and an invalid action leaves the state unchanged, which
    # is an absorbing row the policy cannot leave.
    others = [o[3] for o in game_state['others']]
    bomb_useful = int(have_bomb and bomb_hits_crate(x, y, field, others))

    # Digit 6 is "the direction that matters right now". While a bomb covers the
    # agent's tile that is the way out, and nothing else is worth encoding --
    # a coin four tiles away is irrelevant if the agent is dead in three.
    if own_danger:
        target = escape_direction(x, y, field, danger, occupied)
        # Digit 6 already carries the escape direction here, so there is no
        # target distance for digit 8 to hold. Pinning it to a constant -- which
        # is what this branch used to do -- left two thirds of the table
        # unreachable and the rows where essentially every death happens
        # carrying no structural information at all.
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
