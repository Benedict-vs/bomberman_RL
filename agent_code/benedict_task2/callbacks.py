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
# + a bomb here would pay off
FEATURE_SIZES = (4, 4, 4, 4, 5, 5, 2)
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


def bomb_hits_crate(x: int, y: int, field: np.ndarray) -> bool:
    return any(field[cx, cy] == 1 for cx, cy in blast_coords(x, y, field))


def target_direction(x: int, y: int, field: np.ndarray, coins: list) -> int:
    """First step of a shortest path to whatever the agent is currently after.

    Coins when any are visible; otherwise the nearest tile from which a bomb
    would open a crate. The fallback is E09's requirement 1: on `classic` all
    nine coins start under crates (`environment.py:377-386`), so a coin-only
    target leaves this digit pinned at 0 for essentially the whole round and
    the agent runs on the wall digits alone -- which is how the rung-1 model
    came to freeze on its spawn tile in 300 rounds out of 300.

    Crates are impassable here, as they are in the game. Cost is inverse to how
    much is left to do: with crates everywhere the search stops after a handful
    of nodes, and only floods the board once the arena is nearly cleared.
    """

    if coins:
        targets = set(coins)

        def is_target(pos: tuple[int, int]) -> bool:
            return pos in targets
    else:
        def is_target(pos: tuple[int, int]) -> bool:
            return bomb_hits_crate(pos[0], pos[1], field)

    if is_target((x, y)):
        return NO_TARGET        # standing on it: collect it, or bomb from here

    # Each queue entry carries the first step that led to it, so the direction
    # falls out of the search without reconstructing the path
    queue = [((x, y), None)]
    visited = {(x, y)}
    head = 0
    width, height = field.shape

    while head < len(queue):
        (cx, cy), first = queue[head]
        head += 1

        for action_idx, (dx, dy) in enumerate(DELTAS):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if (nx, ny) in visited or field[nx, ny] != 0:
                continue
            step = action_idx if first is None else first
            if is_target((nx, ny)):
                return step + 1     # +1 because 0 is reserved for NO_TARGET
            visited.add((nx, ny))
            queue.append(((nx, ny), step))

    return NO_TARGET


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
    # is the absorbing-row failure of E09 in a new place.
    bomb_useful = int(have_bomb and bomb_hits_crate(x, y, field))

    features = neighbour_status(x, y, field, danger, occupied) + (
        own_danger,
        target_direction(x, y, field, game_state['coins']),
        bomb_useful,
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

    if self.train or not os.path.isfile(MODEL_FILE):
        self.logger.info("Starting from an empty Q-table")
        self.q = np.zeros((N_STATES, len(ACTIONS)))
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
    q_row = self.q[state]
    best = np.flatnonzero(q_row == q_row.max())
    return ACTIONS[int(best[0] if best.size == 1 else self.policy_rng.choice(best))]
