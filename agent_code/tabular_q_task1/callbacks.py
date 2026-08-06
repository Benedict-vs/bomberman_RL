"""Tabular Q-learning agent — merged task-1 baseline, built to grow into task 2.

Merged from two independently developed rung-1 coin collectors after both
solved the rung (50.00 +- 0.00 coins over 300 rounds each). What came from
where, and why, is in `experiments/task1.md` section 7:

- BFS direction to the nearest coin  (from Maxi's agent)
- random tie-breaking among equal-value actions  (from Maxi's agent)
- per-cell learning rate, full six-action set, dense mixed-radix table,
  seeded RNGs and the measurement harness  (from Benedict's agent)

State: 4 wall bits + BFS direction to the nearest coin -> 16 x 5 = 80 rows.
Deliberately small: task 2 multiplies this by the danger features, and the
reachable-row count is only a useful correctness check while it stays countable.
"""

import os

import numpy as np

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

# Digit 5 of the feature vector: 0 = no coin reachable, 1..4 = ACTIONS[0..3].
NO_COIN = 0

# 4 wall bits (U, R, D, L) + BFS direction to the nearest coin
FEATURE_SIZES = (2, 2, 2, 2, 5)
N_STATES = int(np.prod(FEATURE_SIZES))

# Ties are broken at random, so evaluation would stop being reproducible
# without a fixed seed here. Chosen once and never varied -- unlike the
# training seed, this one ships.
POLICY_SEED = 20260731


def coin_direction(x: int, y: int, field: np.ndarray, coins: list) -> int:
    """Index into ACTIONS of the first step on a shortest path to a coin.

    Breadth-first, so the first coin reached is a nearest one *by path*, which
    Manhattan distance is not -- it cannot see that a wall is in the way.
    Returns NO_COIN when there is no coin or none is reachable.

    Cost scales inversely with coin density: the search stops at the first coin
    found, so it expands a handful of nodes with 50 coins on the board and
    floods it when one distant coin is left (0.002 ms vs 0.12 ms measured).
    """
    if not coins:
        return NO_COIN

    targets = set(coins)        # set, not list: this is tested at every node
    if (x, y) in targets:
        return NO_COIN          # standing on it; it is collected this step

    # Each queue entry carries the first step that led to it, so the direction
    # falls out of the search without reconstructing the path.
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
            if (nx, ny) in targets:
                return step + 1         # +1 because 0 is reserved for NO_COIN
            visited.add((nx, ny))
            queue.append(((nx, ny), step))

    return NO_COIN


def state_to_features(game_state: dict) -> int:
    """Map a game state onto a row index of the Q-table."""

    if game_state is None:      # every caller guards; fail loudly if one stops
        raise ValueError("state_to_features called without a game state")

    field = game_state['field']     # indexing is field[x, y]
    x, y = game_state['self'][3]

    # free tiles are 0, stone is -1, crates are 1; != 0 already covers task 2.
    # TODO task 3+: `others` is not in here, so the agent walks into opponents
    # (environment.py:121 counts them as blocking). Measured cost on rung 1
    # in a two-agent game: 11.5 invalid actions per round.
    blocked = tuple(int(field[x + dx, y + dy] != 0) for dx, dy in DELTAS)

    return encode(blocked + (coin_direction(x, y, field, game_state['coins']),))


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
