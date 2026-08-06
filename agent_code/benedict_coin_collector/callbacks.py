"""Tabular Q-learning agent — task 1 (`coin-heaven`) baseline.

The whole model is one `(N_STATES, len(ACTIONS))` array of Q-values on disk.
`state_to_features` maps a game state onto a row index; `act` takes the argmax
of that row. Everything that learns lives in `train.py`, which the tournament
never imports.

Final task-1 result (5 seeds x 300 rounds): 49.15 +- 1.23 of 50 coins, ~98 %
full sweeps, 100 % survival, 0.1 ms per decision. See `experiments/benedict.md`
E01-E07 for how each piece was arrived at; the two that mattered most were the
clipped coin offset (E02/E04) and the per-cell learning rate (E06).
"""

import os

import numpy as np

# Training-only escape hatch: several training runs launched in parallel would
# otherwise all write the same file and clobber each other. With the variable
# unset -- every normal game, and the tournament, where the environment is not
# ours to set -- this is exactly "q_table.npy", so the submitted path is
# unchanged. Relative to this file, never absolute.
MODEL_FILE = os.path.join(
    os.path.dirname(__file__),
    f"q_table{os.environ.get('BM_MODEL_SUFFIX', '')}.npy",
)

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

# One radix per feature digit: 4 wall bits, then dx and dy to the nearest coin.
# 784 rows, of which 295 are reachable on the task-1 arena (counted in E04).
FEATURE_SIZES = (2, 2, 2, 2, 7, 7)

COIN_CLIP = 3       # dx, dy are clipped to [-COIN_CLIP, +COIN_CLIP]

N_STATES = int(np.prod(FEATURE_SIZES))

# (dx, dy) for UP, RIGHT, DOWN, LEFT - image coords, y grows downwards
DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


def state_to_features(game_state: dict) -> int:
    """Map a game state onto a row index of the Q-table."""

    if game_state is None:      # every caller guards; fail loudly if one stops
        raise ValueError("state_to_features called without a game state")

    field = game_state['field']     # indexing is field[x, y]
    x, y = game_state['self'][3]
    coins = game_state['coins']

    if coins:
        cx, cy = min(coins, key=lambda c: (abs(c[0] - x) + abs(c[1] - y)))  # Manhattan distance to closest coin
        direction = (int(np.clip(cx - x, -COIN_CLIP, COIN_CLIP)) + COIN_CLIP,  # shift to [0, 2*COIN_CLIP] for encoding
                     int(np.clip(cy - y, -COIN_CLIP, COIN_CLIP)) + COIN_CLIP)
    else:
        # Offset (0, 0). Shared with "a coin is on my own tile" -- both mean
        # "no direction information", but they are NOT the same situation.
        # Cost 3 of 300 rounds in E04; separate them before task 2, where the
        # coin list is empty for long stretches.
        direction = (COIN_CLIP, COIN_CLIP)

    # free tiles are 0, stone is -1, crates are 1
    # != 0 means it works for task 2 when crates appear
    blocked = tuple(int(field[x + dx, y + dy] != 0) for dx, dy in DELTAS)

    return encode(blocked + direction)


def encode(features: tuple[int, ...]) -> int:
    """Flatten a feature tuple to a single Q-table row index (mixed radix).

    Each feature is one digit whose base is its entry in FEATURE_SIZES, so
    distinct tuples always map to distinct rows in [0, N_STATES). Adding a
    feature only requires extending FEATURE_SIZES.
    """

    idx = 0
    for value, size in zip(features, FEATURE_SIZES):
        idx = idx * size + value
    return idx


def setup(self):
    """Called once before a set of games to initialize data structures."""

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
    # purely greedy, so the tournament never reaches either of them -- which is
    # what keeps the agent working when train.py is not imported at all.
    if self.train and self.rng.random() < self.eps:
        return ACTIONS[int(self.rng.integers(len(ACTIONS)))]

    return ACTIONS[int(np.argmax(self.q[state]))]
