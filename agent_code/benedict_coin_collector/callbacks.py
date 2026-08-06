import os
import pickle

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

FEATURE_SIZES = (2, 2, 2, 2, 7, 7) # 4 wall bits + dx, dy clipped to [-3, 3]

COIN_CLIP = 3

N_STATES = int(np.prod(FEATURE_SIZES))

# (dx, dy) for UP, RIGHT, DOWN, LEFT - image coords, y grows downwards
DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

def state_to_features(game_state: dict) -> int:
    """Map a game state onto a row index of the Q-table"""
    
    if game_state is None:
        raise ValueError("state_to_features called without a game state")
    
    field = game_state['field']     # indexing is field[x, y]
    x, y = game_state['self'][3]
    coins = game_state['coins']
    
    if coins:
        cx, cy = min(coins, key=lambda c: (abs(c[0] - x) + abs(c[1] - y)))  # Manhattan distance to closest coin
        direction = (int(np.clip(cx - x, -COIN_CLIP, COIN_CLIP)) + COIN_CLIP,  # shift to [0, 2*COIN_CLIP] for encoding
                     int(np.clip(cy - y, -COIN_CLIP, COIN_CLIP)) + COIN_CLIP)
    else:
        direction = (COIN_CLIP, COIN_CLIP)  # offset (0, 0)
    
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
        self.logger.info("Starting from an emtpy Q-table")
        self.q = np.zeros((N_STATES, len(ACTIONS)))
    else:
        self.logger.info(f"Loading Q-table from disk.")
        self.q = np.load(MODEL_FILE)

        # A table left over from an older FEATURE_SIZES would not fail here but
        # deep inside act(), as an IndexError in the middle of a round.
        expected = (N_STATES, len(ACTIONS))
        if self.q.shape != expected:
            raise ValueError(
                f"Q-table on disk has shape {self.q.shape}, expected {expected}. "
                "It was trained with a different feature layout -- retrain."
            )


def act(self, game_state: dict):
    """Called each game step to determine the agent's next action."""
    
    state = state_to_features(game_state)

    # self.eps and self.rng are set in train.py. Outside training the policy is
    # purely greedy, so the tournament never reaches either of them -- which is
    # what keeps the agent working when train.py is not imported at all.
    if self.train and self.rng.random() < self.eps:
        return ACTIONS[int(self.rng.integers(len(ACTIONS)))]

    return ACTIONS[int(np.argmax(self.q[state]))]   # untrained agents will have
                                                    # q[state] = 0, so argmax
                                                    # returns 0, i.e. UP



# # build test board
# W = H = 17
# field = np.zeros((W, H), dtype=int)

# # walls
# field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1    # outside walls
# for x in range(W):
#     for y in range(H):
#         if x % 2 == 0 and y % 2 == 0:
#             field[x, y] = -1            # inside walls every second position

# def fake_state(x, y):
#     return {'field': field, 'self':('me', 0, True, (x, y))}

# assert(state_to_features(fake_state(1, 2)) == 5)
