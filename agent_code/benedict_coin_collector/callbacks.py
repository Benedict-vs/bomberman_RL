import os
import pickle

import numpy as np

MODEL_FILE = os.path.join(os.path.dirname(__file__), "q_table.npy")

ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

FEATURE_SIZES = (2, 2, 2, 2, 7, 7) # 4 wall bits + dx, dy clipped to [-3, 3]

COIN_CLIP = 3

N_STATES = int(np.prod(FEATURE_SIZES))

# (dx, dy) for UP, RIGHT, DOWN, LEFT - image coords, y grows downwards
DELTAS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

def state_to_features(game_state: dict) -> int | None:
    """Map a game state onto a row index of the Q-table"""
    
    if game_state is None:          # bevor the first and after the last step
        return None
    
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
    
    eps = self.eps if self.train else 0.0           # exploration rate: set in train.py
    if np.random.random() < eps:
        return np.random.choice(ACTIONS)
    
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
