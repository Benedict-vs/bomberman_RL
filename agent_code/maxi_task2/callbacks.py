"""

- BFS direction to the nearest coin  (from Maxi's agent)
- random tie-breaking among equal-value actions  (from Maxi's agent)
- per-cell learning rate, full six-action set, dense mixed-radix table,
  seeded RNGs and the measurement harness  (from Benedict's agent)

  now: bfs target direction, danger mapping, escape direction,
  danger features

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

# Digit 5 of the feature vector: direction *and* kind of the nearest target.
#   0       nothing reachable
#   1..4    a coin lies towards ACTIONS[0..3]   -> walk onto it
#   5..8    a crate lies towards ACTIONS[0..3]  -> stop short and bomb it
# One digit rather than (direction, kind) as two: with a separate kind digit the
# two (NO_TARGET, kind) rows would mean the same thing and each learn half the
# experience. The offsets keep every tuple distinct without those dead rows.
NO_TARGET = 0
COIN_OFFSET = 1
CRATE_OFFSET = 5

# 4 wall bits (U, R, D, L) + BFS target + danger (4) + escape BFS (5)
FEATURE_SIZES = (2, 2, 2, 2, 9, 4, 5)
N_STATES = int(np.prod(FEATURE_SIZES))

# Ties are broken at random, so evaluation would stop being reproducible
# without a fixed seed here. Chosen once and never varied -- unlike the
# training seed, this one ships.
POLICY_SEED = 20260731


def target_direction(x: int, y: int, field: np.ndarray, coins: list) -> int:
    """First step on a shortest path to a target, encoded together with its kind.

    Breadth-first, so the first target reached is a nearest one *by path*, which
    Manhattan distance is not -- it cannot see that a wall is in the way.
    Coins take priority; if none is reachable, falls back to the nearest crate,
    which is what gives the agent something to walk towards on `classic` once
    the revealed coins are gone. Both come out of a single BFS pass, so the
    0.12 ms worst case measured on task 1 still holds.

    Returns COIN_OFFSET/CRATE_OFFSET + the ACTIONS index of the first step, so
    the Q-table can learn a different response to each kind: a coin is walked
    onto, a crate is only ever approached and bombed. Returns NO_TARGET when
    neither a coin nor a crate is reachable.
    """
    # Neither coins nor crates on the board -- nothing to search for.
    if not coins and not (field == 1).any():
        return NO_TARGET

    targets = set(coins)        # set, not list: this is tested at every node
    if (x, y) in targets:
        return NO_TARGET        # standing on it; it is collected this step

    # Each queue entry carries the first step that led to it, so the direction
    # falls out of the search without reconstructing the path.
    queue = [((x, y), None)]
    visited = {(x, y)}
    head = 0
    width, height = field.shape

    nearest_crate_step = None

    while head < len(queue):
        (cx, cy), first = queue[head]
        head += 1

        for action_idx, (dx, dy) in enumerate(DELTAS):
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if (nx, ny) in visited:
                continue

            step = action_idx if first is None else first

            # Priority 1: a coin. Nothing beats it, so the search stops here.
            if (nx, ny) in targets:
                return COIN_OFFSET + step

            # Priority 2: a crate. BFS expands in rings, so the first crate seen
            # is the nearest one -- remember the way there, but keep searching
            # for a coin, which would override it.
            if field[nx, ny] == 1:
                if nearest_crate_step is None:
                    nearest_crate_step = CRATE_OFFSET + step
                # Marked visited but never queued: crates block the path.
                visited.add((nx, ny))
                continue

            if field[nx, ny] == -1:     # stone wall
                continue

            visited.add((nx, ny))       # free tile
            queue.append(((nx, ny), step))

    # Queue exhausted, so no coin is reachable. Fall back to the crate.
    if nearest_crate_step is not None:
        return nearest_crate_step

    return NO_TARGET


def state_to_features(game_state: dict) -> int:
    """Map a game state onto a row index of the Q-table."""
    if game_state is None:
        raise ValueError("state_to_features called without a game state")

    field = game_state['field']
    x, y = game_state['self'][3]
    bombs = game_state['bombs']
    explosion_map = game_state['explosion_map']

    # local blockage: 4 bits for U, R, D, L. 1 = blocked, 0 = free
    blocked = tuple(int(field[x + dx, y + dy] != 0) for dx, dy in DELTAS)
    
    # direction to coin or crate
    target = target_direction(x, y, field, game_state['coins'])
    
    # danger map
    danger_map = compute_danger_map(field, bombs, explosion_map)
    
    # am i standing on a danger tile? 
    danger = get_danger_feature(x, y, danger_map)
    
    # escape dir if needed
    escape = escape_direction(x, y, field, bombs, danger_map)

    
    return encode(blocked + (target, danger, escape))

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


NO_STEP = 0 

def compute_danger_map(field: np.ndarray, bombs: list, explosion_map: np.ndarray) -> np.ndarray:
    """
   2-D Matrix, same shape of field with the number of ticks until the next explosion on each cell
    99 = safe, 0 = explosion now, 1 = explosion next tick, 2 = explosion in 2 ticks, ..
    """
    danger_map = np.full(field.shape, 99)
    
    # explosions right now
    danger_map[explosion_map > 0] = 0
    
    # ticking bombs
    for (bx, by), t in bombs:
        # bom map
        danger_map[bx, by] = min(danger_map[bx, by], t)
        
        #radius of explosion (3 steps in each direction)
        for dx, dy in DELTAS:
            for step in range(1, 4): # BOMB_POWER = 3
                nx, ny = bx + dx * step, by + dy * step
                
                # still in bounds?
                if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]):
                    break
                
                # wall blcock explosion, stop propagation
                if field[nx, ny] == -1:
                    break
                
                danger_map[nx, ny] = min(danger_map[nx, ny], t)
                
                # crates block explo, but are destroyed 
                if field[nx, ny] == 1:
                    break
                    
    return danger_map

def get_danger_feature(x: int, y: int, danger_map: np.ndarray) -> int:
    """
    translates dangermap values to feature values for the Q-table.
    """
    t = danger_map[x, y]
    if t == 99: return 0 # safe 
    if t >= 3:  return 1 # still time left
    if t == 2:  return 2 # close call (t == 2)
    return 3             # run forrest, run! 

def escape_direction(x: int, y: int, field: np.ndarray, bombs: list, danger_map: np.ndarray) -> int:
    """
    bfs to search for safe spots in danger map. 
    This function is only called when the agent is in danger
    """
    if danger_map[x, y] == 99:
        return NO_STEP # already safe
        
    #bombs block the path, so treat them as walls 
    bomb_locs = {b[0] for b in bombs}
    
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
            if (nx, ny) in visited:
                continue
                
            # don't walk into walls or bombs
            if field[nx, ny] != 0 or (nx, ny) in bomb_locs:
                continue
                
            step = action_idx if first is None else first
            
            #return first step to safe spot
            if danger_map[nx, ny] == 99:
                return step + 1
                
            visited.add((nx, ny))
            queue.append(((nx, ny), step))
            
    # well, fuck 
    return NO_STEP
