import logging
import os
from pathlib import Path

from fallbacks import pygame

# Game properties
# board size (a smaller board may be useful at the beginning)
COLS = 17
ROWS = 17
SCENARIOS = {
    # modes useful for agent development
	"empty": {
        "CRATE_DENSITY": 0, 
        "COIN_COUNT": 0 
    },
    "coin-heaven": {
        "CRATE_DENSITY": 0,
        "COIN_COUNT": 50
    },
    "loot-crate": { 
        "CRATE_DENSITY": 0.75, 
        "COIN_COUNT": 50 
    }, 
    # this is the tournament game mode
    "classic": {
        "CRATE_DENSITY": 0.75,
        "COIN_COUNT": 9
    }
    # Feel free to add more game modes and properties
    # game is created in environment.py -> BombeRLeWorld -> build_arena()
}
MAX_AGENTS = 4

# Round properties
MAX_STEPS = 400

# GUI properties
GRID_SIZE = 30
WIDTH = 1000
HEIGHT = 600
GRID_OFFSET = [(HEIGHT - ROWS * GRID_SIZE) // 2] * 2

ASSET_DIR = Path(__file__).parent / "assets"

AGENT_COLORS = ['blue', 'green', 'yellow', 'pink']

# Game rules
BOMB_POWER = 3
BOMB_TIMER = 4
EXPLOSION_TIMER = 2  # = 1 of bomb explosion + N of lingering around

# Rules for agents
TIMEOUT = 0.5
TRAIN_TIMEOUT = float("inf")
REWARD_KILL = 5
REWARD_COIN = 1

# User input
INPUT_MAP = {
    pygame.K_UP: 'UP',
    pygame.K_DOWN: 'DOWN',
    pygame.K_LEFT: 'LEFT',
    pygame.K_RIGHT: 'RIGHT',
    pygame.K_RETURN: 'WAIT',
    pygame.K_SPACE: 'BOMB',
}

# Logging levels
#
# Training-only escape hatch (ours, not upstream): with BM_QUIET_LOGS set, the
# engine's per-step logging drops to WARNING. A 40 000-round run on `classic` is
# ~10 M steps, and a five-seed sweep runs five of them at once, so the INFO
# stream is a real share of the wall clock. It is also close to worthless during
# a sweep: agents.py:226-228 hardcodes every run's agent log to
# agent_code/<name>/logs/<name>.log in mode "w", so parallel runs overwrite one
# another's and the training CSV is the only usable record anyway.
#
# Deliberately an environment switch rather than an edited constant: unset --
# every normal game, every evaluation, and the tournament -- these are exactly
# the upstream values, so there is nothing to remember to restore before
# submitting. Note that tools/evaluate.py's settings snapshot does NOT cover log
# levels, so an edited constant here would have been invisible in .meta.json.
_QUIET_LOGS = bool(os.environ.get("BM_QUIET_LOGS"))
LOG_GAME = logging.WARNING if _QUIET_LOGS else logging.INFO
LOG_AGENT_WRAPPER = logging.WARNING if _QUIET_LOGS else logging.INFO
LOG_AGENT_CODE = logging.WARNING if _QUIET_LOGS else logging.DEBUG
LOG_MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB
