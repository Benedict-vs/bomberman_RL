"""Reproduce code-review findings without games, training, or model writes.

Run from the repository root: .venv/bin/python scratchpad/ben_task4_diagnostic_probes_20260913.py
Assertions document CURRENT behavior, not desired production behavior.
"""

from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_code.ben_task4 import train
from agent_code.ben_task4.features import (
    ESCAPE_TILES_CHANNEL,
    state_to_features,
)
from agent_code.ben_task4.replay_buffer import ReplayBuffer


def state(field, position, available=True, explosions=None):
    return {
        "field": field,
        "self": ("ben_task4", 0, available, position),
        "others": [],
        "bombs": [],
        "coins": [],
        "explosion_map": np.zeros_like(field) if explosions is None else explosions,
        "round": 1,
        "step": 10,
        "escape_feature_mode": "reachable_safe_tiles",
    }


def main():
    # The only way out crosses an active explosion. A geometric endpoint
    # remains marked because the search never tests intermediate danger.
    field = np.full((17, 17), -1, dtype=int)
    for tile in [(1, 1), (2, 1), (3, 1), (4, 1), (5, 1), (3, 2), (3, 3)]:
        field[tile] = 0
    explosions = np.zeros_like(field)
    explosions[2:6, 1] = 1
    unsafe = state(field, (1, 1), explosions=explosions)
    features = state_to_features(unsafe)
    assert features[ESCAPE_TILES_CHANNEL, 2, 3] == 1.0
    print("escape: (3,2) marked reachable despite active explosion on the only exit (2,1)")

    # This channel describes a hypothetical new bomb only, so it disappears
    # when the placed bomb actually has to be escaped. Other danger channels
    # still exist; this is an information limitation, not total blindness.
    unavailable = state(field, (1, 1), available=False)
    unavailable["bombs"] = [((1, 1), 3)]
    assert not state_to_features(unavailable)[ESCAPE_TILES_CHANNEL].any()
    print("escape: channel entirely zero while bomb_available=False")

    # Exercise the actual surviving-round callback, but mock every write and
    # never invoke _after_transition or an optimizer. A safe final state has
    # Phi=1. Marking done preserves gamma*Phi(next) in the stored reward.
    safe = state(field, (1, 1))
    base_reward = train.reward_from_events([])
    stored_reward = base_reward + train.safety_potential_shaping_reward(safe, safe)
    expected_terminal_reward = base_reward + train.safety_potential_shaping_reward(safe, None)
    buffer = ReplayBuffer(4)
    f = state_to_features(safe)
    buffer.append(f, 1, stored_reward, f, False)
    stub = SimpleNamespace(
        replay_buffer=buffer,
        n_step_queue=deque(),
        online_network=Mock(),
        trainlog=None,
        logger=Mock(),
        episode_reward=stored_reward,
        episode_events=[],
        episode_losses=[],
        epsilon=0.05,
        safe_offense_bombs=0,
    )
    stub.online_network.state_dict.return_value = {}
    with patch.object(train.torch, "save"), patch.object(train, "N_STEP_RETURN", 1):
        train.end_of_round(stub, safe, "RIGHT", [])
    final = buffer.sample(1)[0]
    assert final.done
    assert np.isclose(final.reward, -0.06)
    assert np.isclose(expected_terminal_reward, -1.05)
    assert np.isclose(final.reward - expected_terminal_reward, train.GAMMA)
    print(f"terminal shaping: stored={final.reward:.2f}, Phi(terminal)=0 convention={expected_terminal_reward:.2f}, difference=+{train.GAMMA:.2f}")
    print("No game, optimizer step, or model write executed.")


if __name__ == "__main__":
    main()
