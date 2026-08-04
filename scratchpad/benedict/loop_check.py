"""Does the greedy policy get stuck in an absorbing loop?

The `coin-heaven` field is deterministic (no crates), so the only thing that
varies between rounds is where the 50 coins are. That makes it cheap to replay
the greedy policy directly from the Q-table without starting the game engine --
and it answers the one question a 300-round evaluation cannot: *when* does the
agent stop making progress, and how much of the round does it waste.

A loop is absorbing here because the coin set only changes when a coin is
collected. If the cycle passes over no coin, the state never changes again.

    uv run python -m scratchpad.benedict.loop_check
"""

import random

import numpy as np

from agent_code.benedict_coin_collector.callbacks import (
    DELTAS, MODEL_FILE, state_to_features,
)

N_ROUNDS = 300
N_COINS = 50
MAX_STEPS = 400
SEED = 20260731          # not the harness seed; only fixes the coin placements here


def build_field() -> np.ndarray:
    w = h = 17
    field = np.zeros((w, h), dtype=int)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
    for x in range(w):
        for y in range(h):
            if x % 2 == 0 and y % 2 == 0:
                field[x, y] = -1
    return field


def main() -> None:
    q = np.load(MODEL_FILE)
    field = build_field()
    free = [(x, y) for x in range(17) for y in range(17) if field[x, y] == 0]
    corners = [(1, 1), (1, 15), (15, 1), (15, 15)]
    rng = random.Random(SEED)

    entries, collected, before = [], [], []
    for _ in range(N_ROUNDS):
        pos = rng.choice(corners)
        coins = set(rng.sample([f for f in free if f != pos], N_COINS))
        seen, got, loop_at = {}, 0, None

        for step in range(MAX_STEPS):
            s = state_to_features(
                {"field": field, "self": ("m", 0, True, pos), "coins": list(coins)}
            )
            a = int(np.argmax(q[s]))
            key = (pos, s, a)
            if key in seen and loop_at is None:
                loop_at = step
                before.append(got)
            seen[key] = step

            if a < 4:
                dx, dy = DELTAS[a]
                nxt = (pos[0] + dx, pos[1] + dy)
                if field[nxt] == 0:
                    pos = nxt
            if pos in coins:
                coins.discard(pos)
                got += 1
                seen = {}       # the state genuinely changed; old cycle is void

        collected.append(got)
        if loop_at is not None:
            entries.append(loop_at)

    print(f"rounds entering an absorbing loop : {len(entries)} / {N_ROUNDS}")
    if entries:
        print(f"loop starts at step               : mean {np.mean(entries):.0f}, "
              f"median {np.median(entries):.0f}")
        print(f"coins collected before the loop   : {np.mean(before):.1f}")
    print(f"coins per round                   : {np.mean(collected):.1f}")


if __name__ == "__main__":
    main()
