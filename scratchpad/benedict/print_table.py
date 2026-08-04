import numpy as np
from agent_code.benedict_coin_collector.callbacks import (
    ACTIONS, FEATURE_SIZES, N_STATES, MODEL_FILE,
)

def decode(idx: int) -> tuple[int, ...]:
    out = []
    for size in reversed(FEATURE_SIZES):
        out.append(idx % size)
        idx //= size
    return tuple(reversed(out))

q = np.load(MODEL_FILE)

# One header label per digit, so the table stays readable when FEATURE_SIZES grows.
LABELS = ["U", "R", "D", "L", "dx", "dy"][:len(FEATURE_SIZES)]
header = " ".join(f"{l:>2}" for l in LABELS)

print(f"{'idx':>4} {header} " + " ".join(f"{a:>8}" for a in ACTIONS) + "   best")
for s in range(N_STATES):
    if not q[s].any():
        continue        # 295 of 784 rows are reachable; the rest are noise in the eye
    digits = " ".join(f"{d:>2}" for d in decode(s))
    row = " ".join(f"{v:8.3f}" for v in q[s])
    print(f"{s:>4} {digits} {row}   {ACTIONS[int(np.argmax(q[s]))]}")

occupied = int(q.any(axis=1).sum())
print(f"\n{occupied} of {N_STATES} rows occupied")