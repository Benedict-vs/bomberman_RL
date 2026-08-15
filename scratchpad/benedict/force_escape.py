"""Re-point the argmax to digit 6 in every danger row (E34 initial condition).

This is the construction that measured the ceiling in audit 5 — score 3.719 →
4.399, `won` 0.372 → 0.442, 5/5 seeds — parameterised so any table can be
transformed, rather than audit 5's `mkesc.py` which hardcodes two E31 seeds.

Where digit 5 > 0 (a blast covers my tile) and digit 6 ≠ 0 (the escape BFS found
a way out) and the row carries value, the escape action's Q is set to
`row.max() + 1.0`. Everything else is untouched.

**As a policy this cannot ship** — `AGENTS.md` forbids a feature that returns the
best action, and a hand-written argmax is exactly that. E34 uses it only as an
*initial condition*: the objective is unchanged and Q-learning then decides
whether to stay. Whether it stays is the experiment, and E34's P3 pre-commits
how each outcome must be reported — including "training did nothing, so do not
ship this".

    uv run python scratchpad/benedict/force_escape.py IN.npy OUT.npy
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from agent_code.benedict_task4.callbacks import FEATURE_SIZES, N_STATES   # noqa: E402


def digits(idx: int) -> list[int]:
    d, r = [], idx
    for size in reversed(FEATURE_SIZES):
        d.append(r % size)
        r //= size
    return list(reversed(d))


def force(q: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Returns (new table, rows re-pointed, danger rows that already agreed)."""
    out = q.copy()
    repointed = agreed = 0
    for i in range(N_STATES):
        d = digits(i)
        if d[4] == 0 or d[5] == 0:          # safe, or no escape route known
            continue
        if not np.abs(q[i]).sum():          # untrained row: nothing to re-point
            continue
        escape = d[5] - 1
        if int(np.argmax(q[i])) == escape:
            agreed += 1
        else:
            out[i, escape] = q[i].max() + 1.0
            repointed += 1
    return out, repointed, agreed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()

    q = np.load(args.src)
    if q.shape != (N_STATES, 6):
        print(f"unexpected shape {q.shape}")
        return 1
    out, repointed, agreed = force(q)
    np.save(args.dst, out)
    total = repointed + agreed
    print(f"{args.dst}: {repointed} re-pointed, {agreed} already agreed "
          f"({agreed / max(1, total):.3f} of {total} valued danger rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
