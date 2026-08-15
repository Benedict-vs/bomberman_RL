"""Warm-start parent for E36: broadcast the danger rows across the new digit 8.

`FEATURE_SIZES` is unchanged, so `warm_start` sees factor = 1 and does a straight
copy — which would leave every `d8 > 0` danger row at zero, i.e. the entire new
information dimension untrained. This does what `np.repeat` does for an appended
digit, restricted to the block that actually changes meaning.

Digit 8 is the least significant digit of the mixed-radix index, so the siblings
of a row `i` with `d8 = 0` are exactly `i+1 … i+4`.

Two things this has to get right, both measured by audit 7:

* the **1 131 valued `d8 = 0` danger rows** carry the values the shipped agent
  actually learned, and are the only sensible prior for their siblings;
* the **820 `d8 > 0` danger rows** hold **stale pre-E20 values** — they were
  reachable before E20 pinned the digit and have been unreachable since. They
  must be *overwritten*, not preserved, or the run starts from a table that is
  part rung-4 policy and part fossil.

    uv run python scratchpad/benedict/broadcast_oppdist.py IN.npy OUT.npy
"""

import argparse
import sys

import numpy as np

sys.path.insert(0, ".")
from agent_code.benedict_task4.callbacks import FEATURE_SIZES, N_STATES   # noqa: E402

D8_SIZE = FEATURE_SIZES[-1]


def digits(idx: int) -> list[int]:
    d, r = [], idx
    for size in reversed(FEATURE_SIZES):
        d.append(r % size)
        r //= size
    return list(reversed(d))


def broadcast(q: np.ndarray) -> tuple[np.ndarray, dict]:
    out = q.copy()
    sources = overwritten = 0
    for i in range(N_STATES):
        d = digits(i)
        if d[4] == 0 or d[7] != 0:      # not a danger row, or not the d8 = 0 base
            continue
        if not np.abs(q[i]).sum():      # nothing learned here to broadcast
            continue
        sources += 1
        for k in range(1, D8_SIZE):
            sib = i + k
            if np.abs(q[sib]).sum():
                overwritten += 1
            out[sib] = q[i]
    stats = {"sources": sources, "stale_overwritten": overwritten,
             "valued_before": int((np.abs(q).sum(axis=1) > 0).sum()),
             "valued_after": int((np.abs(out).sum(axis=1) > 0).sum())}
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src")
    ap.add_argument("dst")
    args = ap.parse_args()

    q = np.load(args.src)
    if q.shape != (N_STATES, 6):
        print(f"unexpected shape {q.shape}")
        return 1
    out, st = broadcast(q)
    np.save(args.dst, out)
    print(f"{args.dst}: {st['sources']} danger rows broadcast x{D8_SIZE - 1}, "
          f"{st['stale_overwritten']} stale pre-E20 rows overwritten, "
          f"valued {st['valued_before']} -> {st['valued_after']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
