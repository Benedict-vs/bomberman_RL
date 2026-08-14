"""D4 symmetry folding for the 8-digit feature map (E29 arm S).

The board's symmetry group is D4: four rotations x an optional mirror. Our state
is direction-indexed throughout, so the group acts on a *row index* exactly
rather than approximately:

    digits 1-4  one per direction, in DELTAS order (UP, RIGHT, DOWN, LEFT)
                -> permuted by the group
    digit 6     0 = NO_TARGET, else direction + 1
                -> value permuted by the group, 0 fixed
    digits 5, 7, 8   grace / bomb-useful / target distance
                -> invariant
    actions     UP RIGHT DOWN LEFT permute; WAIT, BOMB are fixed

DELTAS is listed clockwise, so a 90 deg clockwise rotation is `d -> (d+1) % 4`
and a mirror is `d -> -d % 4`. Every element is therefore

    g(d) = (s*d + k) % 4,    s in {+1, -1},  k in {0,1,2,3}

which is the whole 8-element group and nothing else.

**Why this file exists and `callbacks.py` is untouched.** The fold averages each
orbit and writes the result back into *every* member, so the table keeps its
64 000 x 6 shape and the agent's lookup path is unchanged. Canonicalising at
runtime would be equivalent but would put new code in the tournament agent for
no gain. Sharing updates *during training* is a different job and does need the
runtime version -- that is E29 stage 2, not this.

**The caveat this file cannot fix.** `bfs_first_step` breaks distance ties by
DELTAS order, measured at 45.5 / 30.6 / 13.5 / 10.5 %. On a tied state the
feature map is therefore *not* equivariant: the rotation of a state can carry a
different digit 6 than the rotated original. Folding merges those rows anyway.
That is exactly why E29 measures the fold both with the biased tie-break and
with a uniform one, instead of assuming the bias is harmless.

Usage:
    uv run python scratchpad/benedict/d4.py --self-test
    uv run python scratchpad/benedict/d4.py --fold IN.npy OUT.npy
"""

import argparse
import sys

import numpy as np

FEATURE_SIZES = (4, 4, 4, 4, 5, 5, 2, 5)
N_STATES = int(np.prod(FEATURE_SIZES))
N_ACTIONS = 6
N_DIRS = 4

# (s, k) for each of the 8 elements: d -> (s*d + k) % 4
GROUP = [(s, k) for s in (1, -1) for k in range(4)]


def apply_dir(g: tuple[int, int], d: int) -> int:
    """Image of direction index `d` under group element `g`."""
    s, k = g
    return (s * d + k) % N_DIRS


def inverse(g: tuple[int, int]) -> tuple[int, int]:
    """The element h with h(g(d)) == d for every d.

    g(d) = s*d + k, so h(e) = s*(e - k) = s*e - s*k, using s == 1/s over Z_4
    for both s = +1 and s = -1.
    """
    s, k = g
    return (s, (-s * k) % N_DIRS)


def decode(idx: int) -> tuple[int, ...]:
    """Row index -> digit tuple. Inverse of `encode`; digit 0 is most significant."""
    digits = []
    for size in reversed(FEATURE_SIZES):
        digits.append(idx % size)
        idx //= size
    return tuple(reversed(digits))


def encode(features) -> int:
    idx = 0
    for value, size in zip(features, FEATURE_SIZES):
        idx = idx * size + value
    return idx


def permute_features(features, g: tuple[int, int]) -> tuple[int, ...]:
    """Act on a digit tuple. Neighbour j moves to slot g(j); digit 6 is a direction."""
    out = list(features)
    for j in range(N_DIRS):
        out[apply_dir(g, j)] = features[j]
    d6 = features[5]
    out[5] = 0 if d6 == 0 else apply_dir(g, d6 - 1) + 1
    return tuple(out)


def permute_action(a: int, g: tuple[int, int]) -> int:
    """WAIT (4) and BOMB (5) are fixed points; the four moves permute."""
    return apply_dir(g, a) if a < N_DIRS else a


def _row_tables() -> tuple[np.ndarray, np.ndarray]:
    """`images[g_i, s]` = row index of g_i applied to state s, for all 8 elements."""
    images = np.empty((len(GROUP), N_STATES), dtype=np.int32)
    actions = np.empty((len(GROUP), N_ACTIONS), dtype=np.int8)
    for gi, g in enumerate(GROUP):
        for a in range(N_ACTIONS):
            actions[gi, a] = permute_action(a, g)
        for s in range(N_STATES):
            images[gi, s] = encode(permute_features(decode(s), g))
    return images, actions


def fold(q: np.ndarray, verbose: bool = True) -> tuple[np.ndarray, dict]:
    """Average each D4 orbit and write the result back into every member.

    A row is "trained" iff it has any non-zero entry -- zero is the untrained
    sentinel everywhere in this project, and averaging a trained row with an
    empty one would halve it rather than share it. So the mean runs over
    trained members only, and an orbit with no trained member stays zero.
    """
    images, actions = _row_tables()

    trained = np.abs(q).sum(axis=1) > 0
    canon = images.min(axis=0)                      # orbit representative per state

    total = np.zeros_like(q)
    count = np.zeros(N_STATES, dtype=np.int32)

    for gi, g in enumerate(GROUP):
        # State s maps to canonical row images[gi, s] under g; real action a in s
        # corresponds to canonical action g(a), so column a lands in column g(a).
        src = np.flatnonzero(trained)
        dst = images[gi, src]
        # Only accumulate through the element that actually reaches the canonical
        # row, otherwise every orbit is counted |stabiliser| times too often.
        keep = dst == canon[src]
        src, dst = src[keep], dst[keep]
        perm = actions[gi]
        np.add.at(total, (dst[:, None], perm[None, :]), q[src])
        np.add.at(count, dst, 1)

    merged = np.zeros_like(q)
    nonzero = count > 0
    merged[nonzero] = total[nonzero] / count[nonzero, None]

    # Push the canonical value back out to every orbit member, inverting both the
    # row map and the action permutation.
    out = np.zeros_like(q)
    for gi, g in enumerate(GROUP):
        inv_actions = np.empty(N_ACTIONS, dtype=np.int8)
        for a in range(N_ACTIONS):
            inv_actions[permute_action(a, g)] = a
        src = np.arange(N_STATES)
        dst = images[gi, src]
        take = dst == canon[src]
        out[src[take]] = merged[canon[src[take]]][:, inv_actions]

    stats = {
        "trained_before": int(trained.sum()),
        "trained_after": int((np.abs(out).sum(axis=1) > 0).sum()),
        "orbits_total": int(len(np.unique(canon))),
        "orbits_trained": int(nonzero.sum()),
    }
    if verbose:
        stats["gained"] = stats["trained_after"] - stats["trained_before"]
        for k, v in stats.items():
            print(f"  {k:20s} {v}")
    return out, stats


def self_test() -> int:
    """Group algebra only -- deliberately reads nothing about the real table."""
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            print(f"FAIL: {msg}")
            ok = False

    check(len(set(GROUP)) == 8, "group has 8 distinct elements")

    # closure and inverses on directions
    for g in GROUP:
        images = {apply_dir(g, d) for d in range(N_DIRS)}
        check(images == {0, 1, 2, 3}, f"{g} permutes directions")
        h = inverse(g)
        check(all(apply_dir(h, apply_dir(g, d)) == d for d in range(N_DIRS)),
              f"inverse of {g}")

    # encode/decode round trip
    for s in (0, 1, 12345, N_STATES - 1):
        check(encode(decode(s)) == s, f"encode(decode({s}))")

    # the action on rows is a permutation, and identity is identity
    ident = (1, 0)
    check(all(permute_features(decode(s), ident) == decode(s)
              for s in (0, 7, 999, N_STATES - 1)), "identity acts trivially")

    for g in GROUP:
        rows = {encode(permute_features(decode(s), g)) for s in range(0, N_STATES, 97)}
        check(len(rows) == len(range(0, N_STATES, 97)), f"{g} is injective on rows")

    # composition: applying g then h equals applying the composite on digits
    for g in GROUP:
        h = inverse(g)
        for s in (3, 4242, 55065, 34110):
            back = permute_features(permute_features(decode(s), g), h)
            check(back == decode(s), f"g then g^-1 restores state {s}")

    # a self-symmetric state must stay put under its stabiliser
    check(encode(permute_features(decode(0), (1, 1))) == 0,
          "all-zero digits are fixed by rotation")

    print("self-test PASSED" if ok else "self-test FAILED")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--fold", nargs=2, metavar=("IN", "OUT"))
    args = parser.parse_args()

    if args.self_test:
        return self_test()
    if args.fold:
        src, dst = args.fold
        q = np.load(src)
        if q.shape != (N_STATES, N_ACTIONS):
            print(f"unexpected shape {q.shape}")
            return 1
        out, _ = fold(q)
        np.save(dst, out)
        print(f"wrote {dst}")
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
