"""E24 post-mortem: are the rung-3 deaths in rows the rung-2 training never practised?

Two competing explanations for the survival collapse in E24, with opposite fixes:

  distribution  the danger/escape digits are adequate, but opponents push the agent
                into rows it never trained on  ->  train in an opponent field (E25)
  features      the deaths sit in well-practised rows, so the state simply cannot
                express the situation                    ->  new digits first (E26)

The discriminator is the *death row* rate against a **base rate** over all steps.
"Most death rows are unpractised" is worthless on its own: if half of every step in
an opponent field is unpractised, so are the deaths, and the number says nothing.
This is the same trap as E22's residual argument -- a count is not a rate.

"Practised" has an exact test here: `callbacks.setup` initialises the table with
`np.zeros`, so an all-zero row was never updated by either training stage. A second,
softer test is whether the row occurs at all in a solo rollout of the same policy.

Run:  uv run python scratchpad/benedict/death_rows.py --n-rounds 100
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import settings as s                                    # noqa: E402
from environment import BombeRLeWorld, WorldArgs        # noqa: E402

AGENT = "benedict_task2"
# How many steps before the fatal one to attribute the death to. A bomb is lethal
# BOMB_TIMER steps after it is dropped, so the decision that killed the agent is
# up to that far back.
LOOKBACK = s.BOMB_TIMER


def load_callbacks():
    """Import the agent's callbacks as a module without going through main.py."""
    path = ROOT / "agent_code" / AGENT / "callbacks.py"
    spec = importlib.util.spec_from_file_location(f"{AGENT}_callbacks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_args(log_dir: Path) -> WorldArgs:
    return WorldArgs(
        no_gui=True, fps=1000, turn_based=False, update_interval=0.1,
        save_replay=False, replay=None, make_video=False, continue_without_training=True,
        log_dir=str(log_dir), save_stats=False, match_name="death_rows",
        seed=None, silence_errors=True, scenario="classic",
    )


def rollout(cb, line_up: list[str], n_rounds: int, base_seed: int, log_dir: Path):
    """Play `n_rounds` and record, per step, the row the agent stood in.

    Returns (all_rows, death_windows, death_details). `death_windows` holds the
    last LOOKBACK+1 rows of every round the agent died in.
    """
    args = build_args(log_dir)
    world = BombeRLeWorld(args, [(name, False) for name in line_up])

    all_rows: Counter[int] = Counter()
    death_windows: list[list[int]] = []
    death_details: list[dict] = []

    for round_index in range(n_rounds):
        world.rng = np.random.default_rng(base_seed + round_index)   # same arenas as evaluate.py
        world.new_round()
        # Only `do_step` sets this, and we call `get_state_for_agent` before the
        # first one. The agent never reads it.
        world.user_input = None

        me = world.agents[0]
        assert me.code_name == AGENT, f"slot 0 is {me.code_name}"

        trace: list[int] = []
        context: list[dict] = []

        while world.running:
            if not me.dead:
                gs = world.get_state_for_agent(me)
                row = cb.state_to_features(gs)
                trace.append(row)
                all_rows[row] += 1

                # Mechanism probe, independent of any visitation argument:
                # is the agent standing in a blast with no way out?
                x, y = gs["self"][3]
                danger = cb.danger_map(gs)
                occupied = {pos for pos, _ in gs["bombs"]}
                occupied.update(o[3] for o in gs["others"])
                own_danger = 0 if danger[x, y] >= cb.SAFE else int(danger[x, y]) + 1
                escape = (cb.escape_direction(x, y, gs["field"], danger, occupied)
                          if own_danger else None)
                context.append({
                    "own_danger": own_danger,
                    "escape": escape,
                    # `game_state['bombs']` carries no owner, so an agent cannot
                    # tell its own bomb from anyone else's -- a real limitation of
                    # the observation, and the reason there is no "foreign bomb"
                    # count here. Only the total is measurable.
                    "n_bombs": len(gs["bombs"]),
                })
            world.do_step()

        if me.dead and trace:
            death_windows.append(trace[-(LOOKBACK + 1):])
            death_details.append(context[-1] | {
                "row": trace[-1],
                # Was the agent ever trapped -- in a blast, no escape -- in the
                # window that killed it?
                "trapped_in_window": any(
                    c["own_danger"] > 0 and c["escape"] == cb.NO_TARGET
                    for c in context[-(LOOKBACK + 1):]
                ),
                "max_bombs_in_window": max(
                    (c["n_bombs"] for c in context[-(LOOKBACK + 1):]), default=0
                ),
            })

    return all_rows, death_windows, death_details


def report(name, all_rows, death_windows, death_details, unpractised, solo_rows):
    n_steps = sum(all_rows.values())
    n_deaths = len(death_windows)
    print(f"\n===== {name} =====")
    print(f"rounds with a death: {n_deaths}, steps recorded: {n_steps}")
    if not n_steps:
        return

    def rate(rows, pred):
        rows = list(rows)
        return 100 * sum(pred(r) for r in rows) / len(rows) if rows else float("nan")

    # Base rate: over every step taken in this field, weighted by visits.
    base_unpractised = 100 * sum(c for r, c in all_rows.items() if unpractised[r]) / n_steps
    death_rows = [w[-1] for w in death_windows]
    window_rows = [r for w in death_windows for r in w]

    print(f"  unpractised (all-zero Q row)   base rate over all steps : {base_unpractised:6.2f} %")
    print(f"                                 at the fatal step        : {rate(death_rows, lambda r: unpractised[r]):6.2f} %")
    print(f"                                 anywhere in the {LOOKBACK}-step window: "
          f"{rate(window_rows, lambda r: unpractised[r]):6.2f} %")

    if solo_rows is not None:
        base_unseen = 100 * sum(c for r, c in all_rows.items() if r not in solo_rows) / n_steps
        print(f"  never seen in the solo rollout base rate over all steps : {base_unseen:6.2f} %")
        print(f"                                 at the fatal step        : {rate(death_rows, lambda r: r not in solo_rows):6.2f} %")

    if death_details:
        trapped = 100 * sum(d["trapped_in_window"] for d in death_details) / len(death_details)
        print(f"  in a blast with NO escape route in the window          : {trapped:6.2f} % of deaths")
        print("  bombs on the board in the window: "
              f"{Counter(d['max_bombs_in_window'] for d in death_details).most_common()}")
        dist = Counter(d["own_danger"] for d in death_details)
        print(f"  own_danger at the fatal step: {dict(sorted(dist.items()))}  (0 = tile was safe)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-rounds", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260731)
    args = parser.parse_args()

    cb = load_callbacks()
    q = np.load(ROOT / "agent_code" / AGENT / "q_table.npy")
    unpractised = ~q.any(axis=1)        # never updated by either training stage
    print(f"table {q.shape}: {unpractised.sum()} of {len(unpractised)} rows all-zero "
          f"({100 * unpractised.mean():.1f} %)")

    log_dir = ROOT / "logs" / "death_rows"
    log_dir.mkdir(parents=True, exist_ok=True)

    fields = {
        "solo (rung 2)": [AGENT],
        "3x peaceful_agent": [AGENT] + ["peaceful_agent"] * 3,
        "3x coin_collector_agent": [AGENT] + ["coin_collector_agent"] * 3,
        "3x rule_based_agent": [AGENT] + ["rule_based_agent"] * 3,
    }

    solo_rows = None
    for name, line_up in fields.items():
        all_rows, windows, details = rollout(cb, line_up, args.n_rounds, args.seed, log_dir)
        if solo_rows is None:
            solo_rows = set(all_rows)       # the first field is the solo control
        report(name, all_rows, windows, details, unpractised, solo_rows)


if __name__ == "__main__":
    main()
