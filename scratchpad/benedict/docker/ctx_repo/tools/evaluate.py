#!/usr/bin/env python3
"""Arena-matched evaluation harness for Bomberman agents.

Why this exists instead of ``main.py --save-stats``
---------------------------------------------------
The framework's own ``--save-stats`` writes two things: lifetime totals per
agent, and per-round totals *summed over all agents*. Neither lets us compute a
confidence interval for a single agent, because we never see that agent's score
in an individual round. Without per-round data we cannot say whether a change
helped or whether we are looking at noise -- and "systematic experimentation"
is the main grading criterion.

This harness drives ``BombeRLeWorld`` directly and snapshots every agent's
per-round statistics before the next round resets them. It does not modify any
framework file, so it keeps working when the tournament resets the framework to
upstream.

Arena matching
--------------
``main.py --seed N`` seeds the world RNG *once*. But the world draws from that
same RNG on every step (the agent activation order), so how many draws a round
consumes depends on how long the round lasted. Round 2 onwards therefore differs
between two agents even with an identical ``--seed``, and comparisons are
unpaired.

We reseed the world RNG with ``base_seed + round_index`` before each round.
Round *i* then has byte-identical crates, coins and starting corners for every
agent we ever evaluate with the same base seed. That makes A/B comparisons
*paired*, which typically shrinks the confidence interval on the difference by a
large factor -- we measure the effect of the change, not the luck of the draw.

Usage
-----
    uv run python tools/evaluate.py --agents my_agent rule_based_agent \\
        --n-rounds 300 --label "q_v3_vs_rulebased"

    uv run python tools/evaluate.py --agents my_agent --opponents rule_based \\
        --n-rounds 300 --scenario classic --label q_v3

Output: ``results/eval/<label>.csv`` (one row per round per agent) plus
``results/eval/<label>.meta.json`` (git commit, seed, settings -- everything
needed to reproduce the number that ends up in the report).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

import settings as s  # noqa: E402
from agents import Agent  # noqa: E402
from environment import BombeRLeWorld, WorldArgs  # noqa: E402

# Counters the framework maintains per round in ``Agent.statistics``.
# See agents.py -> EVENT_STAT_MAP and Agent.note_stat.
FRAMEWORK_STATS = [
    "score",     # points this round: 1 per coin, 5 per kill
    "coins",
    "kills",
    "suicides",
    "crates",
    "bombs",
    "moves",
    "invalid",
    "steps",     # steps this agent was alive for
    "time",      # total think time this round, seconds
]

# Preset opponent line-ups, matching the task ladder in AGENTS.md.
OPPONENT_PRESETS = {
    "none": [],
    "peaceful": ["peaceful_agent"] * 3,
    "coin_collector": ["coin_collector_agent"] * 3,
    "mixed": ["peaceful_agent", "coin_collector_agent", "rule_based_agent"],
    "rule_based": ["rule_based_agent"] * 3,
    "random": ["random_agent"] * 3,
}


# --------------------------------------------------------------------------
# Think-time instrumentation
# --------------------------------------------------------------------------
# The framework only sums think time per round. For the 0.5 s tournament limit
# the *tail* is what matters -- a mean of 5 ms with a 700 ms outlier still loses
# steps. We therefore record every individual measurement.
#
# This patches the in-memory class, not the file on disk. Framework sources stay
# untouched, which is what the submission rules require.
_think_times: dict[str, list[float]] = defaultdict(list)
_original_note_stat = Agent.note_stat


def _note_stat_recording(self, name, value=1):
    if name == "time":
        _think_times[self.name].append(float(value))
    return _original_note_stat(self, name, value)


Agent.note_stat = _note_stat_recording


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def git_commit() -> str:
    """Short commit hash, so a result file can be traced back to the code."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
        )
        head = out.stdout.strip() or "unknown"
        # Exclude our own output directory: the CSV this run is about to write
        # is untracked while the run happens, so counting it would stamp every
        # single run "-dirty" and make the marker useless.
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", ":(exclude)results"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return f"{head}-dirty" if dirty else head
    except Exception:
        return "unknown"


def build_world_args(scenario: str, log_dir: Path, seed: int, label: str) -> WorldArgs:
    """WorldArgs with the fields the environment actually reads.

    ``save_stats=False``: we write our own, richer records.
    ``continue_without_training=True``: rounds run to their natural end, exactly
    like a tournament game. Without it the round stops as soon as the training
    agent dies, which would silently truncate every evaluation round.
    """
    return WorldArgs(
        no_gui=True,
        fps=15,
        turn_based=False,
        update_interval=0.1,
        save_replay=False,
        replay=None,
        make_video=False,
        continue_without_training=True,
        log_dir=str(log_dir),
        save_stats=False,
        match_name=label,
        seed=seed,
        silence_errors=False,
        scenario=scenario,
    )


def resolve_agents(agents: list[str], opponents: str | None) -> list[str]:
    """Expand an opponent preset and pad/truncate to MAX_AGENTS."""
    line_up = list(agents)
    if opponents is not None:
        if opponents not in OPPONENT_PRESETS:
            raise SystemExit(
                f"Unknown opponent preset {opponents!r}. "
                f"Choose from: {', '.join(OPPONENT_PRESETS)}"
            )
        line_up += OPPONENT_PRESETS[opponents]
    if len(line_up) > s.MAX_AGENTS:
        raise SystemExit(
            f"{len(line_up)} agents requested but the game supports at most "
            f"{s.MAX_AGENTS}: {line_up}"
        )
    return line_up


# --------------------------------------------------------------------------
# Main evaluation loop
# --------------------------------------------------------------------------
def evaluate(
    agent_names: list[str],
    n_rounds: int,
    scenario: str,
    base_seed: int,
    label: str,
    log_dir: Path,
    progress: bool = True,
) -> tuple[list[dict], dict]:
    args = build_world_args(scenario, log_dir, base_seed, label)

    # (name, train) -- train is always False: evaluation runs the agent exactly
    # as the tournament will, with epsilon at its final value and the 0.5 s
    # timeout enforced.
    world = BombeRLeWorld(args, [(name, False) for name in agent_names])

    records: list[dict] = []
    started = time.time()

    try:
        for round_index in range(n_rounds):
            # The one line that makes comparisons paired.
            world.rng = np.random.default_rng(base_seed + round_index)

            # ...and from task 3 on, this one. All three provided opponents call
            # np.random.seed() with *no argument* in setup (peaceful_agent:5,
            # coin_collector_agent:68, rule_based_agent:69), reseeding the global
            # legacy RNG from OS entropy once per world -- so without this the
            # arenas are paired but the opponents are not, and rerunning the same
            # evaluation moves the numbers (E24 measured 12 vs 7 deaths against
            # peaceful_agent on identical seeds). The framework itself draws only
            # from `world.rng` (environment.py:335) and our agents only from
            # seeded default_rng generators, so this reaches the opponents and
            # nothing else: arenas are untouched and older baselines stay
            # comparable.
            np.random.seed(base_seed + round_index)

            world.new_round()
            while world.running:
                world.do_step()

            # end_round() has run, so Agent.statistics holds this round's
            # counters and Agent.dead tells us who made it out.
            round_records: list[dict] = []
            for slot, agent in enumerate(world.agents):
                record = {
                    "round": round_index,
                    "seed": base_seed + round_index,
                    "slot": slot,
                    "agent": agent.name,
                    "code": agent.code_name,
                    "survived": int(not agent.dead),
                    "round_steps": world.step,
                }
                for key in FRAMEWORK_STATS:
                    record[key] = agent.statistics.get(key, 0)

                times = _think_times.pop(agent.name, [])
                record["think_mean_ms"] = round(1000 * float(np.mean(times)), 4) if times else 0.0
                record["think_max_ms"] = round(1000 * float(np.max(times)), 4) if times else 0.0
                record["think_over_limit"] = int(sum(t > s.TIMEOUT for t in times))

                # How the agent died, split by cause. Both matter and they point
                # at different bugs: own bomb -> the escape logic is broken;
                # opponent's bomb -> positioning and danger awareness.
                # The framework fires GOT_KILLED for every death and additionally
                # KILLED_SELF when it was the agent's own bomb (environment.py,
                # evaluate_explosions), so the difference is exactly the kills by
                # others.
                died = int(agent.dead)
                suicides = record["suicides"]
                record["died"] = died
                record["killed_by_opponent"] = max(0, died - suicides)

                round_records.append(record)

            # Relative standing within the round -- a convenience of this
            # tool, NOT a concept the task defines. `final_project.pdf` §3 says
            # the winner is determined "by total score" over many episodes, so
            # an agent on 5.5 that is reliably second beats one on 5.0 that
            # leads 60 % of rounds. This comment asserted the inverse as fact
            # for the whole of rung 4. Report `won`/`rank` as secondaries.
            best = max(r["score"] for r in round_records)
            for record in round_records:
                # rank 1 = best; ties share the better rank
                record["rank"] = 1 + sum(
                    1 for other in round_records if other["score"] > record["score"]
                )
                record["won"] = int(record["score"] == best)
            records.extend(round_records)

            if progress and (round_index + 1) % 25 == 0:
                elapsed = time.time() - started
                rate = (round_index + 1) / elapsed
                remaining = (n_rounds - round_index - 1) / rate
                print(
                    f"  round {round_index + 1}/{n_rounds}"
                    f"  ({rate:.1f} rounds/s, ~{remaining:.0f}s left)",
                    file=sys.stderr, flush=True,
                )
    finally:
        world.end()

    meta = {
        "label": label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "agents": agent_names,
        "n_rounds": n_rounds,
        "scenario": scenario,
        "base_seed": base_seed,
        "seeds": [base_seed, base_seed + n_rounds - 1],
        "wall_clock_s": round(time.time() - started, 1),
        # Every BM_* switch in this process's environment. `MEASUREMENT.md` has
        # always claimed a run is reproducible "from the commit plus the BM_*
        # variables recorded in each run's .meta.json" -- and until now they were
        # not recorded, so the claim was false.
        #
        # It is not cosmetic. BM_OPPDIST / BM_D8 / BM_ABLATE change what a digit
        # *means*, so they change which row a state indexes. A table trained with
        # a switch and evaluated without it is read at the wrong indices, the
        # shapes still match, nothing errors, and the number is silently wrong.
        # After E36 that question could only be settled by re-running the
        # evaluation two ways, because the artifact did not say.
        "bm_env": {k: v for k, v in sorted(os.environ.items())
                   if k.startswith("BM_")},
        # Snapshot the rules, so we notice if someone evaluated against edited
        # settings (the classic "why can't I reproduce this" cause).
        "settings": {
            "COLS": s.COLS, "ROWS": s.ROWS, "MAX_STEPS": s.MAX_STEPS,
            "BOMB_POWER": s.BOMB_POWER, "BOMB_TIMER": s.BOMB_TIMER,
            "EXPLOSION_TIMER": s.EXPLOSION_TIMER, "TIMEOUT": s.TIMEOUT,
            "REWARD_KILL": s.REWARD_KILL, "REWARD_COIN": s.REWARD_COIN,
            "scenario_config": s.SCENARIOS[scenario],
        },
    }
    return records, meta


def write_outputs(records: list[dict], meta: dict, out_dir: Path, label: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{label}.csv"
    meta_path = out_dir / f"{label}.meta.json"

    fieldnames = list(records[0].keys())
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2, sort_keys=True)

    return csv_path


def print_quick_summary(records: list[dict]) -> None:
    """Rough numbers on stdout. Use tools/analyze.py for the real analysis."""
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_agent[record["agent"]].append(record)

    print()
    print(f"{'agent':<26} {'score':>13} {'coins':>7} {'kills':>7} "
          f"{'suicide':>8} {'surv':>6} {'maxms':>7}")
    print("-" * 80)
    for name, rows in by_agent.items():
        scores = np.array([r["score"] for r in rows], dtype=float)
        half_width = 1.96 * scores.std(ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else 0.0
        print(
            f"{name:<26} "
            f"{scores.mean():>6.2f} ±{half_width:<5.2f} "
            f"{np.mean([r['coins'] for r in rows]):>7.2f} "
            f"{np.mean([r['kills'] for r in rows]):>7.2f} "
            f"{np.mean([r['suicides'] for r in rows]):>8.2f} "
            f"{np.mean([r['survived'] for r in rows]):>6.2f} "
            f"{max(r['think_max_ms'] for r in rows):>7.1f}"
        )
    over = sum(r["think_over_limit"] for r in records)
    if over:
        print(f"\n  WARNING: {over} steps exceeded the {s.TIMEOUT}s limit and were "
              f"forced to WAIT.")
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate agents with per-round, arena-matched statistics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--agents", nargs="+", required=True,
                        help="Agent directory names, ours first.")
    parser.add_argument("--opponents", default=None,
                        choices=sorted(OPPONENT_PRESETS),
                        help="Fill the remaining slots with a preset line-up.")
    parser.add_argument("--n-rounds", type=int, default=300,
                        help="Rounds to play. 300+ for reportable numbers.")
    parser.add_argument("--scenario", default="classic", choices=sorted(s.SCENARIOS),
                        help="classic is the tournament setting.")
    parser.add_argument("--seed", type=int, default=20260731,
                        help="Base seed. Keep it FIXED across all runs you "
                             "intend to compare -- that is what pairs them.")
    parser.add_argument("--label", default=None,
                        help="Output file name. Defaults to a timestamp.")
    parser.add_argument("--out-dir", default=None,
                        help="Default: results/eval/")
    parser.add_argument("--log-dir", default=None,
                        help="Default: logs/")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)

    line_up = resolve_agents(args.agents, args.opponents)
    label = args.label or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "results" / "eval"
    log_dir = Path(args.log_dir) if args.log_dir else REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"Evaluating: {' vs '.join(line_up)}")
        print(f"  scenario={args.scenario}  rounds={args.n_rounds}  "
              f"seeds={args.seed}..{args.seed + args.n_rounds - 1}")

    records, meta = evaluate(
        agent_names=line_up,
        n_rounds=args.n_rounds,
        scenario=args.scenario,
        base_seed=args.seed,
        label=label,
        log_dir=log_dir,
        progress=not args.quiet,
    )

    csv_path = write_outputs(records, meta, out_dir, label)

    if not args.quiet:
        # resolve() first: a relative --out-dir would otherwise make relative_to
        # raise *after* the CSV is safely written, which reads like a lost run.
        shown = csv_path.resolve().relative_to(REPO_ROOT)
        print_quick_summary(records)
        print(f"Wrote {shown}  ({len(records)} rows, {meta['wall_clock_s']}s)")
        print(f"Analyse with: uv run python tools/analyze.py {shown}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
