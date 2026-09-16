"""Position-balanced comparison of frozen mixed-kill and Task-3 DQNs.

The four list slots reuse the same 1,000 arena seeds.  Therefore the unit of
analysis is an arena seed: first average each policy over its four slots, then
compare the resulting 1,000 paired arena means.  Treating all 4,000 rows as
independent would be pseudoreplication.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results" / "eval" / "task4_tournament"

MIXED_STEM_GROUPS = [
    [
        "ben_dqn_task4_mixed_kill_v1_2000ep_seed11__task4_rule_based_retry1_eval1000",
    ],
    *[
        [
        "ben_dqn_task4_mixed_kill_v1_2000ep_seed11__task4_rule_based_"
        f"listslot{slot}_eval1000"
        ]
        for slot in (1, 2, 3)
    ],
]
INCUMBENT_STEM_GROUPS = [
    [
        "ben_dqn_task3_seed13__task4_rule_based_eval1000",
        "ben_dqn_task3_seed13__task4_rule_based_retry1_eval1000",
    ],
    *[
        [f"ben_dqn_task3_seed13__task4_rule_based_listslot{slot}_eval1000"]
        for slot in (1, 2, 3)
    ],
]

METRICS = {
    "score": "score",
    "won": "won",
    "kills": "kills",
    "suicides": "suicides",
    "killed_by": "killed_by_opponent",
    "survived": "survived",
    "coins": "coins",
    "bombs": "bombs",
    "think_ms": "think_mean_ms",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy(
    stem_groups: list[list[str]],
    code: str,
    model_path: Path,
) -> tuple[dict[int, dict[int, list[dict[str, str]]]], list[dict]]:
    expected_model_sha = sha256(model_path)
    expected_callbacks_sha = sha256(model_path.parent / "callbacks.py")
    expected_settings = {
        "BOMB_POWER": 3,
        "BOMB_TIMER": 4,
        "EXPLOSION_TIMER": 2,
        "MAX_STEPS": 400,
        "REWARD_COIN": 1,
        "REWARD_KILL": 5,
        "TIMEOUT": 0.5,
    }
    by_slot: dict[int, dict[int, list[dict[str, str]]]] = {}
    metadata = []

    for expected_slot, stems in enumerate(stem_groups):
        by_slot[expected_slot] = {
            seed: [] for seed in range(20260731, 20261731)
        }
        for stem in stems:
            csv_path = EVAL / f"{stem}.csv"
            meta_path = EVAL / f"{stem}.meta.json"
            with csv_path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            meta = json.loads(meta_path.read_text())

            assert len(rows) == 4_000, (stem, len(rows))
            assert len({int(row["round"]) for row in rows}) == 1_000
            assert {int(row["seed"]) for row in rows} == set(
                range(20260731, 20261731)
            )
            own_rows = [row for row in rows if row["code"] == code]
            assert len(own_rows) == 1_000
            assert {int(row["slot"]) for row in own_rows} == {expected_slot}
            assert sum(int(row["think_over_limit"]) for row in rows) == 0
            assert all(
                float(row["score"])
                == float(row["coins"]) + 5.0 * float(row["kills"])
                for row in rows
            )
            assert all(
                int(row["died"])
                == int(row["suicides"]) + int(row["killed_by_opponent"])
                for row in own_rows
            )
            assert all(
                int(row["survived"]) == 1 - int(row["died"])
                for row in own_rows
            )

            provenance = meta["agent_provenance"][code]
            assert provenance["model_exists"] is True
            assert provenance["model_sha256"] == expected_model_sha
            assert provenance["callbacks_sha256"] == expected_callbacks_sha
            assert meta["base_seed"] == 20260731
            assert meta["n_rounds"] == 1_000
            assert meta["scenario"] == "classic"
            assert meta["agents"].count(code) == 1
            assert meta["agents"].count("rule_based_agent") == 3
            assert all(
                meta["settings"][key] == value
                for key, value in expected_settings.items()
            )
            assert meta["settings"]["scenario_config"] == {
                "COIN_COUNT": 9,
                "CRATE_DENSITY": 0.75,
            }

            for row in own_rows:
                by_slot[expected_slot][int(row["seed"])].append(row)
            metadata.append(meta)

    return by_slot, metadata


def arena_means(
    by_slot: dict[int, dict[int, list[dict[str, str]]]], column: str
) -> np.ndarray:
    seeds = sorted(by_slot[0])
    return np.asarray(
        [
            np.mean(
                [
                    np.mean(
                        [float(row[column]) for row in by_slot[slot][seed]]
                    )
                    for slot in range(4)
                ]
            )
            for seed in seeds
        ],
        dtype=float,
    )


def bootstrap_ci(
    values: np.ndarray, seed: int = 12345, n_boot: int = 10_000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True)
    return tuple(float(value) for value in np.percentile(draws.mean(1), [2.5, 97.5]))


def signflip_p(
    differences: np.ndarray, seed: int = 0, n_perm: int = 20_000
) -> float:
    rng = np.random.default_rng(seed)
    observed = abs(float(differences.mean()))
    null = (
        rng.choice([-1.0, 1.0], size=(n_perm, len(differences)))
        * differences
    ).mean(1)
    exceedances = int((np.abs(null) >= observed).sum())
    return float((1 + exceedances) / (n_perm + 1))


def is_fragile(differences: np.ndarray) -> bool:
    verdicts = []
    for seed in range(12345, 12350):
        low, high = bootstrap_ci(differences, seed=seed)
        verdicts.append(low > 0 or high < 0)
    return len(set(verdicts)) > 1 or verdicts[0] != (
        signflip_p(differences) < 0.05
    )


def slot_means(
    by_slot: dict[int, dict[int, list[dict[str, str]]]], column: str
) -> list[float]:
    return [
        float(
            np.mean(
                [
                    np.mean([float(row[column]) for row in rows])
                    for rows in by_slot[slot].values()
                ]
            )
        )
        for slot in range(4)
    ]


def maximum_max_max(
    by_slot: dict[int, dict[int, list[dict[str, str]]]], column: str
) -> float:
    return max(
        float(row[column])
        for slot_rows in by_slot.values()
        for arena_rows in slot_rows.values()
        for row in arena_rows
    )


def main() -> None:
    mixed, mixed_meta = load_policy(
        MIXED_STEM_GROUPS,
        "ben_task4",
        ROOT / "agent_code" / "ben_task4"
        / "ben_task4_mixed_kill_v1_2000ep_seed11.pt",
    )
    incumbent, incumbent_meta = load_policy(
        INCUMBENT_STEM_GROUPS,
        "dqn_task3",
        ROOT / "agent_code" / "dqn_task3" / "dqn_task3_seed13.pt",
    )

    mixed_shas = {
        meta["agent_provenance"]["ben_task4"]["model_sha256"]
        for meta in mixed_meta
    }
    incumbent_shas = {
        meta["agent_provenance"]["dqn_task3"]["model_sha256"]
        for meta in incumbent_meta
    }
    print(f"mixed model SHA:    {next(iter(mixed_shas))}")
    print(f"incumbent model SHA: {next(iter(incumbent_shas))}")
    print("unit of analysis: 1,000 arena seeds after equal four-slot averaging")
    print("mixed slot 0: current-callback replication only")
    print("incumbent slot 0: two callback-identical replications averaged")
    print()
    print("metric       mixed  incumbent      diff       95% CI   flip-p  fragile")
    print("-" * 79)

    for metric, column in METRICS.items():
        mixed_values = arena_means(mixed, column)
        incumbent_values = arena_means(incumbent, column)
        differences = mixed_values - incumbent_values
        low, high = bootstrap_ci(differences)
        print(
            f"{metric:10s} {mixed_values.mean():8.4f}"
            f" {incumbent_values.mean():10.4f} {differences.mean():+9.4f}"
            f" [{low:+.4f}, {high:+.4f}]"
            f" {signflip_p(differences):8.4g} {str(is_fragile(differences)):>8s}"
        )

    mixed_slot_scores = slot_means(mixed, "score")
    incumbent_slot_scores = slot_means(incumbent, "score")
    print()
    print("score by list slot")
    for slot in range(4):
        print(
            f"slot {slot}: mixed={mixed_slot_scores[slot]:.4f}, "
            f"incumbent={incumbent_slot_scores[slot]:.4f}, "
            f"difference={mixed_slot_scores[slot] - incumbent_slot_scores[slot]:+.4f}"
        )
    print()
    print(
        "maximum observed think time: "
        f"mixed={maximum_max_max(mixed, 'think_max_ms'):.4f} ms, "
        f"incumbent={maximum_max_max(incumbent, 'think_max_ms'):.4f} ms"
    )


if __name__ == "__main__":
    main()
