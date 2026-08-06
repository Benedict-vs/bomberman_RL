# Task 1 (`coin-heaven`) — what Benedict's agent ended up being, and what it taught us

Companion to the blow-by-blow ledger in `experiments/benedict.md` (E01–E07).
This file is the summary: the final specification, the results, and the lessons that
should outlive task 1. Written so it can be diffed against Maxi's and Ben's coin
collectors.

---

## 1 · Final specification

| | |
|---|---|
| **Model** | Tabular Q-learning, dense `np.ndarray` of shape `(784, 6)` |
| **State** | 4 wall bits (U/R/D/L, `field != 0`) + `dx`, `dy` to the nearest coin, clipped to ±3 |
| **Encoding** | Mixed radix, `FEATURE_SIZES = (2,2,2,2,7,7)` → one integer row index |
| **Reachable rows** | 295 of 784 (enumerated over all tile × coin pairs) |
| **Actions** | All six. `BOMB`/`WAIT` are **not** masked |
| **Policy** | Deterministic `argmax`, ε = 0 outside training |
| **Rewards** | coin +5 · `INVALID_ACTION` −1 · `WAITED` −0.1 · `KILLED_SELF` −5 · step −0.1 |
| **α** | **per cell**, `1 / N(s,a)^0.7` |
| **γ** | 0.9 |
| **ε** | 0.2 → 0.02, ×0.9995 per episode |
| **Training** | 10 000 rounds, world seed 810731 (≠ evaluation seed), 5 seeds per configuration |
| **Shipped model** | run index 0, declared before the results were seen |

Distance to the nearest coin is **Manhattan**, not BFS — deliberately, so that the
distance metric and the direction encoding remain independently swappable.

## 2 · Results (5 seeds × 300 rounds, `coin-heaven`, seed 20260731)

| Metric | Value |
|---|---|
| coins (of 50) | **49.15 ± 1.23**, worst seed 47.16 |
| rounds collecting all 50 | ~98 % |
| completion time, full sweeps | median 129 steps |
| survival | 100 % |
| invalid actions per round | 0.055 |
| `think_max_ms` | 0.1 (limit 500) |

Progression: 1.35 → 12.8 → 16.7 → 46.1 → 49.9 → **49.15 ± 1.23**. The last step *lowers*
the headline number, because it is the first one measured over five seeds instead of one.

## 3 · Lessons about method

These cost the most time and generalise beyond this project.

1. **A single training run is not a result.** Two runs of *identical code* differed by
   17 coins (E05b). Reported means from n = 1 were wrong twice: E05's +3.8 vanished under
   replication, and its explanation turned out to be wrong as well (E07). Everything from
   E06 on uses n = 5 and reports the spread and the **worst** seed.
2. **A training curve is not a result either.** At ε = 0.02 a random action every ~50 steps
   knocks the policy out of whatever cycle it is stuck in, so a deadlock-prone table trains
   at 44.8 coins/episode and evaluates at 11.1. Nothing counts until `tools/evaluate.py`
   has run at ε = 0. (Maxi reached the same conclusion independently from a different
   failure — worth stating once in the report with both pieces of evidence.)
3. **Aliasing and undertraining look identical in a Q-table.** Both leave a row whose
   values do not separate. E04 diagnosed one as the other and concluded a feature was too
   coarse; E05 disproved it without touching the feature. Check whether the states were
   ever *visited* before blaming the representation.
4. **Write the prediction down, including what would refute it.** E07's decisive finding
   came from a branch written before the run ("if training steps stay near 61 while
   evaluation still reaches ~49, the mechanism story is wrong"). Without it the same
   numbers would have read as a shrug.
5. **Replaying the greedy policy beats a 300-round evaluation for diagnosis.**
   `scratchpad/benedict/loop_check.py` reconstructs the deterministic policy and detects
   absorbing cycles in seconds. It found the real story in E04 and E06 faster and more
   precisely than the evaluation did.
6. **Reproducibility needs two seeds.** The agent's exploration RNG *and* `main.py --seed`.
   Either alone leaves runs incomparable. Training must not use the evaluation seed.

## 4 · Lessons about the agent

1. **The learning rate was the single largest effect.** Constant α = 0.1: 20.78 ± 6.98
   coins. Per-cell `1/N^0.7`: 49.15 ± 1.23. L26's conditions (`sum α = ∞`, `sum α² < ∞`)
   are not a formality — a constant α leaves high-traffic cells jittering, and here an
   unsettled cell is not a small error but an **absorbing deadlock**: the wrong action wins
   by 0.02, and if it is a move into a wall the state never changes, so the agent repeats it
   forever. One bad cell out of 1770 cost 17 coins and 163 wasted actions per round.
2. **A deterministic memoryless policy is eventually periodic — the question is only
   whether the cycle is reached before or after the coins are gone.** Every configuration
   loops; good ones loop at step ~135 with 50 coins collected, bad ones at step 25 with 10.
3. **`KILLED_SELF = −5` removed every suicide at once.** Without it the only pressure
   against `BOMB` is that the episode ends and the terminal update carries no bootstrap
   term — worth about 0.02 once the coin reward inflates a row. This is why we did **not**
   need to mask `BOMB` out of the action space.
4. **Credit for a delayed hazard only fails to propagate if the agent leaves the hazard.**
   Q-learning bootstraps with `max`, so a penalty cannot flow back through states that look
   safe. But the agent stood still and re-issued `BOMB` until it died, which made the credit
   assignment direct. Expect the opposite on task 2, where escaping is the point.
5. **A good state abstraction makes "phases" of a round disappear.** φ is position-relative
   with no notion of time or coins remaining, so the late round is not a distinct region of
   state space. The agent trains almost exclusively in the first 70 steps and plays fine at
   step 300. This is why the ε schedule earns nothing here — and why it should matter on
   task 2, where danger states *are* a distinct region reachable only by surviving.
6. **Feature saturation is survivable.** Beyond ±3 the offset degrades to a bare direction,
   which is still learnable — the residual failures were undertraining, not the clip.

## 5 · Refuted along the way (kept deliberately)

| Claim | Status |
|---|---|
| ε decay improves task 1 (+3.8 coins, E05) | **Refuted.** Lucky draw; 5 seeds show no demonstrated difference (E07) |
| ...because training never reaches the late round | **Refuted.** Constant ε trains at 68–72 steps and still evaluates at 48.7 |
| The unfinished rounds are the ±3 clip saturating (E04) | **Refuted.** Those states were unvisited, not unlearnable |
| More training data helps (E05b control) | **Inconclusive** — swamped by run-to-run variance |
| The deadlock is always a blocked-direction argmax (E05b) | **Too narrow.** Usually a 2-cycle between two tiles |

## 6 · Carried into task 2

- Per-cell α, n = 5 seeds, prediction-before-run, `loop_check` replay.
- The ε schedule — *not* justified by task-1 numbers, kept because danger states are a
  distinct region of the state space that only a surviving agent reaches.
- **Known defect:** offset `(0, 0)` means both "no coins left" and "a coin is on my own
  tile". Cost 3 of 300 rounds on task 1; must be separated on task 2, where the coin list
  is empty for long stretches.
- **Overdue:** ablate `COIN_COLLECTED` +5 against the game's actual +1. Open since E01.
- **Needed:** a danger feature (in blast radius / steps to detonation / is there an escape),
  and `CRATE_DESTROYED` in the reward table.

## 7 · Points of comparison with the other two

Not better or worse — different, and the differences are the interesting part of the report.

| | Benedict | Maxi |
|---|---|---|
| Q storage | dense array + mixed-radix index | dict keyed by feature tuple |
| Coin feature | clipped offset (±3), Manhattan | BFS-derived direction |
| Action set | all six | `BOMB`/`WAIT` masked on rung 1 |
| Suicides | solved by reward (`KILLED_SELF` −5) | solved by removing the action |
| Tie-breaking | none — deterministic argmax | random among the best actions |
| Deadlock fix | make the values converge (per-cell α) | randomise ties |

The last two rows are the same problem solved two ways, which is worth a paragraph:
random tie-breaking hides an unconverged table, a converging learning rate removes the
reason ties decide anything.

## 8 · Reproducing the final model

```bash
# five seeds, in parallel; each writes its own table
for i in 0 1 2 3 4; do
  ( export BM_MODEL_SUFFIX="_s${i}"
    BM_RUN_INDEX=$i uv run python main.py play --no-gui \
      --agents benedict_coin_collector --train 1 --scenario coin-heaven \
      --n-rounds 10000 --seed 810731 >/dev/null 2>&1 ) &
done
wait
cp agent_code/benedict_coin_collector/q_table_s0.npy \
   agent_code/benedict_coin_collector/q_table.npy
rm -f agent_code/benedict_coin_collector/q_table_s*.npy

uv run python tools/evaluate.py --agents benedict_coin_collector --opponents none \
  --scenario coin-heaven --n-rounds 300 --label benedict_final__task1
```

Defaults in `train.py` are the task-1 settings (`BM_ALPHA=visit`, `BM_EPS=decay`).
