# Pre-run adversarial audit: Task-4 `suicide7` design (2026-08-30)

## Scope

This audit tries to invalidate the proposed 5,000-episode seed-11 fine-tune against three `rule_based_agent` opponents before compute is spent. I inspected every file in `agent_code/ben_task4`, compared it with `ben_task3`, checked the copied model, paths, reward semantics, metadata fields and tests, and separated a useful candidate search from a causal reward experiment.

## What is correct

- `ben_task4_task3_baseline_seed13.pt` is byte-identical to frozen `dqn_task3_seed13.pt`, SHA-256 `1285ac5cb78a25b0e6cc6a0e0a68fdd86e153db537832938f940cb1ad4fd8b63`.
- Feature extraction, model architecture, augmentation, replay and DQN optimization are semantically inherited. The differing hashes in augmentation/features/model come only from Task-3→Task-4 docstrings; `dqn.py` and `replay_buffer.py` are byte-identical.
- Runtime configuration uses only `BM_TASK4_*`. Output names are Task-4-specific: model `agent_code/ben_task4/ben_task4_rule_based_suicide7_v1_5000ep_seed11.pt`, logger files under `results/train/ben_task4`, and checkpoints `results/train/ben_task4/task4_rule_based_suicide7_v1_5000ep_seed11__episode_<N>.pt`. Evaluation-only checkpoint selection also points only to that directory.
- The intended model, CSV, metadata, log directory and all fifty 100-step checkpoint targets are currently absent. No collision was found.
- The death semantics are deliberate and correct: every death fires `GOT_KILLED=-5`; an own-bomb death additionally fires `KILLED_SELF=-2`. Therefore opponent-caused death is `-5` and suicide is `-7` (plus the universal step reward `-0.05` on that stored terminal transition). The lethal transition is stored in `end_of_round`, so the added penalty reaches replay.
- Other inherited reward/hyperparameter settings remain unchanged: coin `+1.5`, crate `+0.3`, kill `+5`, invalid `-1`, step `-0.05`, safety potential `1`, crate WAIT penalty `-0.03`, epsilon fixed at `0.05`, learning rate `1e-4`, replay/target-update settings and eleven channels.
- Training metadata records the two death rewards separately, source/output model files, seed, planned episodes, arm, stated opponent field, features, shaping and optimizer settings. Five scaffold tests pass.

## What can break the experimental claim

### 1. It is not a one-variable comparison to the frozen baseline

Relative to frozen `dqn_task3`, the proposed run changes at least three things simultaneously:

1. learning continues for 5,000 episodes;
2. the training field becomes three `rule_based_agent` opponents;
3. the extra suicide reward changes from `0` to `-2`.

The code contains a useful `baseline_v1` arm which loads the same start model and leaves `KILLED_SELF=0`, but no equal-length control run is currently planned. Without that control, a resulting policy may be a useful candidate, but improvement or regression cannot be attributed specifically to `-7` suicide pricing. The proper causal comparison is `baseline_v1` versus `rule_based_suicide7_v1`, trained for the same length, training seed and opponent lineup. `TRAINING_OPPONENTS` is only metadata; it does not enforce the CLI lineup, so the launcher itself must pass exactly three rule-based opponents for both arms.

### 2. A single training seed cannot support a Task-4 conclusion

Seed 11 over 5,000 episodes is reasonable as a **first screening run**: it can reveal catastrophic forgetting or a promising direction before paying for more runs. It is not an accepted Task-4 effect under the repository rules. Rule-based opponents reseed NumPy from OS entropy and also use unseeded stdlib `shuffle`; consequently even repeating seed 11 does not reproduce the same training experience. A surviving arm must be repeated with at least seeds 12 and 13 before a report-level improvement, stopping decision or frozen replacement.

Five thousand episodes is defensible for the pilot because the source policy is already competent and epsilon is only `0.05`, but the final checkpoint must not be assumed best. Any checkpoint inspection must be pre-bounded before seeing greedy results; otherwise it becomes checkpoint fishing.

### 3. Isolation is good, but tests do not prove the CLI or completed metadata

The unit tests prove local model identity, default path strings, CPU inference and reward arithmetic. They do not prove that the eventual command uses three rule-based opponents, that all output targets were checked immediately before launch, that a completed CSV has 5,000 rows, or that the final model equals checkpoint 5,000. Those are required post-run checks. `MODEL_VARIANT=auto` may silently select a newly created final model during evaluation; reported evaluations should explicitly set `BM_TASK4_MODEL_VARIANT=baseline` or `trained`.

## Pre-registered interpretation and success criteria

The primary metric remains **official score**. Lower deaths alone cannot make the arm an improvement.

For the seed-11 pilot:

1. First run a 100-round greedy screen against three rule-based opponents. Stop the arm if score collapses or any safety metric is grossly worse; the screen is only a gate, not evidence.
2. If it passes, evaluate 1,000 rounds at the fixed default evaluation seed and compare with the frozen baseline in the same list slot. A candidate improvement requires score difference `>0`, bootstrap 95% CI excluding zero, sign-flip `p<0.05`, and no `(fragile)` flag. Because Task-4 opponent noise has an observed roughly `0.12` score floor, a smaller single-run difference is not practically interpretable even if nominally favorable.
3. The suicide hypothesis succeeds only if `suicides` falls from the frozen slot-0 level (about `0.474–0.478`) with a non-fragile paired CI excluding zero. A score gain without this reduction may still identify a candidate, but does not validate the proposed reward mechanism.
4. Regression guards: `killed_by_opponent` must not show a non-fragile increase from roughly `0.12–0.13`, and survival must not show a non-fragile decrease from roughly `0.39–0.40`. Preferably killed-by falls and survival rises; report both regardless.
5. Timing: zero `think_over_limit`, observed global `think_max_ms <500`; use `<50 ms` as a practical safety margin on the development machine, while recognizing reference hardware is slower.

For an actual arm conclusion, apply the same evaluation to seeds 11/12/13 and analyze the training seed—not 3,000 evaluation rounds—as the replication unit. Require the mean score improvement to survive across seeds, with suicide reduction consistent rather than driven by one seed. If the scientific claim is specifically that `KILLED_SELF=-2` caused the improvement, train the equal-length `baseline_v1` control across the same seeds and opponent field. Without this control, phrase the result only as “Task-4 rule-based fine-tuning with suicide `-7` produced this policy,” not as an isolated reward effect.

## Verdict before launch

No path, model, reward-semantics or isolation defect blocks the run. The design is safe to launch as a **seed-11 pilot candidate search**, provided collision checks and the exact three-rule-based CLI are used. It is not yet a controlled reward experiment, and one 5,000-episode seed must not be used to replace the frozen baseline, claim Task-4 improvement, or decide to stop. The strongest design would add an equal-length `baseline_v1` control before attributing any outcome to the extra `-2` suicide penalty.

## Post-run audit: seed-11 control versus `suicide7`

### Completeness and provenance

Both arms contain exactly 5,000 unique training rows numbered 1–5,000, exactly 50 checkpoints at 100-episode intervals, a final model and Episode-5,000 checkpoint with tensor-identical state dictionaries. Their serialized file hashes differ because separate `torch.save` calls need not be byte-deterministic; tensor identity is the relevant check.

- `baseline_v1` final model SHA: `ea27ab3d483298e5c56dfa93b97e2a6dc3453ed0bdb6bff79e2a605abf7457ab`.
- `rule_based_suicide7_v1` final model SHA: `ab0e911523d36851046ace2f87aad436ab955c0ec05c530689f2c4f3fa3d3817`.

Training metadata confirms the same source model, seed 11, 5,000 episodes and all inherited hyperparameters. The only recorded reward difference is `killed_self_reward: 0` versus `-2`; `GOT_KILLED` remains `-5`. The suicide arm records the expected three rule-based opponents. The control metadata says `configured_by_main_cli`, so the artifact alone does not prove its actual CLI opponent lineup; this remains a provenance weakness even though the experiment was launched as the intended rule-based control. Neither static metadata string enforces actual opponents.

Each greedy evaluation contains 4,000 rows, 1,000 complete rounds and one `ben_task4` plus three rule-based observations per round. Our agent is slot 0 in both. Metadata selects the correct arm with `BM_TASK4_MODEL_VARIANT=trained`, and its model SHA exactly matches the corresponding final model. All rows satisfy official score and death-cause arithmetic. Both runs use the prescribed arena seeds and settings.

### Direct reward-control result

The analysis direction is `suicide7 - baseline`, so negative suicides would favor the extra penalty while positive score would favor it on the primary metric:

| metric | control | suicide7 | difference (95% CI) | sign-flip p | result |
|---|---:|---:|---:|---:|---|
| score | 3.408 | 3.290 | `-0.118 [-0.301,+0.069]` | 0.213 | no effect shown |
| kills | 0.108 | 0.101 | `-0.007 [-0.036,+0.022]` | 0.688 | no effect shown |
| suicides | 0.274 | 0.265 | `-0.009 [-0.048,+0.029]` | 0.684 | no effect shown |
| killed by opponent | 0.070 | 0.055 | `-0.015 [-0.035,+0.006]` | 0.190 | no effect shown |
| survival | 0.656 | 0.680 | `+0.024 [-0.016,+0.064]` | 0.261 | no effect shown |
| win rate (secondary) | 0.385 | 0.356 | `-0.029 [-0.068,+0.011]` | 0.165 | no effect shown |

No row is fragile and every sign-flip test agrees with the bootstrap verdict. The observed score difference is almost exactly the known approximately `0.12` Task-4 opponent-noise floor and its CI spans both practically relevant benefit and harm. Only 143/1,000 DQN outcome vectors match across runs, so equal evaluation seeds do not make opponent trajectories paired; stdlib `shuffle` remains unseeded. Training is even less pairable: opponent randomness differs and `main.py` does not reseed arenas per episode, so different round lengths cause later arena sequences to drift.

Timing is safe: zero over-limit steps, global DQN maxima `15.639 ms` (control) and `26.285 ms` (`suicide7`), both within the pre-registered 50 ms development margin and far below 500 ms.

### Attempt to rescue a useful `-2` effect

The point estimates lean slightly toward fewer deaths under `suicide7`, but the intended diagnostic—suicides—moves by only `-0.009` and its CI spans `-0.048` to `+0.029`. Thus the extra reward fails its pre-registered mechanism criterion. It also fails the primary score criterion: score is nominally lower, not higher, and no improvement is demonstrated. The data do **not** prove that `-2` has exactly zero population effect or is universally harmful; they show that this concrete seed-11 arm produced no useful demonstrated incremental effect over equal-length ordinary fine-tuning.

Running seeds 12/13 specifically to rescue `suicide7` is therefore not justified by the pilot gate. Multiseed replication is required for a positive Task-4 conclusion, but it need not be spent on an arm that showed neither its intended suicide signal nor primary score benefit. Rejecting this concrete reward arm is a screening decision, not a universal statement about all suicide penalties.

### What did show a useful signal

Both trained policies are much safer than the original frozen Task-3 baseline. For the no-extra-penalty control versus the original slot-0 baseline:

- score `+0.160 [-0.035,+0.350]` (not demonstrated);
- suicides `-0.200 [-0.243,-0.157]`;
- killed by opponents `-0.062 [-0.089,-0.036]`;
- survival `+0.262 [+0.218,+0.305]`.

The `suicide7` policy shows similar safety gains, which is why comparing it only to frozen Task 3 would misleadingly credit the new reward. The control demonstrates that ordinary 5,000-episode rule-based fine-tuning already supplies the dominant safety improvement. It does not yet demonstrate higher primary score, and kills decline nominally, so it is a promising safety-preserving direction rather than a finished Task-4 winner.

### Updated decision

Do not advance `rule_based_suicide7_v1` to more seeds. Retain all artifacts as a negative controlled result. If compute is spent on seeds 12 and 13, the better-supported next step is the unchanged-reward `baseline_v1` fine-tune, because it produced the large relevant safety signal without the extra reward and did not show a score loss. Final acceptance still requires a multiseed score analysis under Task-4 rules; safety metrics remain diagnostics and cannot substitute for score. Alternatively, a new arm should target the remaining clear weakness—kills below the rule-based field—while guarding the control's safety gains, rather than increasing the suicide penalty again.

## Final multiseed audit: unchanged-reward `baseline_v1`

### Artifact audit

Seeds 11, 12 and 13 are all complete. Each has exactly 5,000 unique training rows numbered 1–5,000, exactly 50 checkpoints at 100-episode intervals, and a final model tensor-identical to its Episode-5,000 checkpoint. Training metadata identifies `baseline_v1`, the correct training seed, the same local frozen Task-3 source, `KILLED_SELF=0`, `GOT_KILLED=-5` and otherwise identical hyperparameters.

Final model hashes are:

- seed 11: `ea27ab3d483298e5c56dfa93b97e2a6dc3453ed0bdb6bff79e2a605abf7457ab`;
- seed 12: `499787df4b57f344377da5d5cd264528447d2c914655c41cb3a1eceaa0f6225e`;
- seed 13: `91e71b114e8482d03ef5026c67c262c43927e5b26d910b9741893254c8a4de19`.

Every evaluation contains 4,000 rows, 1,000 distinct rounds and exactly one DQN plus three rule-based rows per round. The DQN is list slot 0 in all three; each meta file selects `MODEL_VARIANT=trained`, carries the correct arm/seed and hashes the matching final model. Official score and death-cause identities hold in every row. No timeout occurs; global DQN maxima are `15.639/18.613/47.929 ms` for seeds 11/12/13, within the pre-registered 50 ms development margin and far below 500 ms.

The earlier metadata limitation remains: the control arm labels training opponents as `configured_by_main_cli`, so its CSV/meta cannot independently prove the CLI lineup. The evaluation lineups are fully recorded and correct.

### Per-seed greedy results

| training seed | score | coins | kills | suicides | killed by opponent | survived | win rate |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 3.408 | 2.868 | 0.108 | 0.274 | 0.070 | 0.656 | 0.385 |
| 12 | 3.055 | 2.600 | 0.091 | 0.296 | 0.074 | 0.630 | 0.314 |
| 13 | 3.411 | 2.781 | 0.126 | 0.348 | 0.073 | 0.579 | 0.365 |
| equal seed mean | **3.291** | **2.750** | **0.108** | **0.306** | **0.072** | **0.622** | **0.355** |

Within each run, DQN score minus the per-round mean of the three rule-based agents is positive and non-fragile:

- seed 11: `+0.617 [+0.433,+0.799]`;
- seed 12: `+0.181 [+0.009,+0.355]`;
- seed 13: `+0.610 [+0.432,+0.789]`.

An arena-clustered, equally seed-weighted within-field estimate is `+0.469 [+0.351,+0.585]`. This confirms that every trained policy outscores its contemporary rule-based field. It is not by itself improvement over frozen Task 3, because the frozen model already did so position-robustly.

### Correct comparison with frozen Task 3

For list-slot comparability I averaged the two independent frozen slot-0 evaluations before comparing training-seed means. The frozen reference is score `3.329`, kills `0.145`, suicides `0.476`, killed-by `0.126`, survival `0.398`. The position-balanced frozen score is approximately `3.333`, essentially the same reference, so list-slot bias cannot rescue the conclusion.

The three seed-level changes versus the averaged frozen slot-0 reference are:

| metric | seed 11 | seed 12 | seed 13 | equal-seed mean |
|---|---:|---:|---:|---:|
| score | +0.080 | -0.274 | +0.083 | **-0.037** |
| kills | -0.037 | -0.054 | -0.019 | **-0.036** |
| suicides | -0.202 | -0.180 | -0.128 | **-0.170** |
| killed by opponent | -0.056 | -0.052 | -0.053 | **-0.054** |
| survived | +0.258 | +0.232 | +0.181 | **+0.224** |

The seed is the replication unit for the training intervention. With only three seeds, a two-sided sign-flip test cannot attain `p<0.05` even when all three changes share a sign (minimum approximately `0.25`). The safety directions are nevertheless strikingly consistent and far larger than evaluation noise; the score direction is not. Treating all 3,000 evaluation rounds as independent would create false precision because they reuse the same 1,000 arena seeds and do not represent 3,000 training replications.

Opponent behavior is still only partly reproducible because stdlib `shuffle` is unseeded. Arena pairing is exact; opponent trajectories are not. The known approximately `±0.12` score noise floor reinforces the result: seed 11 and 13 score changes are smaller than it, while seed 12's negative change drives the family mean. No scientifically defensible score improvement exists.

### Attempted conclusion and what survives

The broad statement “`baseline_v1` is a Task-4 improvement” fails under the project's primary-metric rule. Equal-seed mean score is slightly lower than frozen, and kills are lower for all three seeds. The family therefore does **not** demonstrate Task-4 progress in official score or aggression.

A narrower, useful conclusion survives: unchanged-reward rule-based fine-tuning robustly learns a much safer policy family—fewer own-bomb deaths, fewer opponent-bomb deaths and substantially higher survival—while mean score stays approximately level within observed variability. This is a valuable intermediate result and a better source for a future combat arm, but safety cannot replace score as the acceptance metric.

### Model selection and next step

Selecting seed 11 or 13 now because each scored about `3.41` would be post-hoc selection on the same Task-4 measurement. No single seed may be called the family winner from these files. Options that preserve scientific validity are:

1. retain the family and use seed 11 only as the neutral first training seed, explicitly not because of its measured score; or
2. pre-register an independent selector in a new opponent context (for example a heterogeneous field), evaluate all three identically, and reserve a later field for confirmation.

Further unchanged fine-tuning is not the best next experiment: it already traded kills for safety without raising family score. The next controlled arm should start from a pre-selected safe policy and target the missing kill/score behavior while guarding the gains (suicides, killed-by, survival and timing). It must be tested against an equal-length unchanged-reward control and across several training seeds before replacing frozen `dqn_task3`. The current frozen Task-3 agent remains the demonstrated primary-score baseline; `baseline_v1` remains the demonstrated safety family, not a new final Task-4 agent.
