# Adversarial audit: Task 3 coin-collector fine-tuning (2026-08-29)

## Scope and audit question

This audit attempts to falsify any claim that the 2,000-episode `coin_collector_finetune_v1` policy is an established improvement over the 5,000-episode `peaceful_v1` policy. It checks evaluation completeness, artifact provenance, metric semantics, comparability, statistical evidence, and limits on the conclusion.

## Evidence inspected

- Complete 1,000-round CSVs and metadata for `peaceful_v1` (`retry2`) and `coin_collector_finetune_v1`.
- The two evaluated model files, the fine-tune episode-2,000 checkpoint, both training CSVs/metadata, and the current Task-3 callback/feature/training selection code.
- `tools/analyze.py --compare ... --preset task3 --markdown` plus independent reconstruction of score and death components from the raw CSVs.
- The bounded current Task-3 section at the top of `BEN.md` and the repository measurement rules.

## Completeness and provenance

- Both evaluation CSVs contain exactly 4,000 rows: 1,000 unique `(round, seed)` pairs and four slots per round. Both metadata files specify `classic`, the same base-seed range `20260731..20261730`, the same settings snapshot, the same dirty commit (`39c770a-dirty`), and the same callback SHA-256 (`567f77...82cc`). Thus feature extraction, action masking, and inference semantics are directly comparable at evaluation time.
- The metadata identifies different intended models and their hashes correctly. The recorded fine-tune SHA-256 `a9bd850...6c8f5` and peaceful SHA-256 `cbf1f05...e95f` match the current files byte-for-byte.
- The fine-tune training CSV has 2,000 episodes, and its final model is tensor-identical to the episode-2,000 checkpoint. Their file hashes differ (`a9bd...` versus `e628...`) only because PyTorch serialization bytes are not a tensor-identity test.
- Training metadata says the fine-tune loaded `ben_task3_peaceful_v1_5000ep_seed11.pt`, used input size 11, fixed epsilon `0.05`, and changed neither architecture nor reward constants. However, it records only the starting filename, not its SHA at training start. The currently present source model has the expected peaceful hash, but strict historical start-file identity cannot be proven from metadata alone.

## Independently reconstructed outcomes

For slot 0 (`ben_task3`), the raw means are:

| Metric | peaceful policy | fine-tuned policy | change |
|---|---:|---:|---:|
| official score | 1.848 | 1.801 | -0.047 |
| coins | 1.143 | 1.261 | +0.118 |
| kills | 0.141 | 0.108 | -0.033 |
| suicides | 0.644 | 0.183 | -0.461 |
| killed by opponent | 0.184 | 0.039 | -0.145 |
| total deaths | 0.828 | 0.222 | -0.606 |
| survival | 0.172 | 0.778 | +0.606 |
| crates | 26.173 | 30.791 | +4.618 |

The identities `score = coins + 5*kills` and `died = suicides + killed_by_opponent` hold in the aggregates. The earlier BEN entry's fine-policy values (score `1.801`, kills `0.108`, suicides `18.3%`, opponent deaths `3.9%`, survival `77.8%`) are therefore correctly reconstructed.

## Statistical and pairing challenge

`analyze.py` reports score `-0.047 [-0.220,+0.128]`, so **no score improvement is demonstrated**. This difference is also below the project's empirical approximately `0.12` score noise floor for repeated active-opponent evaluations. The fine-tuned model cannot be called better on the primary metric.

The safety changes are enormous: suicides `-0.461 [-0.500,-0.422]` and survival `+0.606 [+0.572,+0.640]`, both with sign-flip `p=0`. They are far too large to dismiss as the known opponent noise alone and establish that these two concrete evaluated policies behave very differently. The kill decrease is only `-0.033 [-0.064,-0.001]`, `p=0.044`; despite not being marked fragile by the tool, it sits on the boundary and is not trustworthy as a general effect under the active-opponent reproducibility limitation.

The comparison is paired on arenas and slots, **not on opponent trajectories**. Independent raw comparison found `0/1000` rounds with identical opponent outcome vectors between the two runs. This is partly an unavoidable response to a different focal policy, but the provided opponents also use unseeded stdlib randomness. Therefore the paired intervals must not be described as if every run shared the same stochastic opponent realization. In particular, small differences such as score, kills, or invalid actions are unreadable as treatment effects from these two files alone.

## What the evidence does and does not support

Supported narrowly:

- This specific seed-11 fine-tuned policy is dramatically safer in this one 1,000-round evaluation than this specific peaceful policy: both own-bomb and opponent-bomb deaths fall, and survival rises.
- The safety gain did not produce a detectable score gain. More coins are offset by fewer kills.
- Selecting it as a **safer intermediate candidate** is reasonable if suicide/survival is an explicit regression guard, but it is not a primary-score victory.

Not supported:

- A causal claim that exposure to active opponents alone produced the effect with all other training state controlled. The continuation resets replay, optimizer, and target-network state; there is no equal-duration continuation control from the same starting model.
- Robustness over training seeds. Only training seed 11 exists, contrary to the preferred rung-3 multi-seed evidence rule. A long stochastic training run can land at a seed-specific policy.
- Superiority over `coin_collector_agent`: the fine policy's score `1.801` remains below the opponents' approximately `2.8-2.9`, and it averages only `0.108` kills.
- Task-3 completion, tournament readiness, or a justified stop. The primary score is unchanged, kill output fell, and evaluation against `rule_based_agent` is still absent.

## Verdict

The apparent broad conclusion "fine-tuning improved the Task-3 agent" does **not** survive audit because the primary score did not improve and training-seed robustness is unmeasured. The narrower conclusion "the concrete fine-tuned seed-11 policy is substantially safer against coin collectors, without demonstrated score improvement" does survive.

Stopping Task-3 development is not justified. The next experiment should be pre-registered around **score recovery while retaining the large safety gain**, with the current fine policy as baseline and suicides/survival as regression guards. Before any final model claim, use multiple training seeds or explicitly report the user-chosen single-seed limitation; repeat active-opponent evaluation should be treated as confirmation rather than pretending exact stochastic pairing.

## Frozen-agent follow-up

### Artifact and implementation checks

- `agent_code/dqn_task3_coin_collector/dqn_task3_coin_collector_seed11.pt` is byte-identical to the audited development source model: both SHA-256 values are `a9bd850a8b3eb4052b3c56aa87d39235b674e3b01cd5532325e3805e15f6c8f5`.
- Frozen `features.py` and `model.py` are byte-identical to the development versions. The callback diff removes the development arm/model/checkpoint switches and fixes `dqn_task3_coin_collector_seed11.pt`; after that configuration hunk, the inference implementation is unchanged. This justifies exact policy equivalence **for an identical game-state/history input**, not exact equivalence of separately simulated games.
- The model path is local and relative to `__file__`; no absolute path, `results/` reference, `ben_task3` import, checkpoint selector, `BM_TASK3` model switch, MPS/CUDA use, or training artifact dependency was found. Inference explicitly uses CPU, instantiates 11 channels, and loads with `map_location=cpu`.
- The three focused frozen tests pass. They enforce byte-identical model provenance, absence of experiment/checkpoint switches, CPU loading, valid action output, and `(11,17,17)` features.
- One hygiene caveat remains for eventual submission packaging: the agent directory currently includes generated `__pycache__/` files and an empty `logs/dqn_task3_coin_collector.log`. They do not affect inference or the evidence, but a final submission archive should exclude generated cache/log artifacts. This audit did not remove them.

### Frozen evaluation completeness and provenance

- Quick evaluation: 100 rounds / 400 rows, correct frozen model SHA and callback SHA `da9cab...a9b1`, expected seed range and settings.
- Full evaluation: 1,000 rounds / 4,000 rows, exactly 1,000 unique focal `(round,seed)` pairs, correct frozen model/callback hashes, `classic`, seeds `20260731..20261730`, and only `BM_QUIET_LOGS=1` in captured environment. No development-selection environment variable is present.
- Full frozen result: score `1.823` [1.699,1.952], kills `0.121`, suicides `0.170`, killed by opponent `0.045`, total deaths `0.215`, survival `0.785`, coins `1.218`, invalid actions `0.447`. Aggregate score and death identities remain consistent.
- CPU inference is comfortably inside the 500 ms limit: mean per-round `think_mean_ms` averages about `0.187 ms`, observed maximum single-step `think_max_ms` is `14.1702 ms`, and `think_over_limit` sums to zero.

### Attempt to falsify frozen equivalence

The independent frozen and development 1,000-round evaluations have `0/1000` identical full outcome vectors; the 100-round runs have `0/100`. This does **not** reveal a frozen-code defect. The opponent agents use unsynchronized stdlib randomness, and changing that randomness changes subsequent interactive trajectories even when the focal policy is identical. Therefore exact row-for-row replay equivalence cannot be demanded from these separately executed evaluations.

The observed full-evaluation differences are small and all unresolved: frozen minus development score `+0.022 [-0.128,+0.173]`, kills `+0.013 [-0.015,+0.041]`, suicides `-0.013 [-0.044,+0.018]`, survival `+0.007 [-0.027,+0.041]`. They are compatible with the documented opponent noise and give no evidence of a semantic regression. The analyzer's paired presentation should still be read cautiously because opponent trajectories were not synchronized.

### Frozen verdict

No hidden dependency, path/device/channel mismatch, model substitution, incomplete result, metric inconsistency, or runtime problem was found. It is justified to accept `dqn_task3_coin_collector` as the **concrete frozen seed-11 safe Task-3 intermediate** corresponding to the audited fine-tuned model.

The qualifier is essential: freezing establishes faithful packaging, not multi-seed robustness, primary-score improvement, superiority over `coin_collector_agent`, Task-3 completion, or tournament readiness. The frozen policy remains around score `1.82` versus roughly `2.9` for each coin collector and must not be promoted as a final best agent on this evidence.
