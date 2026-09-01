# Pre-run adversarial audit: 12-channel opponent alignment (2026-08-31)

## Scope and hypothesis

The proposed 1,000-episode seed-11 pair trains against three `rule_based_agent` opponents from the same safe policy. `opponent_alignment_zero_v1` receives an always-zero twelfth channel; `opponent_alignment_v1` receives a spatial map of crate/wall-free source cells whose hypothetical bomb blast intersects a current opponent. I tried to invalidate legality, geometry, tensor/state consistency, source conversion, isolation, causal control and the planned acceptance rule.

## ML-rule legality

The feature is legal under the project's machine-learning requirement. It returns a full 17×17 situational relation map, not an action, path, selected target or “best bomb location.” Multiple cells may be active, the channel does not account for travel cost, timing, escape feasibility or whether `BOMB` should be chosen, and the DQN must learn how the spatial pattern relates to actions and score. It is comparable to an engineered danger/opponent channel, not a rule-based policy oracle.

The feature still encodes a fairly strong tactical relation, so the report should describe it precisely and include the zero-channel ablation. A result may support usefulness of this channel; it cannot be generalized to “alignment features” without multiseed replication.

## Geometry and coordinates

- Framework state uses `field[x,y]`; CNN storage uses `[channel,y,x]`. The implementation reads source/opponent positions in `(x,y)` and writes `[source_y,source_x]`, so it is not transposed.
- A source is considered only when `field[source_x,source_y] == 0`; stone and crate cells are excluded as hypothetical bomb sources. Dynamic occupants/bombs do not change `field`, so such cells may still be marked. This is acceptable as a board relation/future opportunity but should not be described as “immediately legal bomb placements.”
- `_blast_coordinates` includes the source and extends up to `BOMB_POWER=3` orthogonally, stopping at stone walls. This matches the actual submitted framework's `items.py:Bomb.get_blast_coords` and the repository specification that stone walls block explosions. The framework does **not** stop at crates, so continuing through crates is correct for this codebase even if one assumes different Bomberman rules.
- The unit geometry test checks horizontal/vertical alignment and stone-wall blocking. A dedicated crate-pass-through assertion would make the non-obvious framework behavior explicit and protect it from an accidental “fix” to different rules.

## State, replay and augmentation consistency

- `act()` injects the selected alignment mode for both visit-count branches and caches exactly the feature tensor used for the action.
- Replay reuses the cached old tensor when possible; fallback old-state extraction and next-state preview both inject the same alignment mode. Thus no 11/12 mismatch exists between action, stored state and next state.
- The network and target network use dynamic `INPUT_CHANNELS`; alignment arms instantiate 12 channels, all older arms remain 11 channels.
- Symmetry augmentation rotates/reflection-transforms every channel identically and transforms movement actions consistently. The alignment map is spatial and needs no channel-specific remapping.
- Zero and enabled arms therefore differ only in values of channel 11, not tensor shape, model architecture or augmentation.

## Converted source provenance

The 12-channel source is derived from `ben_task4_rule_based_continue_control1000_v1_1000ep_seed11.pt` (SHA-256 `c9677540849340e915f74dec7cf9c2d24e9c6c27f9d5132453798035a1f8a9ec`). Converted source SHA-256 is `f0e153902c930770739a29fc5bdf4d755405534f3ca3a46f4bf9b59837cceb65`.

Tensor inspection confirms:

- first convolution changes only from 11 to 12 input channels;
- its first eleven channel weights are exactly equal to the source;
- every new-channel weight is exactly zero;
- every other tensor is exactly equal.

Consequently both 12-channel arms initially produce the same Q-values and policy as the 11-channel source. The zero arm controls for continued training and architectural conversion; only the enabled arm can learn nonzero new-channel weights.

## Controlled configuration and isolation

Both arms load the same converted source, use seed 11, 1,000 episodes, three declared rule-based opponents, identical rewards/features/hyperparameters and separate Task-4 output names. All proposed final models, CSVs, metadata files, log directories and ten checkpoints per arm are currently absent. No Task-3 path is loaded or written.

As in prior arms, `TRAINING_OPPONENTS` is metadata only. The launch commands must visibly pass `ben_task4 rule_based_agent rule_based_agent rule_based_agent` for both and be retained as evidence. Training trajectories are not exactly paired because provided opponents use OS entropy and unseeded stdlib shuffle; the experiment compares training distributions, not identical transitions.

Seed 11 is defensible as a neutral pilot because both treatments start from the pre-existing seed-11 pure-rule-based path, not a source selected for alignment performance. A future multiseed experiment cannot simply change `BM_TASK4_TRAINING_SEED`: the source is currently hardcoded to the converted seed-11 policy. Seeds 12/13 would require matching safe sources, separate 11→12 conversions and source-selection tests.

## Blocking provenance defect found

The runtime network is correctly 12-channel, but training metadata currently writes the literal:

```python
"input_channels": 11
```

instead of `INPUT_CHANNELS`. A completed alignment CSV would therefore falsely claim an 11-channel model. This must be changed to the dynamic value and covered by a test before training. The nine existing tests pass under each actual arm with `TOTAL_EPISODES=1000`, but they do not assert this metadata value, so passing tests do not catch the defect.

## Computational cost

The enabled channel evaluates at most 289 source cells × four rays × three tiles per feature extraction. A local microbenchmark measured approximately `0.48 ms` per enabled extraction versus `0.02 ms` for the zero control. This is a material relative increase but tiny against 500 ms; final evaluation must still measure real end-to-end inference on CPU and reference-hardware risk. The nested Python loops are bounded and contain no path search or multiprocessing.

## Pre-registered pilot criteria

Evaluate both end models explicitly as `trained` against the same three-rule-based field. Use 100 rounds only as a gate, followed by 1,000 rounds for any claim. The direct contrast is `alignment enabled - zero channel`:

1. **Score (primary):** positive difference, bootstrap 95% CI excluding zero, sign-flip `p<0.05`, no fragile flag, and magnitude greater than the roughly `0.12` Task-4 opponent-noise heuristic.
2. **Kills (mechanism):** positive, non-fragile difference with CI excluding zero. Score without kills can identify a candidate but does not validate the intended attack-alignment mechanism.
3. **Safety guards:** no demonstrated worsening and practical point margins no worse than `+0.03` suicides, `+0.02` killed-by-opponent and `-0.03` survival relative to the zero control.
4. **Timing:** zero `think_over_limit`; global observed maximum below 500 ms and preferably below 50 ms on the development machine. Report the actual global maximum, not only the analyzer's mean of per-round maxima.

One seed is pilot screening only. A pass permits correctly source-mapped seeds 12/13; it does not permit immediate frozen replacement or report-level attribution. Do not search the ten checkpoints after seeing the endpoint unless a bounded selection rule is registered before evaluation.

## Verdict before launch

The feature is legal, geometrically faithful to this framework, internally consistent and paired with an unusually strong zero-channel control. The design should not launch yet because its metadata would incorrectly record `input_channels=11`. After changing that field to `INPUT_CHANNELS`, adding a regression assertion, rerunning both actual arm configurations, confirming free output paths and using explicit identical CLI lineups, the 1,000-episode seed-11 pilot is justified.

## Post-run audit: enabled channel versus zero placebo

### Artifacts and provenance

Both arms contain exactly 1,000 unique training rows numbered 1–1,000, ten checkpoints at 100-episode intervals, and final models tensor-identical to Episode 1,000. Both load the same audited 12-channel source, declare three rule-based opponents and record the same rewards/hyperparameters. The metadata defect was fixed: both training metadata files correctly record `input_channels: 12` and distinguish `opponent_alignment_mode: zero` from `enabled`.

Final model hashes:

- zero placebo: `028884dc74703f85cf77dfb1d897994ecc37f2ebf8bcc2d118e6bdcb0b3e45df`;
- enabled alignment: `68966f6806aa376a403873afe61048cee4b1f5a5764897ebb35681add84f0718`.

Both quick evaluations contain 400 rows/100 complete rounds and both full evaluations contain 4,000 rows/1,000 complete rounds. Each uses slot-0 `ben_task4` plus three rule-based opponents, explicitly selects the correct trained arm, and hashes the matching final model. Score and death-cause arithmetic hold in every row. The callbacks hash is identical, so inference code differs only through the recorded arm/mode and selected model.

### Direct result

The comparison is correctly oriented as `enabled alignment - zero placebo`:

| metric | zero | enabled | difference (95% CI) | sign-flip p | verdict |
|---|---:|---:|---:|---:|---|
| score | 3.388 | 3.320 | `-0.068 [-0.245,+0.112]` | 0.461 | no effect shown |
| kills | 0.116 | 0.114 | `-0.002 [-0.031,+0.027]` | 0.946 | no effect shown |
| suicides | 0.325 | 0.263 | `-0.062 [-0.101,-0.023]` | 0.0021 | better |
| killed by opponent | 0.088 | 0.072 | `-0.016 [-0.039,+0.008]` | 0.211 | no effect shown |
| survival | 0.587 | 0.665 | `+0.078 [+0.036,+0.119]` | 0.0003 | better |
| win rate (secondary) | 0.349 | 0.368 | `+0.019 [-0.021,+0.060]` | 0.381 | no effect shown |

No result is fragile; sign-flip and bootstrap verdicts agree. Opponent stdlib randomness means pairing is on arenas/list slot, not exact trajectories, and this remains one training seed. That caveat does not convert score/kills into a positive result, nor plausibly explain away safety changes of this size and consistency.

The quick screens are not used as evidence, but they do not contradict the final direction: score/kills were similar while the enabled policy survived more often (`0.63` versus `0.70` was noisy in the opposite direction for survival at n=100, illustrating why the 1,000 run is decisive rather than allowing quick-screen selection).

### Pre-registered gates

The intended attack feature fails both necessary gates:

1. score is nominally `-0.068`, not positive, its CI includes zero and it is below the practical `0.12` noise threshold;
2. kills are essentially identical (`-0.002`) with a wide zero-containing CI.

Therefore the experiment provides no evidence for improved bomb alignment as an aggression mechanism. It must not replace the existing policy or advance to seeds 12/13 under the original hypothesis.

All safety guards pass. Suicides improve by `-0.062`, killed-by is nominally lower by `-0.016`, and survival improves by `+0.078`; none approaches a harmful practical margin. The channel is thus worth retaining as an observed **safety component**, with the restricted claim that this concrete seed-11 policy used the relation map more safely. It is not yet a multiseed safety result and safety does not substitute for primary score.

### Timing

No action exceeds 500 ms. Mean action time rises from `0.193 ms` to `0.469 ms`, consistent with the extra bounded Python geometry work. Analyzer mean per-round maxima are `0.915` versus `1.179 ms`, with no demonstrated difference. Actual global maxima are `62.841 ms` (zero) and `65.218 ms` (enabled).

Both global outliers exceed the preferred 50 ms development margin, including the zero control which does not compute alignment. They remain 7.7× below the official 500 ms limit and are likely runtime/system outliers rather than normal feature cost, but reference-hardware or Docker timing is required before submission. Do not report the analyzer's `1.2 ms` as the global maximum.

### Decision and next experiment

Reject `opponent_alignment_v1` as an attack arm and do not run unchanged seeds 12/13. Preserve the artifacts as a negative score/kill ablation and a promising safety observation. Do not inspect its ten checkpoints now; no checkpoint rule was registered and the endpoint already answers the planned question.

A scientifically motivated next experiment can combine the two independently observed levers rather than repeating either unchanged: peaceful exposure increased kills/score but destroyed safety, while alignment reduced suicides/survival loss without increasing attack. Test a small, pre-specified mixed-opponent dose with a **12-channel zero placebo versus enabled alignment**, both from the same converted safe source and with identical mixed lineup/rewards/length. The question becomes whether alignment moderates the already demonstrated mixed-training safety cost while preserving its aggression signal. Start as a bounded seed-11 pilot (preferably shorter than the failed 2,000-episode mixed dose), retain the same score/kill requirements and safety guards, and proceed to matched-source multiseed replication only if both sides pass. This is an interaction test; it should not reuse or select existing alignment checkpoints post hoc.

## Short pre-run audit: mixed alignment interaction

The implemented `opponent_alignment_mixed_zero_v1` and `opponent_alignment_mixed_v1` pair is correctly controlled for a 1,000-episode seed-11 pilot:

- both load the exact same converted source `ben_task4_safe_seed11_12ch_alignment_source.pt`, SHA-256 `f0e153902c930770739a29fc5bdf4d755405534f3ca3a46f4bf9b59837cceb65`;
- both instantiate 12-channel online/target networks and use the same action, replay-old, replay-next and augmentation paths already audited;
- rewards, potentials, optimizer, epsilon, all feature channels 0–10, seed, duration and declared lineup `peaceful_agent,rule_based_agent,rule_based_agent` are identical;
- the only configured treatment is channel 11: always zero in the placebo and geometrically enabled in the candidate;
- model, CSV, metadata, log and ten checkpoint names are arm-specific and all are currently free under Task-4 paths;
- training metadata uses dynamic `input_channels=12` and records the channel mode;
- all ten tests pass under each actual arm with `BM_TASK4_TOTAL_EPISODES=1000`; the subprocess pair test explicitly verifies equal source, channel count, rewards and lineup with only mode differing.

The standard launcher caveat remains: `TRAINING_OPPONENTS` records intent but does not enforce the world. Both CLI commands must explicitly list `ben_task4 peaceful_agent rule_based_agent rule_based_agent` in the same order. Final greedy evaluation must use three rule-based opponents, not the easier training mixture, because the hypothesis is that alignment preserves mixed-learned aggression under the Task-4 field.

Pre-registered direct contrast is `enabled - zero`: score must be positive, non-fragile, CI exclude zero and exceed the practical `0.12` noise threshold; kills must be positive/non-fragile with CI excluding zero. Safety guards are point changes no worse than `+0.03` suicides, `+0.02` killed-by and `-0.03` survival, with no demonstrated regression. Timing requires zero 500-ms overruns and actual global maximum below 500 ms, preferably below 50 ms.

This is only a seed-11 interaction screen. Passing permits matched-source multiseed work; failing score/kills or any safety guard stops the arm. The implementation is ready to launch after the usual all-target collision check immediately before execution.
