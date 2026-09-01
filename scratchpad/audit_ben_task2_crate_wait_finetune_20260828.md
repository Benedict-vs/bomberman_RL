# Adversarial audit: crate-WAIT fine-tune (2026-08-28)

## Audit question and standard

Claim under attack: `ben_task2_escape_crate_wait002_finetune2000_from10000_v1_seed11.pt` is a credible, large Task-2 DQN improvement over `ben_task2_escape_multiseed_reachable_v1_10000ep_seed11.pt`, caused specifically by a targeted safe-crate-WAIT penalty.

Standard: filenames and aggregate improvements are not causal evidence. The claim requires artifact identity/provenance, a genuinely paired evaluation, correct trigger semantics, and isolation of the intended reward intervention from continued training and reset optimizer/replay state. Findings below distinguish facts, refutations/limitations, risks, and required follow-ups.

## Confirmed facts (incremental)

- Both evaluation metadata files declare 1,000 rounds, `classic`, base seed 20260731, the interval 20260731--20261730, identical game settings, Python/platform, training seed 11, and git state `39c770a-dirty`.
- Evaluation metadata records different model SHA-256 values: baseline `1421cd78adecdf814723083941f543042f0722b20a505510a023428e1a4d83dd`; fine-tune `5e72661f9d3b214d1d8f3400dc2f0f203e57e8d434df9a71979c1adab694cd22`.
- Training metadata declares that the fine-tune loads `ben_task2_escape_multiseed_reachable_v1_10000ep_seed11.pt`, uses 2,000 planned episodes, epsilon fixed at 0.05, and changes `safe_crate_wait_penalty` from 0 to -0.02. Both runs declare the same architecture/features and the same remaining listed reward and optimizer hyperparameters.
- Code constructs a fresh Adam optimizer, fresh replay buffer, fresh target network copied from the loaded online network, and resets environment/optimization step counters for every training invocation. Thus this is weight-only continuation, not a continuation of full DQN training state.
- File hashes of both deployed models exactly match their respective evaluation metadata. The deployed baseline and its episode-10000 checkpoint have different serialization hashes but all 118,646 tensors are bit-identical. Likewise the deployed fine-tune and its episode-2000 checkpoint have different serialization hashes but all tensors are bit-identical. This confirms final checkpoint/deployed-model identity at tensor level and explains why raw `.pt` SHA alone is not a tensor-identity test.
- Baseline and fine-tune tensors are genuinely different. Changes occur throughout the CNN/head (not merely the final action biases); for example 639/768 final-layer weights differ.
- Training CSVs are complete for episodes 1--10,000 and 1--2,000. Fine-tune epsilon is exactly 0.05 throughout. Its fresh buffer first reaches the 5,000-transition optimization threshold at episode 34, so episodes 1--33 alter no weights and the subsequent run trains from a distribution collected by the already-loaded policy.
- The penalty implementation returns -0.02 only for action `WAIT` when: a bomb is currently available, an orthogonally adjacent tile is a crate, the current tile has zero encoded danger, and the hypothetical-bomb escape channel contains at least one marked endpoint. It is applied to normal transitions and to the otherwise-missing lethal terminal transition.
- All 42 current unit tests for callbacks/features/model/training pass. The focused penalty test checks a simple adjacent-crate case and rejection when an existing bomb makes the current tile dangerous.
- Raw evaluation files each have exactly 1,000 rows and 1,000 unique seeds. Every ordered `(round, seed, slot)` key matches, with seeds 20260731--20261730 and no gaps. For Task 2 (one agent, no opponents), this is genuinely arena-paired.
- Direct paired analysis (candidate minus baseline) reports: score/coins +4.672 [4.489, 4.853], crates +64.466 [62.020, 66.912], bombs +21.254 [20.325, 22.174]. Score rises 1.270 to 5.942. Survival changes 0.937 to 0.930 and suicides 0.063 to 0.070; neither is demonstrated. The score gain is therefore real for these two frozen policies on these arenas, not an aggregation error.
- Framework counters are internally interpretable here. `steps` increments once per requested action; action events partition into moves, bombs, invalids and waits. Thus `WAIT = steps - moves - bombs - invalid` is exact. Reconstructed mean WAIT falls from 232.694 to 98.721 per round; no row gives a negative residual. Invalid actions are zero in both runs.
- No evaluation inference exceeded the 0.5 s limit (`think_over_limit=0` in all 2,000 rows). Maximum observed single-step inference was 8.6777 ms baseline and 25.317 ms candidate, comfortably under the limit. Hence forced timeout-WAITs do not explain the WAIT difference.
- Evaluation means are behaviorally plausible and mutually coherent: candidate score/coins 5.942, crates 95.961, bombs 29.590, moves 256.149; baseline 1.270, 31.495, 8.336, 140.955. More bombs accompany more destroyed crates and fewer waits. There are no opponents, so score equals coins and every death is a suicide, exactly as the CSV shows.
- Over the fine-tune training run versus the baseline's final 2,000 training episodes, means move in the same direction: coins 1.131 to 2.418, crates 29.851 to 50.339, bombs 10.173 to 14.230, WAITED 55.767 to 30.301. These training figures are noisy epsilon-0.05 diagnostics, not evaluation evidence.

## Refuting findings / limitations (incremental)

- The intended reward change is not isolated by the experiment as described: the candidate also receives 2,000 additional episodes, a fresh replay buffer, a reset Adam state, a reset target network, and a restarted environment-step schedule. A no-penalty 2,000-episode fine-tune control from the identical 10k weights is required to attribute any gain to the -0.02 reward.
- The metadata says which input filename code was configured to load, but does not record that input artifact's SHA at training start. It therefore cannot by itself prove the exact audited 10k bytes were the starting weights; tensor-level checkpoint comparison is still required.
- Both evaluations are from a dirty worktree and record different `callbacks.py` hashes. The differing hash may be explained solely by environment-selected constants, but provenance is not source-clean and the effective-code difference must be reconstructed before treating the pair as controlled.
- Strongest verdict: the *policy improvement* is convincingly measured, but the claim that it was caused by the targeted crate-WAIT penalty is not supported by this design. The result is equally compatible with ordinary weight-only continuation, optimizer/replay reset effects, or their interaction with the reward.
- Training provenance remains one link short. Configuration and code select the audited baseline filename, and that baseline equals its episode-10000 checkpoint tensor-for-tensor, but neither training metadata nor an episode-0 checkpoint records the loaded tensor SHA. The claim that these exact bytes were loaded is highly plausible, not cryptographically proven.
- Provenance collection hashes only `callbacks.py` and the selected model. It does **not** hash `features.py`, `model.py`, or other inference dependencies. Since the worktree is dirty and `callbacks.py` changed between evaluations (`b72f...` baseline, current/candidate `b4f1...`), the metadata cannot independently prove identical inference code. Current file modification times suggest `features.py`/`model.py` predate both evaluations, but timestamps are weaker than hashes.
- The adjective "safe" is stronger than the trigger proves. The escape channel performs a bounded static BFS and requires a safe endpoint, but it does not simulate time-indexed bomb/explosion dynamics along the path. An endpoint existing within four moves is not a formal guarantee that every traversed tile remains safe at the required time. This matters less in opponent-free Task 2 but is still a semantic limitation.
- Actual penalty firing frequency cannot be reconstructed from the training CSV. It logs total `WAITED`, not a dedicated `SAFE_CRATE_WAIT_PENALTY` event/count, and safety/adjacency/escape conditions cannot be recovered from episode aggregates. Therefore we can verify implementation and eligibility conditions, but not that the intervention fired often enough to explain the effect or what fraction of the 60,602 fine-tune WAITs it affected.
- The unit test does not exhaustively test each gate (no bomb available, no adjacent crate, no escape endpoint, non-WAIT) or temporal counterexamples. Passing tests do not close the semantic gap above.

## Risks (incremental)

- Single training seed (11) cannot establish that the effect is robust to DQN training variance. Fine-tune checkpoint/episode selection may add selection bias.
- A reward intervention only affects training; evaluation has no reward callback. Therefore any causal interpretation rests entirely on a controlled training contrast, which is currently absent.
- Checkpoint selection risk is material: the reported candidate is the endpoint at 2,000 episodes. Without a preregistered horizon or evaluation of all preserved 100-episode checkpoints under a selection-safe protocol, it is unknown whether 2,000 was chosen after inspecting quick/evaluation results.
- The effect is from one training seed and one baseline trajectory. The unusually large paired evaluation difference defeats arena noise for these two policies, but says nothing about whether the training recipe reliably produces the gain across seeds.
- Baseline and candidate evaluation callback hashes differ because the source evolved between runs. Re-evaluating both frozen models under one immutable callback/features/model snapshot would eliminate the residual source-provenance concern.
- Fine-tune training deaths are worse, not better: over its 2,000 episodes, suicide/death rate is 0.9015 versus 0.806 over the baseline's last 2,000. Evaluation survival is statistically unchanged. The improvement is aggressive crate/coin throughput, not demonstrated safety progress.

## ML and submission-rule audit

- Confirmed ML: inference is a learned CNN with 118,646 parameters mapping ten state channels to six Q-values; features do not directly output a best action. Training uses replay-buffer DQN updates. This is not a purely rule-based submitted policy.
- Confirmed tournament mechanics in inspected code: inference explicitly uses CPU; no multiprocessing appears in agent inference; model paths are relative; legal-action masking prevents invalid moves but does not replace learned action selection. Measured inference is far below 0.5 s.
- Scope caveat: this establishes compliance of the inspected agent implementation, not packaging/Docker completeness or the report's two-model requirement.

## Required follow-up checks
- Run the decisive control: from the same verified baseline tensor, fine-tune 2,000 episodes with epsilon 0.05 and **zero** crate-WAIT penalty, resetting optimizer/replay/target identically. Use multiple training seeds and evaluate every arm on the same 1,000 arena seeds under one frozen source snapshot.
- Better factorial design: compare (continued full-state vs weight-only reset) x (penalty 0 vs -0.02), or at minimum state explicitly that the estimand is "the entire fine-tune recipe" rather than the penalty.
- Record `load_model_sha256` and an episode-0 checkpoint in training metadata; hash all inference dependencies in evaluation provenance.
- Add a custom training-only trigger counter for eligible crate-WAIT penalties (without changing submitted inference behavior), plus tests for every gate and adversarial timed-bomb paths.
- Pre-register checkpoint/horizon selection. If intermediate checkpoints have already been inspected, reserve fresh held-out arena seeds for final selection confirmation.

## Bottom line

The audit does **not** refute that the frozen candidate is dramatically better than the frozen 10k baseline on the recorded paired Task-2 evaluation; that part is unusually strong (+4.672 score over 1,000 exactly matched arenas, with coherent bombs/crates/WAIT changes and no timeout artifact). It **does refute the claimed causal attribution as currently phrased**: the experiment changes a whole weight-only fine-tuning procedure, not just the safe-crate-WAIT reward. Until a zero-penalty continuation control and multiple training seeds exist, the defensible statement is: "a 2,000-episode weight-only fine-tune recipe that included a -0.02 eligible crate-WAIT penalty produced a large single-seed policy improvement." It is not yet defensible to say the targeted penalty produced that improvement.
