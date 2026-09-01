# Adversarial audit: `ben_task2` Escape-Reachable-Tiles-v1 (2026-08-26)

## Scope and audit posture

Question under attack: whether `agent_code/ben_task2` with the `reachable_safe_tiles` tenth channel demonstrates genuine Task-2 progress. This audit is read-only except for this report. It distinguishes confirmed facts, risks/insufficiencies, and required follow-ups.

## Interim findings: provenance and comparison design

### Confirmed facts

- The four named evaluation metadata files all report `base_seed = 20260731`, scenario `classic`, the same recorded game constants, and 1000 rounds spanning seeds 20260731–20261730. Thus arena construction is paired in principle.
- Escape and 10-channel-zero evaluations both name agent `ben_task2`, report commit `39c770a-dirty`, and were run on the same platform. The visit-count reference instead names `ben_task2_visit_count_eval`; the crate-reward reference again names `ben_task2`.
- The current callbacks select `ben_task2_escape_reachable_tiles_v1_7000ep_seed11.pt`, instantiate a 10-input-channel network, set visit-count encoding `linear_10`, and set `ESCAPE_FEATURE_MODE = "reachable_safe_tiles"`.

### Risks / insufficiencies

- The metadata does not record the loaded model path, model checksum, callback/configuration snapshot, feature mode, channel count, visit encoding, reward configuration, or training-log/checkpoint identity. Because all runs are marked `39c770a-dirty`, the metadata cannot reconstruct which uncommitted code/model each CSV actually evaluated. Matching filenames/labels are not provenance evidence.
- Reusing the same mutable agent directory and model filename makes silent artifact/configuration swaps possible between evaluations. The CSV alone cannot prove that the escape arm used the intended escape-trained model or that the zero-control arm used its intended independently trained/control model.
- Pairing the arena seeds controls board generation, but does not repair an unidentified treatment artifact. Any causal claim about the tenth channel requires verified weights and exact runtime configuration for both arms.

### Required follow-ups

- Compute and preserve hashes for every evaluated `.pt` file; link each CSV to model hash, callback/config snapshot, exact command, and training log/checkpoint.
- Re-evaluate from immutable agent snapshots (or otherwise archived configurations), not repeated edits to `ben_task2`, before treating labels as treatments.

## Interim findings: observed metrics

### Confirmed facts

Using `tools/analyze.py --compare ... --preset task2` on the named 1000-round CSVs:

- Versus the 10-channel zero control, escape has score 0.368 vs 0.186, paired difference +0.182, 95% bootstrap CI [+0.150, +0.216], sign-flip p reported as 0; crates 13.73 vs 9.19, +4.538 [+4.150, +4.922]. Suicides (0.043 vs 0.034) and survival (0.957 vs 0.966) show no demonstrated difference. Bombs rise by +2.460 [+0.693, +4.242]. The current analyzer also prints Wilcoxon p=0.1138 for bombs; this distributional disagreement is a caution, although the repository's formal `(fragile)` rule is based on CI/sign-flip/bootstrap stability rather than Wilcoxon, and the row is not marked fragile.
- Versus the older 9-channel visit-count model, score rises +0.083 [+0.046, +0.121] and crates +2.750 [+2.272, +3.240], while suicides worsen +0.027 [+0.012, +0.042] and survival worsens -0.027 [-0.042, -0.012]. This is not an unqualified escape/safety improvement.
- Versus the 9-channel crate-reward-0.3 arm, score difference is only +0.021 [-0.015, +0.058]: no score improvement is demonstrated. Escape has more crates and markedly more bombs, and fewer suicides, but the primary Task-2 score does not improve.

### Adversarial interpretation

- The strongest result is relative to one unusually weak independently trained zero-channel run. It establishes that these two final policies differ on these arenas; by itself it does not establish that escape information generally improves learning.
- Against the closest strong reward-matched 9-channel reference, the primary metric is null. Therefore the defensible claim is at most a policy-tradeoff (more crate destruction/bombing and fewer suicides), not demonstrated overall Task-2 progress.
- Against the older visit-count baseline, the score gain comes with a statistically demonstrated safety regression. Calling the result an “escape” advance without foregrounding that regression is misleading.

## Interim findings: feature semantics and rule fidelity

### Confirmed semantics and limitations

- **Correction to an interim hypothesis:** `_blast_coordinates` stops at stone walls but not crates. Direct inspection of `items.Bomb.get_blast_coords` shows that the actual supplied framework does exactly the same: only `arena == -1` stops a ray. Therefore this is rule-faithful for this repository, not a defect. (The report deliberately retains this correction so the rejected attack is visible.)
- `_reachable_escape_tiles` returns an all-zero channel whenever `bomb_available` is false. Immediately after the agent drops a bomb, bomb availability is false; consequently the added channel supplies no escape endpoints during the actual post-bomb escape phase. It can inform a pre-drop decision, but it is not a continuing “escape route” feature.
- The search is spatial rather than temporal. It computes positions reachable within four moves, but tests endpoints against a static union-like `danger` map (`danger > 0`) and allows traversal through dangerous tiles without checking when each bomb explodes. It neither models arrival time, explosion duration, chain reactions, nor waiting. Hence “reachable safe tiles” overstates the semantics: marked endpoints are only outside a static danger footprint under a simplified geometry.
- Existing bombs are treated as blocked positions, but the hypothetical bomb at `start` is not inserted into the blocked set. The BFS can therefore conceptually revisit/pass the origin except that `distances` prevents revisiting it; this happens not to create a second visit in the current BFS, but illustrates that the search is not a faithful forward simulator.

### Consequence

The tenth channel is learned input, so it does not violate the ML requirement or directly prescribe an action. There is no obvious forbidden “best action” leakage. But its name and causal story are stronger than its actual information: it mainly describes a conservative, partly rule-incorrect pre-bomb spatial mask. Any performance effect cannot automatically be attributed to learned post-bomb escape behavior.

Amendment: “partly rule-incorrect” above refers to absent temporal simulation, not crate blocking; blast geometry itself matches the supplied framework.

## Artifact and training audit

### Confirmed facts

- Both zero-v2 and escape training logs contain exactly episodes 1–7000. The final replay-buffer sizes differ (181,613 vs capped 200,000), which is consistent with different learned trajectories/episode lengths, not unequal episode counts.
- The final model state dict is tensor-for-tensor identical to its corresponding episode-7000 checkpoint for both arms (maximum absolute tensor difference 0). Different SHA-256 values for final file and checkpoint are only serialization-level differences, not weight differences.
- Training metadata agrees across arms on all recorded hyperparameters: 10-channel-capable architecture context, crate reward +0.3, death reward -5 on `GOT_KILLED` and 0 on `KILLED_SELF`, safety-potential scale 1, linear-10 visit channel, legal mask, augmentation, seed 11, and 7000 episodes. The death reward therefore prices every death once as required.
- All evaluation CSVs contain rounds 0–999 with seeds exactly 20260731–20261730. Maximum recorded `think_max_ms` is 43.013 ms for zero control and 11.144 ms for escape, with no proximity to the 500 ms tournament limit.
- Symmetry augmentation rotates/flips all channels uniformly and transforms movement actions consistently. Nothing special about channel 10 is dropped by augmentation.

### Remaining provenance limitations

- The present on-disk final weights matching episode-7000 checkpoints strongly supports training-artifact identity, but evaluation metadata still does not hash the model that was loaded at evaluation time. It cannot prove retrospectively that those exact bytes were used.
- Crucially, training metadata omits `escape_feature_mode` entirely. Zero and escape metadata are identical except run name/start time; the claimed manipulation exists only in mutable code/run naming. This is the single largest reproducibility gap.

## 9-channel versus 10-channel fairness

### Confirmed / defensible comparison

- The zero-v2 arm is the appropriate architecture control in concept: same 10-channel network and recorded hyperparameters, with channel 10 held at zero. Comparing escape to older 9-channel policies alone would confound information with 144 extra first-layer parameters and shifted initialization.

### Why the current control is insufficient for a general claim

- Both arms were trained only once, at training seed 11. DQN training is path-dependent; an independently trained zero arm can be an unusually poor draw even with the same nominal seed because the treatment immediately changes actions, trajectories, replay contents, and subsequent random-number consumption. The paired 1000-round evaluation quantifies evaluation uncertainty for these two fixed policies, not training-seed uncertainty.
- The escape-vs-zero contrast therefore supports “this escape-trained final policy beat this zero-trained final policy,” not “the feature reliably improves Task 2.” Multiple matched training seeds are necessary, with per-seed immutable model/config provenance and evaluation.
- The strong 9-channel crate-reward-0.3 arm is useful as a practical benchmark despite architecture confounding. Its null score comparison (+0.021 [-0.015, +0.058]) is direct evidence against calling the current result established overall progress.

## Leakage and submission-rule review

### Confirmed facts

- The feature uses only fields available in `game_state`, local constants matching the framework, and the agent's own visit-history array. No future state, hidden coin location, opponent information, absolute path, framework modification, multiprocessing, or `tools/` import in inference was found.
- The BFS-derived channel encodes a set of candidate endpoints rather than an action. The CNN must learn how to use it, so it remains a machine-learning feature under the project rules.

### Risk of overclaim, not formal leakage

- The channel embeds substantial hand-built search and hypothetical-blast knowledge. That is allowed feature engineering, but the report should avoid describing it as a learned escape planner. The learned part is only the mapping from this engineered mask (plus other channels) to Q-values.

## Verdict

**The broad claim “Escape-Reachable-Tiles-v1 is genuine Task-2 progress” is not yet demonstrated.** What is demonstrated is narrower:

1. On 1000 paired fixed arenas, one seed-11 escape-trained policy decisively beats its one seed-11 10-channel-zero policy in score and crates.
2. It does **not** demonstrate score progress over the active reward-matched 9-channel arm.
3. Against the older safe visit-count arm, its score/crate gain accompanies significantly worse suicides and survival.
4. The new channel disappears exactly during post-bomb escape (`bomb_available == false`) and lacks temporal hazard modeling, so attributing the result to improved escaping is unsupported by implementation and outcomes.
5. One training seed plus metadata that omits treatment mode and model hash is inadequate for a causal/reproducible feature verdict.

The safest ledger wording is: **promising single-seed evidence that a pre-bomb reachable-endpoint channel changes the learned policy and improves score/crate destruction relative to one 10-channel null run; no demonstrated overall score advance over the strongest reward-matched reference, and no demonstrated generalization across training seeds.**

## Necessary follow-up checks before promotion

1. Add `escape_feature_mode`, input channel count, source snapshot/commit, and evaluated model SHA-256 to training/evaluation metadata.
2. Run a preregistered matched multi-training-seed sweep of 10-channel zero vs escape, evaluating every seed on the same arenas; aggregate by training seed, not by pooling thousands of evaluation rounds as if they were independent training replicates.
3. Add behavioral diagnostics conditional on bomb placement: survival after own bomb, probability of leaving own blast before detonation, and suicides per bomb. Existing aggregate suicides cannot establish the proposed mechanism.
4. Ablate the gating: compare the present pre-bomb-only channel with a temporally meaningful post-bomb escape/safe-arrival channel. Unit-test states immediately after `BOMB` and with bombs of several timers.
5. Report the null primary-score result versus crate-reward-0.3 and the safety regression versus visit-count-v1 alongside the positive zero-control comparison.
