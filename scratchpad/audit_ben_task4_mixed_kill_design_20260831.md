# Pre-run adversarial audit: `mixed_kill_v1` versus `rule_based_continue_v1` (2026-08-31)

## Question and attempted refutation

The proposed seed-11 pilot continues the safe `baseline_v1` policy for 2,000 episodes. The control trains against three `rule_based_agent`; the candidate replaces one with `peaceful_agent`. The intended causal question is whether this easier kill opportunity restores kills and official score without giving back the learned safety. I checked source identity, configuration equality, paths, tests, confounding and the proposed decision rule before either run exists.

## What is correctly controlled

- Both arms load exactly `ben_task4_baseline_v1_5000ep_seed11.pt`, whose SHA-256 is `ea27ab3d483298e5c56dfa93b97e2a6dc3453ed0bdb6bff79e2a605abf7457ab`. This is the completed safe seed-11 model already audited; it is tensor-identical to its Episode-5,000 checkpoint.
- Both use the same 11-channel features, network, augmentation, replay, optimizer, epsilon `0.05`, learning rate, target update, potentials and WAIT shaping.
- Reward tables are identical: coin `+1.5`, crate `+0.3`, kill `+5`, death `-5`, `KILLED_SELF=0`, invalid `-1`, step `-0.05`. Thus the intended code-level difference is only training opponent composition.
- Task-4 output isolation is intact. For the actual 2,000-episode configuration the two final models, CSVs, metadata files, log directories and all twenty checkpoints per arm are currently absent. Output/checkpoint names include the arm, `2000ep` and seed 11 and point only into `agent_code/ben_task4`, `results/train/ben_task4` and the new top-level log directories. No Task-3 model/result path is loaded or written.
- Arm selection is explicit and rejects unknown names. `rule_based_continue_v1` records three rule-based opponents; `mixed_kill_v1` records one peaceful plus two rule-based opponents. Both select the same source.

## Defects and limits found

### Actual 2,000-episode test configuration currently fails one test

The six tests pass under the module's default `TOTAL_EPISODES=5000`, but running them under the planned `BM_TASK4_TOTAL_EPISODES=2000` produces one failure: `test_artifact_names_and_paths_are_task4_only` hardcodes the expected model/run names as `5000ep`. The runtime code correctly produces `2000ep`; the test is stale/configuration-dependent. This does not corrupt training, but the claim “tests pass for the planned configuration” is currently false. Fix the expectation to derive from `callbacks.TOTAL_EPISODES` (or test arm configuration in a subprocess) and rerun under `BM_TASK4_TOTAL_EPISODES=2000` before the long launch.

### Metadata does not enforce the CLI

`TRAINING_OPPONENTS` is descriptive metadata only. It cannot verify that `main.py` was actually launched with the matching four-agent order. The collision-safe launcher must explicitly use:

- control: `ben_task4 rule_based_agent rule_based_agent rule_based_agent`;
- candidate: `ben_task4 peaceful_agent rule_based_agent rule_based_agent`.

Post-run audit can check recorded arm and declared lineup, but without recording CLI agents it cannot prove actual training opponents from the CSV alone. Keep the launcher itself as evidence.

### The causal claim is about a distribution, not perfectly paired trajectories

Changing one opponent is the intended treatment, but training experiences cannot be episode-paired. Provided opponents draw OS entropy and the rule-based agent also uses unseeded stdlib `shuffle`; `main.py` does not reseed per episode, and different round lengths make later arena sequences drift. Therefore seed 11 controls our initialization/exploration and the starting arena stream, not realized opponent trajectories or identical per-episode arenas. The direct pair is a legitimate pilot comparison of two training distributions, not a deterministic one-variable replay.

### Later multiseed extension needs source mapping

Seed 11 is neutral here because it is the pre-existing first safety seed, not selected for its observed Task-4 score; both arms start from that same policy, so the pilot comparison is fair. However, callbacks currently hardcode `ben_task4_baseline_v1_5000ep_seed11.pt` for both arms regardless of `BM_TASK4_TRAINING_SEED`. A later seed-12/13 replication would therefore change only optimizer/exploration seed while still starting every run from source seed 11. If the scientific target is the safe *multiseed family*, source files must be mapped 11→11, 12→12, 13→13 before those runs and covered by tests. Do not silently call the current hardcoded-source design a three-source multiseed experiment.

## Is 2,000 episodes sensible?

Yes as a bounded first pilot. The policy already has useful safety, epsilon is low, and 2,000 episodes limit catastrophic drift while allowing substantial replay updates after the 5,000-transition warm-up. It is not enough by itself for a Task-4 conclusion. Save all twenty checkpoints, but do not search them after seeing the end result unless a small checkpoint set and rule are pre-registered first.

The peaceful opponent changes one third of the threat field: it may make kills easier, but also exposes the policy to fewer bombs. Any kill gain accompanied by worse safety against the all-rule-based evaluation field is not automatically useful.

## Pre-registered evaluation and success criteria

Evaluate **both** end models greedily against the same three-rule-based field, with explicit `MODEL_VARIANT=trained`, first for 100 rounds as a gate and then 1,000 if the candidate remains plausible. The scientific contrast is `mixed_kill_v1 - rule_based_continue_v1`; comparison with the 5,000-episode source is secondary.

The candidate passes the seed-11 pilot only if:

1. **Official score (primary):** mixed minus control is positive, bootstrap 95% CI excludes zero, sign-flip `p<0.05`, no fragile flag, and the magnitude exceeds the known roughly `0.12` Task-4 opponent-noise floor.
2. **Kills (mechanism):** mixed minus control is positive with a non-fragile CI excluding zero. A score gain without a kill gain does not validate the “mixed kill” mechanism, though it may identify a different candidate.
3. **Safety guards:** relative to control, suicides must not increase demonstrably, killed-by-opponent must not increase demonstrably, and survival must not decrease demonstrably. Practical guard margins should also be shown: no more than `+0.03` suicides, `+0.02` killed-by, or `-0.03` survival in point estimates. Compare absolute behavior with the seed-11 safe source (`0.274` suicides, `0.070` killed-by, `0.656` survival) as context, not as the causal contrast.
4. **Timing:** zero over-limit actions, global observed `think_max_ms <500`, preferably `<50 ms` on the development machine.

Because rule-based stdlib randomness is unseeded, evaluations are paired on arenas and list slot but not on opponent trajectories. One 1,000-round seed-11 result can screen an arm; it cannot replace the frozen agent or establish a report-level Task-4 effect. A passing pilot must be replicated over training seeds. For true family replication, use the correspondingly seeded safe source policies and analyze training seed as the replication unit.

## Verdict before launch

The runtime design is a useful and mostly clean controlled pilot: same safe source, rewards, features and hyperparameters, with training lineup as the intended treatment. Seed 11 and 2,000 episodes are defensible for screening. One pre-run issue should be fixed first: the planned 2,000-episode environment currently makes the artifact-name unit test fail because it expects `5000ep`. After that test passes under the actual configuration and the launcher visibly enforces both lineups and collision checks, the two seed-11 runs are justified. A negative or mechanism-free pilot should stop this arm; a positive pilot proceeds to correctly source-mapped multiseed replication, not immediate model selection.

## Post-run audit: seed-11 pilot

### Artifact and provenance checks

Both arms contain exactly 2,000 unique training rows numbered 1–2,000, exactly twenty checkpoints at 100-episode intervals, and final models tensor-identical to their Episode-2,000 checkpoints. Both metadata files identify the same source `ben_task4_baseline_v1_5000ep_seed11.pt`, seed 11, 2,000 episodes, identical reward/hyperparameter values and the intended declared training lineups. The earlier limitation remains: lineup text is metadata supplied by the arm, not proof of the actual CLI, although the completed experiment was launched as designed.

Final model hashes:

- `rule_based_continue_v1`: `f7e7206db1fb19829f931d2b2a959db36bbaa0dab331a60a0856e5bbd1ce1c8e`;
- `mixed_kill_v1`: `d50ae3d1ce80018a8a834f7df28cff3dfead5c0c99d578d7af1604a3cf0806c6`.

Both 100-round screens are complete and hash the correct model. Both final evaluations contain 4,000 rows, 1,000 complete rounds, exactly one slot-0 `ben_task4` and three rule-based rows per round. Evaluation metadata explicitly selects the right arm and `MODEL_VARIANT=trained`; model hashes match the final files. Official score and death-cause arithmetic hold in every row.

### Direct result and orientation

The comparison is correctly oriented as `mixed_kill - rule_based_continue`. Therefore positive score/kills favor the candidate, while positive suicides/killed-by and negative survival violate safety:

| metric | continue control | mixed candidate | candidate − control (95% CI) | sign-flip p | verdict |
|---|---:|---:|---:|---:|---|
| score | 3.606 | 3.892 | `+0.286 [+0.084,+0.484]` | 0.0064 | better |
| kills | 0.119 | 0.184 | `+0.065 [+0.031,+0.099]` | 0.0003 | better |
| suicides | 0.288 | 0.431 | `+0.143 [+0.102,+0.183]` | <0.0001 | worse |
| killed by opponent | 0.043 | 0.072 | `+0.029 [+0.010,+0.049]` | 0.0066 | worse |
| survival | 0.669 | 0.497 | `-0.172 [-0.213,-0.132]` | <0.0001 | worse |
| win rate (secondary) | 0.399 | 0.409 | `+0.010 [-0.030,+0.051]` | 0.664 | no effect |

No comparison is marked fragile; bootstrap and sign-flip conclusions agree. The `+0.286` score gain exceeds the known approximately `0.12` Task-4 evaluation-noise floor, and the separate kill increase validates the intended aggression mechanism in this concrete pilot. The 100-round screen points in the same qualitative direction (score `4.45` versus `3.59`, kills `0.23` versus `0.06`) while already warning of worse safety.

However, the pre-registered safety limits are missed by large margins:

- suicides allowed at most `+0.03`, observed `+0.143`;
- killed-by allowed at most `+0.02`, observed `+0.029` and statistically worse;
- survival allowed at least `-0.03`, observed `-0.172`.

The failures are not boundary effects. They are 4.8×, 1.45× and 5.7× their practical margins respectively, with CIs wholly on the harmful side. Thus no reinterpretation of signs, bootstrap fragility or the `±0.12` score heuristic rescues acceptance.

Timing remains safe. There are zero over-limit actions; global observed DQN maxima are `21.239 ms` for the control and `22.242 ms` for the candidate. The analyzer's lower `think max` table values are means of per-round maxima, not global maxima.

### RNG caveat and strength of conclusion

The evaluations share arenas and list slot but not fully paired opponent trajectories because rule-based stdlib `shuffle` remains unseeded. Training trajectories are also unpaired. This weakens precise causal magnitude claims from one seed, but it cannot reasonably erase safety changes of `+0.143` suicides and `-0.172` survival alongside non-fragile tests. The correct narrow statement is that this concrete mixed-training policy gains score and kills while paying an unacceptable safety cost; not that every peaceful/rule-based mixture must do so.

### Decision

`mixed_kill_v1` must **not replace** the safe policy under the pre-registered rules. It should be retained as an informative aggression/safety trade-off and negative acceptance result. Do not spend seeds 12/13 evaluating the unchanged 2,000-episode mixed arm: the seed-11 pilot failed explicit safety gates, so multiseed expansion would be an attempt to rescue a rejected design rather than replication of a passing candidate.

The result does reveal a useful lever: limited peaceful exposure can restore kills and score. The next controlled approach should test whether active-opponent consolidation can recover safety while retaining that behavior. An efficient equal-length curriculum comparison is:

1. candidate path: the existing 2,000-episode mixed model, followed by a bounded continuation against three rule-based opponents;
2. control path: the existing 2,000-episode rule-based-continue model, followed by the same number of additional episodes against three rule-based opponents.

This keeps total post-source training length equal and asks whether the earlier mixed phase leaves a durable kill/score benefit after identical safety consolidation. Start with a short, pre-specified continuation (for example 1,000 episodes), do not search checkpoints post hoc, and reuse the same primary/mechanism criteria and safety guards. Only if that seed-11 curriculum passes should source-mapped seeds 12/13 be trained. A simpler reduced-dose mixed phase is another valid arm, but must likewise have an equal-length all-rule-based control. Raising kill reward or suicide penalty again is not supported by the existing negative experiments.

## Post-run audit: 1,000-episode active-opponent consolidation

### Completeness, source and evaluation provenance

Both consolidation paths contain exactly 1,000 unique training rows numbered 1–1,000, ten checkpoints at 100-episode intervals, and final models tensor-identical to Episode 1,000. The control correctly loads `ben_task4_rule_based_continue_v1_2000ep_seed11.pt`; the candidate correctly loads `ben_task4_mixed_kill_v1_2000ep_seed11.pt`. Both then use the same three-rule-based declared lineup, rewards, features, hyperparameters, seed 11 and 1,000-episode length. Thus the paths have equal total post-safety-source length: 2,000 treatment/control episodes plus 1,000 identical-distribution continuation episodes.

Final model hashes are:

- `rule_based_continue_control1000_v1`: `c9677540849340e915f74dec7cf9c2d24e9c6c27f9d5132453798035a1f8a9ec`;
- `mixed_kill_consolidate1000_v1`: `485cc4bb0fcc4b0019cfddb1be8f3b15f783650cc7399a4d15101ebedc2adc4d`.

Both quick evaluations contain 400 rows/100 rounds and both full evaluations contain 4,000 rows/1,000 rounds. The DQN is slot 0, each field has three rule-based opponents, model hashes and explicit arm environments match the intended final files, and all score/death identities hold. No overwrite or model-selection mismatch is present.

### Direct consolidated comparison

Orientation is `mixed-history consolidation - pure-rule-based continuation`:

| metric | pure control | mixed-history candidate | difference (95% CI) | sign-flip p | interpretation |
|---|---:|---:|---:|---:|---|
| score | 3.663 | 3.353 | `-0.310 [-0.506,-0.115]` | 0.0022 | clearly worse |
| kills | 0.142 | 0.111 | `-0.031 [-0.063,+0.000]` | 0.0619 | no effect shown, fragile |
| suicides | 0.355 | 0.327 | `-0.028 [-0.069,+0.013]` | 0.192 | no effect shown |
| killed by opponent | 0.069 | 0.052 | `-0.017 [-0.038,+0.004]` | 0.135 | no effect shown |
| survival | 0.576 | 0.621 | `+0.045 [+0.004,+0.087]` | 0.0390 | better |
| win rate (secondary) | 0.405 | 0.385 | `-0.020 [-0.061,+0.022]` | 0.367 | no effect shown |

The preliminary conclusion survives attempted refutation on the **primary metric**. Score is lower by `0.310`, the CI excludes zero, sign-flip agrees, the row is not fragile, and the magnitude is well beyond the approximately `0.12` opponent-noise heuristic. The candidate loses `0.155` coins and `0.031 × 5 = 0.155` kill points, exactly splitting the observed `-0.310` score gap.

The kill row must not be overstated: its bootstrap upper endpoint rounds to `+0.000`, sign-flip `p=0.0619`, and the analyzer correctly flags it fragile. Therefore “kills are proven lower” is not defensible. But acceptance required score **and** kills to improve; a clear score loss rejects the candidate even if the true kill difference were zero.

Safety consolidation did work. Relative to the equal-length control, point changes satisfy all practical guards (`-0.028` suicides, `-0.017` killed-by, `+0.045` survival), and survival is demonstrably better. Relative to the unconsolidated mixed policy, the candidate changes from suicides `0.431` to `0.327`, killed-by `0.072` to `0.052`, and survival `0.497` to `0.621`. It recovered safety but did so while erasing the earlier score/kills advantage. This is failure of the intended balance, not failure of consolidation itself.

### Think-time anomaly

The analyzer reports mean per-round maximum latency `0.7 ms` for control versus `2.6 ms` for candidate, a statistically detectable difference. This is not a timeout defect. Actual global maxima are `18.622 ms` and `21.206 ms`, respectively; both runs have zero actions above 500 ms and remain below the pre-registered 50 ms development margin. The candidate's mean action time is only `0.207 ms` versus `0.188 ms`. Record the timing difference, but it cannot explain rejection or threaten tournament legality.

### RNG and scope

As before, identical evaluation seeds pair arenas and list slots, not stdlib-shuffled opponent trajectories. The one-seed training paths are also stochastic and not episode-paired. This limits universal causal claims, particularly the fragile kill row. It does not plausibly erase the large non-fragile score deficit. The narrow conclusion is about these concrete seed-11 policies and curriculum, not every possible consolidation length.

### Decision and next experiment

Reject `mixed_kill_consolidate1000_v1` as a replacement and stop this exact curriculum; do not run seeds 12/13. It failed the pilot's primary gate after consolidation, while the earlier unconsolidated model failed all safety gates. Keep both as a documented trade-off sequence. Do not inspect the ten consolidation checkpoints now: no checkpoint-selection rule was registered, so doing so would be post-hoc fishing for the transition point.

The strongest next evidence is no longer another mixed/correction cycle. The pure all-rule-based path now reaches score `3.663`, kills `0.142`, suicides `0.355`, killed-by `0.069`, survival `0.576` on seed 11. It is a promising single-seed policy, but cannot be selected as Task-4 progress yet. A scientifically economical next step is to reproduce the same total unchanged-reward rule-based continuation length from the already existing safe seed-12 and seed-13 sources, with source mapping preserved, then evaluate all three. This tests whether longer active-opponent learning raises or at least preserves primary score across training seeds without introducing another reward/curriculum confound. If that family again improves only safety and not score, move to a genuinely new aggression representation or auxiliary learning design rather than further peaceful exposure, higher kill reward, or stronger death penalties—all of which now have direct negative evidence.
