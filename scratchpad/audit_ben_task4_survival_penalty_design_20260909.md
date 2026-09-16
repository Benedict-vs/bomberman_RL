# Adversarial design audit: survival-penalty pilot (2026-09-09)

## Hypothesis

The incumbent's common failure is self-bomb death. A stronger terminal death signal
may teach the existing 11-channel policy to preserve escape options without adding a
hand-coded action rule or a new feature.

## Isolation checks

- Both arms load the identical audited
  `ben_task4_mixed_kill_v1_2000ep_seed11.pt` source.
- Both use 11 channels, the same mixed training field
  `peaceful_agent,rule_based_agent,rule_based_agent`, seed 11, 1,000 episodes,
  features, optimizer and exploration schedule.
- `survival_penalty_control_v1` keeps `GOT_KILLED=-5`; `survival_penalty10_v1`
  changes only `GOT_KILLED` to `-10`. `KILLED_SELF` remains zero, so own-bomb
  deaths are not accidentally double-priced.
- The change is a reward signal, not a best-action feature or hard-coded policy.
  Separate output names and overwrite protection are active.
- Shell syntax, both configuration imports and the existing 29 Task-4 tests pass.

## Gates

Train control first, then candidate. Quick100 is required at epsilon zero. The
candidate must reduce suicides by a meaningful amount without losing primary score;
a suicide reduction alone is not enough. Full1000 is allowed only after the paired
Quick100 has a plausible score direction, no fragile verdict, and no material score
regression. External-agent performance is a later transfer check, not the training
gate.

## Verdict

Design is allowed as the next isolated test. No performance claim exists before the
two training artifacts and their greedy evaluations are complete.

