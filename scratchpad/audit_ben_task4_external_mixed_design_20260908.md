# Adversarial design audit: Mixed-Kill versus external-opponent training pilot

## Hypothesis

Starting from the same audited `mixed_kill_v1` 2,000-episode model, training against
Li-Jesse, Bindist and Binary may improve robustness and kill conversion against
non-rule-based play.

## Checks before training

- `mixed_external_control_v1` and `mixed_external_v1` load the identical
  `ben_task4_mixed_kill_v1_2000ep_seed11.pt` source and keep the same 11-channel
  features, rewards and DQN hyperparameters.
- The control uses the established `peaceful_agent,rule_based_agent,rule_based_agent`
  field; the candidate uses exactly
  `ext_lijesse_featureeverything,ext_xiaoxiae_bindist_v2,ext_xiaoxiae_binary_v6`.
  Configuration imports confirm these lineups and the common source.
- Outputs have separate names and overwrite protection. The candidate's external
  agents remain opponents only; their code is not imported into the submitted agent.
- This is a new opponent-distribution hypothesis, not a claim that the earlier
  external-trio pilot proved improvement. That earlier pilot started from an older
  model and is not a valid control for this question.

## Advancement gates

Train control first, then candidate, each for 1,000 episodes with seed 11. Evaluate
both at epsilon zero on both the external trio and the three-rule field, without
`--quiet` so progress is visible. Advance only if the candidate shows a plausible
primary-score direction in Quick100, does not increase suicides by `+0.03` or more,
and passes provenance/timing checks. Full1000 and any promotion require a paired
non-fragile score result plus no safety regression.

## Verdict

Design is allowed to proceed. No performance claim is made before evaluation.

