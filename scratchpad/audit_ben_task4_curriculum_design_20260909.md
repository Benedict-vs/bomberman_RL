# Adversarial design audit: alternating-opponent curriculum (2026-09-09)

## Hypothesis

The fixed external-trio continuation failed, but a curriculum alternating between
the established mixed Rule-based field and the external trio may retain the
incumbent's safety while exposing the policy to diverse play.

## Isolation and implementation checks

- Control and candidate start from the identical audited
  `ben_task4_mixed_kill_v1_2000ep_seed11.pt` source.
- Each arm has four sequential 250-episode phases, with each phase loading the
  previous phase's end model. The candidate uses Mixed, External, Mixed, External;
  the control uses Mixed in all four phases.
- Rewards, features, network, optimizer, exploration and seed 11 are unchanged.
  Only the opponent schedule differs. The training metadata receives the actual
  phase lineup through `BM_TASK4_TRAINING_OPPONENTS`.
- Each phase has separate model/CSV/metadata/checkpoint names and overwrite
  protection. The external agents remain opponents only.
- Shell syntax, configuration imports and the existing 29 Task-4 tests pass.

## Gates

Run control and candidate sequentially by default; they may be started in parallel
only because every phase has separate artifacts. Evaluate the final phase greedily
in the 3RB field and the external trio, without `--quiet`. The candidate must not
lose primary score or materially increase suicides. A positive result requires a
non-fragile paired effect and an independent audit; otherwise the curriculum is
stopped without extra phases or seeds.

## Verdict

Design is allowed as the next pilot. No claim is made before all eight phase runs
and final evaluations are complete.

