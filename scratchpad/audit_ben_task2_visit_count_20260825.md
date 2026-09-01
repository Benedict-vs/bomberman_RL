# Adversarial audit: Task-2 visit-count claim

Date: 2026-08-25

## Claim under audit

`visit_count_v1` improves Task 2 because the ninth visit-count channel resolves
repeated-state aliasing.

## Checks that passed

- Both 1,000-round evaluations contain exactly 1,000 unique seeds with the same
  range `20260731..20261730`; no duplicates or missing paired seeds were found.
- Scenario, game constants, Python/platform metadata and `BM_QUIET_LOGS` match.
- Neither agent produced invalid actions or inference timeouts; maximum observed
  think times were 22.7 ms (Safety) and 13.3 ms (Visit Count), far below 0.5 s.
- For both agents, recorded score equals collected coins in every row.
- The first 300 rounds of each 1,000-round file reproduce its earlier 300-round
  measurement exactly.
- The 9-channel Safety conversion preserves every historical weight bit-for-bit
  and gives the ninth channel exact zero weights. Its first 100 rounds reproduce
  the historical 8-channel Safety evaluation exactly.
- The paired 1,000-round result is statistically clear and not marked fragile:
  score `+0.074` [95% CI `+0.036..+0.112`], sign-flip `p=0.0001`.

## Finding that breaks the causal claim

The two trained models do not form an architecture-matched feature ablation.
Safety was initialized and trained as an 8-channel CNN, while Visit Count was
initialized and trained as a 9-channel CNN. Even with the same training seed,
changing the first convolution from 8 to 9 input channels changes fan-in,
initialization scale and the number of random draws; later layers therefore also
start from a different random state. The two policies follow different training
trajectories for reasons beyond whether the ninth channel contains visit counts.

The post-hoc zero extension proves that the old Safety policy can be evaluated
through a 9-channel interface without changing its behavior. It does **not** turn
its historical training run into a 9-channel control.

## Correct verdict

The concrete `visit_count_v1` model is the strongest measured Task-2 policy and
beats the concrete Safety model on the paired 1,000 arenas. The experiment does
not yet demonstrate that the visit-count information caused the improvement.

## Required falsification control

Train a `visit_count_zero_control` with the same 9-channel architecture, seed,
hyperparameters and code path as `visit_count_v1`, but keep channel 9 identically
zero during both training and inference. Compare both trained models on identical
arenas. Ideally repeat the pair across several training seeds before treating the
feature effect as causal.
