# Adversarial design audit: training-duration checkpoint diagnostic

## Question

Does additional training within a 1,000-episode arm still improve greedy
external performance enough to justify a longer, separately controlled run?

## Method

For a completed `learning_rate_control_v1` or `learning_rate5e5_v1` run, evaluate
the existing checkpoints at episodes 200, 500, 800 and 1,000 over the same 100
External-Trio arenas (`base_seed=20260731`). The callback's checkpoint mode loads
the exact saved checkpoint, not the end model. `plateau_check.py` reports the
last two score and suicide changes.

## Interpretation guardrails

- This diagnoses the first 1,000 episodes only; it cannot prove a 3,000-episode
  result before that run exists.
- A stable rise from 500 to 800 to 1,000 with no suicide regression supports a
  single longer control/candidate study. Flat or falling late checkpoints make
  an unchanged extension unjustified.
- Quick100 is noisy. The diagnostic gates compute allocation, not model
  promotion. Any actual candidate selection remains a paired Full1000 decision.

## Verdict

The checkpoint plan avoids choosing a favourable end checkpoint after the fact
and uses only already-generated artifacts. It is approved for manual execution
after the learning-rate pilot completes.
