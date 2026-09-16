# Adversarial design audit: lower learning-rate pilot

## Hypothesis

The current `1e-4` Adam learning rate may overwrite useful behavior during
fine-tuning. Reducing it to `5e-5` could preserve the incumbent's learned
safety and coin behavior while allowing slower adaptation.

## Controlled comparison

- Control `learning_rate_control_v1`: `1e-4`.
- Candidate `learning_rate5e5_v1`: `5e-5`.
- Both start from `ben_task4_mixed_kill_v1_2000ep_seed11.pt`, train 1,000
  episodes with seed 11 against peaceful + two rule-based opponents, and keep
  rewards, features, action mask, replay, target update and epsilon unchanged.
- Each receives a greedy External-Trio Quick100 with visible progress.

## Falsification gates

No Full1000 unless Quick100 score and kills are directionally non-negative and
suicides do not rise by more than `0.03`. Promotion still requires a non-fragile
paired Full1000 score advantage. A null result only rejects this exact learning
rate and horizon, not learning-rate tuning generally.

## Verdict

The candidate is a single, novel hyperparameter change; no prior Task-4 ledger
entry tested learning rate in isolation. It is approved for manual execution.
