# Adversarial design audit: Double-DQN pilot

## Hypothesis

Classic DQN selects and evaluates the next action with the target network,
which can overestimate attractive bomb actions. Double DQN selects the legal
next action using the online network and evaluates that selected action using
the target network. This could improve value calibration without hand-coded
policy behavior.

## Controlled comparison

- `double_dqn_control_v1` retains the existing target maximum.
- `double_dqn_v1` changes only the Bellman target selection/evaluation split.
- Both start from the same Mixed-Kill-2000 checkpoint, train 1,000 episodes
  with seed 11 in the same Mixed field, and retain features, rewards, replay,
  target-update rate, optimizer, epsilon and legal mask.
- The legal mask is applied before both the classic maximum and Double-DQN
  online action selection.

## Technical checks

The new unit test constructs disagreeing online/target action rankings: classic
DQN uses target value 10, while Double DQN selects online action 1 and uses its
target value 1. The test passed with the complete 32-test Task-4 suite.

## Gates

Quick100 must show score and kill direction non-negative and no suicide rise
above `0.03`; only then is Full1000 justified. No promotion without a
non-fragile Full1000 primary-score advantage.

## Verdict

This is a genuine isolated algorithmic change, not a reward/feature/duration
variation. It is approved for manual control/candidate execution.
