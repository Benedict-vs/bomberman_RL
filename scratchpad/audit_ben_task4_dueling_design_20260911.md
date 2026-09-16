# Adversarial design audit: Dueling-DQN architecture pilot

## Hypothesis

Separating state value and action advantages may better distinguish bomb/escape
actions while retaining a stable estimate of otherwise similar board states.

## Controlled comparison

- `dueling_control_v1` uses the existing classic CNN/Q head.
- `dueling_v1` uses the same convolutional trunk and hidden layer, followed by
  value and advantage heads combined as `V + A - mean(A)`.
- Both run 2,000 episodes, seed 11, in the Mixed field with identical rewards,
  replay, optimizer, epsilon, target update and features.

## Q-preserving conversion

The classic convolutional and hidden-layer weights are copied. The old final Q
head becomes the advantage head. The value head receives the mean weight and
bias across old actions. Thus `V + A - mean(A)` equals the old Q vector exactly
before training. A unit test verifies equality on random 11×17×17 batches.

## Gates

Checkpoints 500/1000/1500/2000 diagnose whether this new architecture still
learns beyond the prior fine-tuning plateau. Quick100 requires non-negative
score/kill direction and no suicide increase over 0.03; only then is Full1000
allowed. The classic incumbent is never overwritten.
