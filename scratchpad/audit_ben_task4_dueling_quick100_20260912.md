# Adversarial audit: Dueling-DQN Quick100

## Result

Dueling minus classic control across 100 paired External-Trio rounds: score
`+0.090 [−0.290,+0.470]`, coins `+0.140 [−0.110,+0.390]`, kills
`−0.010 [−0.060,+0.040]`, suicides `+0.110 [−0.040,+0.260]`, survival
`−0.070 [−0.210,+0.060]`.

## Verdict

The exact Q-preserving initialization and both completed 2,000-episode arms
are present, so this is a valid architecture comparison. However, score is not
demonstrated, kills are directionally lower, and the suicide point estimate
exceeds the +0.03 safety guard. No Full1000 and no promotion. This result does
not condemn dueling architectures generally; it rejects this transfer, field,
seed and training horizon.
