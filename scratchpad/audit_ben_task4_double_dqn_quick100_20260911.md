# Adversarial audit: Double-DQN Quick100

## Claim to challenge

Double DQN improves external score and kill conversion without a safety cost.

## Result

Candidate minus classic-DQN control over 100 paired External-Trio rounds:
score `+0.120 [−0.300,+0.540]`, coins `+0.170 [−0.120,+0.470]`, kills
`−0.010 [−0.070,+0.050]`, suicides `−0.100 [−0.220,+0.020]`, survival
`+0.020 [−0.100,+0.140]`.

## Verdict

No row demonstrates a primary or kill benefit. The score direction is positive
and the safety direction favourable, but the intended kill signal is slightly
negative and the Quick100 intervals are broad. The pre-registered gate requires
non-negative score and kill direction; it is not met. No Full1000 and no
promotion. The result does not prove Double DQN harmful; it only provides no
justification to spend a Full1000 measurement on this checkpoint/seed/horizon.
