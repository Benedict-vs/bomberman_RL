# Adversarial audit: lower learning-rate Quick100

## Claim to challenge

Reducing the Adam learning rate from `1e-4` to `5e-5` preserves useful behavior
and improves external performance during fine-tuning.

## Result

Candidate minus control on 100 paired External-Trio rounds: score `−0.320`
`[−0.770,+0.140]`, coins `−0.270 [−0.530,−0.010]` (fragile), kills `−0.010`
`[−0.070,+0.060]`, suicides `+0.060 [−0.070,+0.190]`, survival `−0.030`
`[−0.150,+0.090]`.

## Verdict

The candidate fails the pre-registered Quick100 gate: score and kills point in
the wrong direction and the nominal suicide change exceeds `+0.03`. The coin
row is fragile and cannot support a separate claim. No Full1000 and no
promotion. This rejects only `5e-5` for this source, field, seed and horizon.
