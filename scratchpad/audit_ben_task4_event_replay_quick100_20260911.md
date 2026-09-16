# Adversarial audit: event-balanced replay Quick100

## Claim to challenge

Event-balanced replay improves external kill conversion without sacrificing
score or safety.

## Evidence checked

- Both 1,000-episode arms, their models, evaluation CSVs and metadata exist.
- Model hashes match metadata: control
  `529710da8c4672e10b50b9a6bef1b475711cf4766af671f3a16c770534a7d240`,
  balanced `12d87f332cbefad25c0ed3fb732891f1d1ea7cece1df7827e7a6d360c4dfaa32`.
- Both evaluations use trained models, training seed 11, base seed 20260731,
  100 External-Trio rounds and the same agent field.

## Result

Balanced minus uniform: score `−0.100 [−0.640,+0.440]`, coins
`−0.450 [−0.740,−0.150]`, kills `+0.070 [−0.010,+0.150]`, suicides
`+0.110 [−0.030,+0.250]`, survival `−0.060 [−0.190,+0.070]`.

## Verdict

The rare-event sampler does not pass its pre-registered Quick100 gate. The
intended kill increase is not demonstrated, primary score is directionally
worse, coins are demonstrably worse, and the nominal suicide increase exceeds
the `0.03` safety guard. No Full1000 and no promotion. The result is specific
to this sampling mixture, checkpoint, field and seed; it does not prove that
all prioritized replay methods are harmful.
