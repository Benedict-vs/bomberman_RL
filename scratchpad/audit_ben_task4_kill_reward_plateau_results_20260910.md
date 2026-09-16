# Adversarial audit: kill-reward plateau pilot results

## Claim to challenge

The `+7.5` kill-reward candidate might be a better continuation than the `+5`
control, and the phase evaluations might show that more training is still useful.

## Evidence checked

- Both arms have four phase models, four evaluation CSVs and metadata.
- Final model hashes match the evaluation metadata: control p4
  `c1cc5fde71c7645e02b14a0ead33fec539f7453b6042edaea021202b30fe35fe`, candidate
  p4 `4e6ca615ed462b1e5eda1f28e469f4b088154c8544a9753f1ff6bca1cf621842`.
- Both evaluations use training seed `11`, base seed `20260731`, 100 rounds and
  the same External Trio.
- Candidate minus control at p4: score `+0.460` `[+0.150,+0.780]`, coins
  `+0.310` `[+0.050,+0.580]`, kills `+0.030` `[0.000,+0.070]`, suicides
  `−0.070` `[−0.200,+0.060]`. Kills and safety are not demonstrated at n=100.

## Attempted refutation

The phase curve does not support a clean plateau or monotonic improvement. The
candidate scores are `2.590, 2.470, 1.980, 2.280`; the control scores are
`2.260, 2.320, 2.440, 1.820`. The checker reports `PLATEAU=still_improving`
for both, but this is driven by noisy reversals, not a consistent upward trend.
The candidate's suicide rate is not regressing across the last two phases, but
it remains high (`0.45` at p4).

## Verdict

The pilot is technically valid and the candidate passes a provisional Quick100
score-direction gate, but it is not a plateau proof and not a promotion. Run
one paired External-Trio Full1000 for p4 control versus p4 candidate if the
remaining budget justifies confirmation. Do not select the candidate from
Quick100 alone; require non-fragile score improvement, no kill loss, and no
safety regression.
