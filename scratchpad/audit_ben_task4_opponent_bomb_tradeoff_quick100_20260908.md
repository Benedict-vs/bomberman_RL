# Adversarial audit: opponent-bomb tradeoff Quick100 (2026-09-08)

## Claim under test

`opponent_bomb_tradeoff_v1` improves the Task-4 agent over the 12-channel zero
control.

## Checks intended to break the result

- Both CSVs contain exactly 400 agent rows for rounds 0--99, so the comparison is
  complete at the expected four slots per round.
- Both metadata files report 100 rounds, base seed `20260731`, training seed `11`,
  the same `classic` scenario and the same four-agent lineup (`ben_task4` plus
  three `rule_based_agent` opponents).
- The model SHAs differ as expected (`8d584c...` control and `cd23ab...`
  candidate), while the training configuration differs only in the intended
  tradeoff mode (`zero` versus `enabled`) and the corresponding trained model.
- The comparison is paired by the existing analysis tool on identical arena seed
  indices. No Full1000 result is inferred from this Quick100 run.

## Result

Candidate minus control:

| Metric | Difference (95% CI) | Interpretation |
|---|---:|---|
| Score | `+0.020 [-0.470, +0.520]` | no effect shown |
| Kills | `+0.010 [-0.080, +0.100]` | no effect shown |
| Suicides | `+0.180 [+0.060, +0.300]` | worse |
| Survival rate | `-0.160 [-0.290, -0.030]` | worse |

The score and kill intervals include zero, while the safety regression is clear and
violates the predeclared `+0.03` suicide guard. The result is therefore a No-Go for
Full1000 and for further training of this arm. The feature hypothesis is rejected
as a useful improvement for this configuration; the established Mixed-Kill 2000
model remains the current candidate.

