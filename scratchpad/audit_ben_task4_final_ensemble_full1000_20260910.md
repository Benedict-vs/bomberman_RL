# Adversarial audit: final ensemble selection Full1000 (2026-09-10)

## Claim under test

The Q-value ensemble is a better official submission than the Mixed-Kill-2000
incumbent because it is safer and might preserve score.

## Result

Both policies were freshly evaluated over 1,000 paired `classic` rounds against
three Rule-based agents. The ensemble averages the incumbent and Curriculum-p4
Q-values before the same legal action mask.

| Metric | Ensemble minus incumbent (95% CI) | Verdict |
|---|---:|---|
| Score | `-0.289 [-0.489, -0.086]` | worse |
| Coins | `+0.111 [+0.015, +0.209]` | better |
| Kills | `-0.080 [-0.114, -0.046]` | worse |
| Suicides | `-0.184 [-0.223, -0.145]` | better |
| Survival | `+0.216 [+0.176, +0.257]` | better |

The ensemble converts improved survival and coin collection into fewer kills and a
significant total-score loss. The result is non-fragile and the score loss is the
primary tournament criterion. Timing remains safe, but cannot compensate for lost
score.

## Verdict

Do not use the ensemble as the official submission. Keep it as a documented safe
fallback/research artifact. Freeze the Mixed-Kill-2000 policy as the official
candidate unless a genuinely new kill-conversion mechanism is developed and passes
the existing score and safety gates.

