# Adversarial audit: training-duration checkpoint results

## Question

Is more unchanged training likely to improve either learning-rate arm?

## Evidence

Each checkpoint is evaluated greedily for 100 External-Trio rounds on the same
base seeds. The `1e-4` control curve is:

| episodes | score | kills | suicides | survival |
| --- | ---: | ---: | ---: | ---: |
| 200 | 1.910 | 0.020 | 0.630 | 0.210 |
| 500 | 2.490 | 0.070 | 0.540 | 0.390 |
| 800 | 2.460 | 0.070 | 0.450 | 0.340 |
| 1000 | 2.510 | 0.050 | 0.480 | 0.350 |

The `5e-5` candidate curve is `2.340, 2.410, 2.400, 2.190`; its late suicide
changes are `+0.050, +0.040`.

## Attempted refutation

The control improves clearly from 200 to 500, so the run was not immediately
converged. However, the two final score changes are `−0.030` and `+0.050`, both
below the pre-specified `0.10` practical threshold. Its kills do not rise after
500. The plateau result is therefore not based on an early noisy point. The
low-learning-rate arm has both a plateau and a safety regression.

## Verdict

Do not spend compute on a longer unchanged continuation of either arm. The
control has reached a practical performance plateau by approximately episode
500 in this source/field/seed; the candidate is late-worse and less safe. This
does not rule out longer training after a genuinely new mechanism or over more
training seeds, but it rejects a blind 3,000-episode extension.
