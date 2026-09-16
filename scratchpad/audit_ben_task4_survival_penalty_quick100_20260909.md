# Adversarial audit: survival-penalty Quick100 (2026-09-09)

## Claim under test

Changing `GOT_KILLED` from `-5` to `-10` improves the incumbent's survival and
therefore Task-4 score.

## Checks

- Control and candidate each contain 400 rows for the same 100 paired `classic`
  rounds against three `rule_based_agent` opponents.
- Both start from the same Mixed-Kill-2000 source, use 11 channels, seed 11 and
  identical features/hyperparameters. Metadata confirms only the intended reward
  difference: `GOT_KILLED=-5` versus `-10`; `KILLED_SELF=0` in both.
- The candidate model is loaded by its own SHA and the output files are complete.

## Candidate minus control

| Metric | Difference (95% CI) | Interpretation |
|---|---:|---|
| Score | `-0.020 [-0.650, +0.610]` | no effect shown |
| Kills | `+0.020 [-0.080, +0.120]` | no effect shown |
| Suicides | `+0.100 [-0.030, +0.230]` | no effect shown, wrong direction |
| Survival | `-0.060 [-0.190, +0.070]` | no effect shown, wrong direction |

The candidate does not pass the advancement gate: it has no positive score direction,
does not reduce suicides, and nominally worsens survival. The result is not a
significant defeat at n=100, but a Full1000 would be an expensive rescue attempt
without a plausible mechanism.

## Verdict

No-Go for Full1000 and further training of this reward-only arm. The strong common
suicide problem is not fixed by simply doubling the terminal death penalty.

