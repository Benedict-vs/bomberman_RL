# Adversarial audit: alternating-opponent curriculum Quick100 (2026-09-09)

## Claim under test

The alternating curriculum improves the Mixed-Kill incumbent's performance against
the external trio.

## Checks

- Control and candidate p4 each contain 400 rows for 100 paired `classic` rounds with
  the same four-agent External-Trio lineup and base seed `20260731`.
- Both p4 models are the final outputs of four 250-episode phases from the same
  Mixed-Kill-2000 source. The control used Mixed opponents in all phases; the
  candidate used Mixed, External, Mixed, External.
- The comparison uses the same arena seeds. Model provenance and phase metadata are
  present; no training curve is used as evidence.

## Candidate minus control in the External-Trio field

| Metric | Difference (95% CI) | Interpretation |
|---|---:|---|
| Score | `+0.320 [-0.120, +0.800]` | positive direction, not yet demonstrated |
| Coins | `+0.120 [-0.160, +0.410]` | positive direction, not shown |
| Kills | `+0.040 [-0.030, +0.120]` | positive direction, not shown |
| Suicides | `-0.130 [-0.260, +0.000]` | safer direction, not shown |
| Survival | `+0.140 [+0.010, +0.270]` | fragile; do not quote as established |

The external-field score direction is plausibly positive and there is no safety
regression, so the predeclared advancement gate permits Full1000 confirmation. The
3RB result is negative in score (`-0.280`) and must remain a transfer warning; it
cannot be hidden by the external result.

## Verdict

Advance to one Full1000 confirmation in the External-Trio field for both p4 models.
Do not promote, add seeds, extend phases, or claim improvement unless the Full1000
score result is non-fragile, positive, and accompanied by no material safety or
timing regression.

