# Adversarial audit: external Mixed-Kill pilot Quick100 (2026-09-09)

## Claim under test

Continuing the audited Mixed-Kill-2000 model against Li-Jesse, Bindist and Binary
improves performance against that external opponent field.

## Provenance and pairing checks

- Control and candidate each contain 400 rows: 100 rounds times four agents.
- Both metadata files use `classic`, base seed `20260731`, training seed `11`, the
  same external four-agent lineup, and the same 11-channel source model.
- The control field is the established
  `peaceful_agent,rule_based_agent,rule_based_agent`; the candidate field is
  `ext_lijesse_featureeverything,ext_xiaoxiae_bindist_v2,ext_xiaoxiae_binary_v6`.
- The analysis is paired on the same arena seeds. The external agents' own random
  streams remain a known noise source, so this is a Quick100 gate, not a final
  tournament claim.

## Result: candidate minus control

| Metric | Difference (95% CI) | Interpretation |
|---|---:|---|
| Score | `-0.330 [-0.750, +0.100]` | no effect shown; wrong direction |
| Kills | `-0.050 [-0.120, +0.010]` | no effect shown; wrong direction |
| Suicides | `-0.010 [-0.150, +0.120]` | no effect shown |
| Survival | `+0.040 [-0.090, +0.170]` | no effect shown |

The candidate fails the advancement gate because neither the primary score nor the
kill mechanism has a plausible positive direction. The result is not a statistically
significant defeat at n=100, but it is sufficient to reject an expensive Full1000
as a pre-registered improvement test. The earlier external-trio pilot from the older
safe model also did not establish a benefit; this new source-controlled pilot does
not rescue that line.

## Verdict

No-Go for Full1000 and further external-opponent training in this configuration. A
3RB Quick100 transfer check may be run as a diagnostic only; it cannot promote the
external-training arm if the external-field gate remains negative.

