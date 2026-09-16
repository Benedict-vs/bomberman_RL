# Adversarial audit: alternating-opponent curriculum Full1000 (2026-09-10)

## Claim under test

The four-phase curriculum Mixed → External → Mixed → External improves the concrete
Mixed-Kill-2000 policy in the External-Trio field.

## Attempts to break the claim

- Both files contain 4,001 lines: 1,000 rounds times four agents plus header.
- Both use the same `classic` arena seeds (`20260731..20261730`), same four-agent
  External-Trio lineup, and greedy inference. The model hashes differ as expected;
  metadata and training phase chains are complete.
- Candidate minus control is paired on the same rounds. The result is not inferred
  from the earlier Quick100, whose control mean was `2.120` and therefore visibly
  overestimated the later control Full1000 mean `1.909`.
- Timing remains well below the 500 ms limit: candidate maximum `18.1489 ms`,
  control maximum `20.5178 ms`; mean target think time is about `0.218 ms` for both.
- Score arithmetic and primary-vs-secondary interpretation follow the repository
  measurement rules. No claim is made about beating every external agent
  individually; the claim concerns our agent's total score in the field.

## Result: candidate minus control

| Metric | Difference (95% CI) | sign-flip p | Verdict |
|---|---:|---:|---|
| Score | `+0.230 [+0.096, +0.365]` | `0.0011` | better |
| Coins | `+0.115 [+0.034, +0.196]` | `0.0054` | better |
| Kills | `+0.023 [+0.002, +0.045]` | `0.0459` | better |
| Suicides | `-0.052 [-0.096, -0.008]` | `0.0261` | better |
| Survival | `+0.073 [+0.030, +0.116]` | `0.0013` | better |

All primary and safety rows are non-fragile in the analysis output. The candidate's
score advantage is therefore supported for this concrete External-Trio comparison,
subject to the known opponent-RNG and single-training-seed limitations.

## Verdict

Promote the p4 curriculum checkpoint as the External-Trio challenger, not as a
general proof that curriculum training always wins. Do not add more phases or
hyperparameter search before a second training seed or a held-out confirmation.

