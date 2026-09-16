# Adversarial audit: checkpoint ensemble Quick100 (2026-09-10)

## Claim under test

A 50/50 Q-value ensemble of the Mixed-Kill-2000 incumbent and Curriculum-p4 is a
better general-purpose policy.

## Checks and result

- The ensemble loaded both fixed 11-channel models, applied the existing legal
  action mask after averaging, and stayed well below the 500 ms step limit.
- In 100 3RB rounds it reached score `3.760`, kills `0.110`, suicides `0.180`,
  opponent deaths `0.060`, and max think time `0.5 ms`.
- In 100 External-Trio rounds it reached score `2.450`, kills `0.060`, suicides
  `0.300`, opponent deaths `0.110`, and max think time `0.9 ms`.
- The available prior incumbent comparisons are separate runs and have known
  opponent-RNG noise; they do not form a clean paired control for this ensemble
  Quick100. The direct external comparison showed only score `+0.050` with CI
  `[-0.310, +0.420]`, not a demonstrated effect.

## Verdict

No promotion and no Full1000 based on this validation: the ensemble has no proven
score or kill advantage and its apparent safety advantage is not sufficient to
offset the missing primary effect. Keep it as a low-cost research artifact; the
Mixed-Kill-2000 policy remains the general-purpose incumbent.

