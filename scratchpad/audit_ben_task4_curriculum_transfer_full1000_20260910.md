# Adversarial audit: curriculum transfer to 3RB (2026-09-10)

## Claim under test

The External-Trio curriculum challenger is a generally stronger replacement for
the Mixed-Kill-2000 incumbent.

## Result

The curriculum candidate was evaluated over 1,000 `classic` rounds against three
Rule-based agents and compared with the existing 1,000-round incumbent file on the
same arena seed convention:

- Candidate: score `3.278`, kills `0.117`, suicides `0.232`, survival `0.711`.
- Incumbent: score `3.892`, kills `0.184`, suicides `0.431`, survival `0.497`.
- Candidate minus incumbent: score `-0.614 [-0.810, -0.418]`, kills
  `-0.067 [-0.100, -0.034]`, suicides `-0.199 [-0.240, -0.157]`, survival
  `+0.214 [+0.172, +0.255]`.

All differences are non-fragile and significant. The curriculum produces a real
safety gain but loses too much scoring and kill activity in the Rule-based field.
Its positive External-Trio result is therefore field-specific, not a general
improvement. The external agents still score above it in that field.

## Verdict

No promotion to the submission agent and no claim of being better than the three
external agents. Keep the curriculum checkpoint as an analyzed research result;
the Mixed-Kill-2000 model remains the general-purpose incumbent.

