# Adversarial design audit: checkpoint ensemble (2026-09-10)

## Hypothesis

A simple average of the Q-values from the Mixed-Kill-2000 incumbent and the
curriculum-p4 checkpoint may combine the incumbent's 3RB strength with the
curriculum's External-Trio robustness.

## Isolation and safety checks

- This is inference-only; no training, reward, feature, action set or action mask
  changes are made.
- Both models use the same 11-channel feature representation and the same action
  ordering. The ensemble averages their Q-values before applying the existing legal
  action mask and argmax.
- The incumbent model remains the first model; the curriculum model is loaded from
  its fixed p4 checkpoint. Both files are separate and immutable during evaluation.
- `ensemble_v1` is a separate development arm and is not yet copied into
  `agent_code/dqn_task4`.
- Setup import, model loading configuration, Python syntax and all 29 Task-4 tests
  pass. CPU-only inference remains unchanged.

## Validation gate

Evaluate the ensemble on a fixed 100-round validation set in both 3RB and the
External-Trio field, without `--quiet`. Compare against the existing incumbent on
the same base seed. Promote only if score is not lower in either field, safety does
not regress, timing remains below 500 ms, and the result survives a Full1000 audit.

## Verdict

Design is allowed as a low-cost validation test. No promotion or claim of ensemble
improvement exists before evaluation.

