# Adversarial audit: Task 3 `kill_reward75` (2026-08-29)

## Scope and provisional finding

This audit tries to falsify the claim that raising the training reward for
`KILLED_OPPONENT` from `+5` to `+7.5` improved the safe Task-3 policy. The
complete 1,000-round result does not support that claim: the quick-screen score
gain does not replicate, and the candidate does not meet the pre-registered
requirement of a demonstrated primary-score improvement.

## Evidence inspected

- Safe source model, `kill_reward75` end model and episode-2,000 checkpoint.
- Both training CSV/metadata files and both quick-100 evaluation CSV/metadata
  files.
- Complete 1,000-round `coin_collector` evaluations and metadata for the safe
  source policy and the `kill_reward75` policy.
- Current Task-3 callback/training selection code, `tools/analyze.py` comparison
  semantics, and the bounded relevant section at the top of `BEN.md`.

## Artifact completeness and provenance

- Both training logs contain exactly episodes 1 through 2,000 without gaps,
  and both arms have all 20 expected checkpoints (100, 200, ..., 2,000).
- The `kill_reward75` final model has SHA-256 `7e449e92...ea9236`, exactly the
  hash recorded by both its quick and full evaluation metadata. Its tensors are
  exactly equal, key by key, to the episode-2,000 checkpoint; the differing file
  SHA of the checkpoint is only PyTorch serialization, not differing weights.
- Training metadata names the safe model
  `ben_task3_coin_collector_finetune_v1_2000ep_seed11.pt` as the source, records
  `KILLED_OPPONENT=7.5`, and otherwise matches the safe arm's recorded features,
  rewards and hyperparameters. The currently present source file has SHA-256
  `a9bd850a...f6c8f5`, matching its evaluations. As in the previous arm, the
  training metadata records the source filename but not its historical SHA, so
  strict proof of the bytes loaded at training start is unavailable.
- Both full evaluation CSVs are complete: 4,000 rows, 1,000 unique rounds, four
  slots per round, identical arena seeds `20260731..20261730`, scenario and game
  settings. The intended model file and hash are correctly recorded in each.

## Callback comparability caveat

The callback hashes differ (`567f77...82cc` for the safe reference and
`7f2e05...989d` for `kill_reward75`). Inspection of the current source shows the
new training-arm branch selects the new source/output names, while inference,
feature extraction, visit counting and legal-action masking remain shared and
arm-independent. This makes semantic comparability plausible, but the directory
is untracked and no historical callback snapshot is stored, so the metadata
alone cannot prove that adding the arm selector was the *only* callback change.
The comparison should not be described as having byte-identical callbacks.

## Independently reconstructed evaluation results

For focal slot 0, all 1,000 rows satisfy both `score = coins + 5*kills` and
`died = suicides + killed_by_opponent` exactly.

| Metric | safe source | `kill_reward75` | change |
|---|---:|---:|---:|
| official score | 1.801 | 1.782 | -0.019 |
| coins | 1.261 | 1.432 | +0.171 |
| kills | 0.108 | 0.070 | -0.038 |
| suicides | 0.183 | 0.171 | -0.012 |
| killed by opponent | 0.039 | 0.049 | +0.010 |
| total deaths | 0.222 | 0.220 | -0.002 |
| survival | 0.778 | 0.780 | +0.002 |
| crates | 30.791 | 33.939 | +3.148 |
| bombs | 13.202 | 19.240 | +6.038 |
| invalid actions | 0.399 | 0.257 | -0.142 |

The tournament reward remains `+5` per kill; the shaped `+7.5` exists only in
training, so using `coins + 5*kills` for evaluation is correct.

`analyze.py --compare ... --preset task3` reports:

- score `-0.019 [-0.166,+0.126]`, sign-flip `p=0.8082`: no effect shown;
- kills `-0.038 [-0.064,-0.013]`, `p=0.0045`: worse in these files;
- suicides `-0.012 [-0.044,+0.020]`: no effect shown;
- survival `+0.002 [-0.032,+0.036]`: no effect shown;
- invalid actions `-0.142 [-0.195,-0.091]`: lower.

None of those rows is marked fragile by the tool. The safety guardrails are met
(78.0% survival and 17.1% suicides), but safety is statistically unchanged from
the already-safe source policy rather than newly improved.

## Why the quick result was misleading

The quick-100 score moved from 1.95 to 2.24 (+0.29), while the complete
1,000-round difference is -0.019. Kills were 0.12 in both quick files but fall
from 0.108 to 0.070 in the full files. Thus the screening result was correctly
treated only as a gate to the larger measurement; it is not evidence of a score
gain.

The two full runs share arenas and slots, but opponent outcomes match exactly in
`0/1000` rounds. The opponent implementations use unseeded stdlib randomness,
and changing the focal policy also changes their trajectories. Small paired
differences therefore cannot be interpreted as exact controlled treatment
effects. The score change is additionally far below the project's approximately
`0.12` active-opponent noise floor. The kill decrease is statistically clear in
these files, but a general causal claim that the larger reward *reduces kills*
would still overreach without repeat evaluations and training seeds.

## Training and causal limits

Only training seed 11 exists. Moreover, this is a fresh continuation that resets
optimizer, target-network and replay state; it is not paired against a same-seed,
same-duration `+5` continuation control from the same safe source. Consequently
the policy difference cannot be attributed uniquely to changing `+5` to `+7.5`.
The result evaluates this concrete resulting policy, not a robust reward effect.

The exploratory training tail is not a policy result: both arms show roughly
11-13% survival and roughly 85-87% own-bomb deaths in their final 100-500 logged
episodes, yet their final greedy evaluations survive about 78%. It must not be
used to override the epsilon-zero evaluation.

## Verdict

The claim that `kill_reward75` recovered score or aggression is broken. It
produces no demonstrated score gain, has fewer rather than more measured kills,
and adds no demonstrated safety benefit over the source policy. It therefore
fails the pre-registered acceptance rule even though it stays inside the safety
guardrails.

Rejecting this **specific candidate as the next baseline** is justified. Deleting
its evidence or claiming that `+7.5` is universally harmful is not justified:
the experiment has only one training seed, lacks an equal-duration `+5` control,
uses non-repeating opponents, and has a callback-provenance caveat. The safe
`coin_collector_finetune_v1` policy remains the better-supported intermediate
baseline; further Task-3 development is still warranted because neither policy
has demonstrated superiority on primary score.
