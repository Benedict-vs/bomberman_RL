# Adversarial audit: `mixed_safety_multiseed_v1`

Date: 2026-08-30

Scope: independently try to refute the proposed conclusion from the three 5,000-episode runs (training seeds 11/12/13). This report is written without modifying implementation, models, results, or `BEN.md`.

## Provisional finding

All three stored end policies are genuine, distinct, complete runs and all three independently satisfy the predeclared safety thresholds in their 1,000-round evaluations. The experiment supports a robust *family-level* claim: continuation from the mixed policy against three active coin collectors raises score largely through coins while restoring safety. It does **not** preselect a scientifically justified single seed to freeze, and it does not demonstrate better hunting/killing.

## Artifact and implementation checks

- Each training CSV has exactly episodes 1--5,000; each seed has exactly 50 checkpoints (100--5,000), one end model, and one metadata file.
- Each end model is tensor-identical to its episode-5,000 checkpoint. Serialization bytes differ, which is normal for separately saved PyTorch archives; tensor equality, not file-byte equality, is the relevant identity test.
- End-model SHA-256 values are distinct and match both quick-100 and 1,000-round evaluation provenance: seed 11 `a25a0677...9afe6`, seed 12 `1277fedc...3fc9`, seed 13 `1285ac5c...d8b63`.
- All three policies changed all ten state-dict tensors relative to the common mixed source model (`d5c083f9...1680`). The code selects that same source for every seed and writes seed-specific model/checkpoint/log names.
- `setup()` seeds Python `random`, NumPy, and Torch with `BM_TASK3_TRAINING_SEED`; the launcher also passes that seed to `main.py`. Thus the three runs are genuinely different, although the seed changes both agent RNG and arena sequence and cannot isolate either source of variation.
- Training metadata agree on 11 channels, reachable-safe escape feature, visit encoding, legal mask, symmetry augmentation, rewards (`coin +1.5`, kill `+5`, death once via `GOT_KILLED -5`, `KILLED_SELF 0`), zero opponent potential, and the same optimizer/hyperparameters. The launcher command and metadata agree on three `coin_collector_agent` opponents. Runtime metadata do not independently hash the starting model or record opponent identities, so those two provenance facts rest on current code plus the preserved launcher rather than a self-contained training artifact.
- All three 1,000-round evaluations use the same callback SHA, scenario/settings, arena-seed interval, trained-model switch, and correct model hash. Therefore inference semantics are comparable.

## Recomputed 1,000-round results

| training seed | score | coins | kills | suicides | killed by opponent | survived |
|---:|---:|---:|---:|---:|---:|---:|
| 11 | 2.615 | 2.235 | 0.076 | 17.8% | 3.5% | 78.7% |
| 12 | 2.760 | 2.220 | 0.108 | 20.5% | 3.8% | 75.7% |
| 13 | 2.692 | 2.252 | 0.088 | 16.3% | 2.9% | 80.8% |
| seed mean | **2.689** | **2.236** | **0.091** | **18.2%** | **3.4%** | **78.4%** |

For every row, `score = coins + 5*kills` and `died = suicides + killed_by_opponent` exactly. Every seed individually passes the stated guards (survival at least 70%, suicides at most 25%); the seed mean passes too. The preregistered family-level criterion in `BEN.md` therefore genuinely passes.

## Attempts to break the conclusion

1. **The gain is not hunting progress.** Against the safe source (score 1.801 = 1.261 coins + 5×0.108 kills), the seed-mean gain is +0.888 score. Coins contribute +0.975 while kills contribute **−0.087**. Against the mixed source (score 2.193 = 1.593 coins + 5×0.120 kills), coins contribute +0.643 and kills contribute **−0.147**, for net +0.496. The correct claim is improved tournament score against this opponent field, driven entirely by more coins; “better killer/hunter” is contradicted.
2. **Apparent pairing is not full pairing.** Arenas and slots match, but opponent stdlib randomness is not synchronized. Only 134/3,000 opponent-slot outcomes match across all three evaluations. The paired bootstrap/sign-flip output therefore overstates causal pairing. None of the score comparisons was marked fragile, and the gains (+0.814/+0.959/+0.891 versus safe; +0.422/+0.567/+0.499 versus mixed) are far above the documented ~0.12 single-evaluation opponent-noise floor, so this caveat does not plausibly erase the family-level score result; it does invalidate overly precise causal intervals.
3. **Only three training seeds exist.** The arithmetic seed mean is the preregistered decision statistic, but three models cannot support a precise population CI over training randomness. Do not pool the 3,000 rounds as if they were 3,000 independent training replicates.
4. **Callback hashes differ from older controls.** The three new evaluations share callback SHA `bf26...bcc`; the safe and mixed-source evaluations record older callback SHAs. Current code and metadata indicate that intervening changes added arm/model selection while preserving feature/reward inference semantics, but the agent directory is untracked and the old callback files are not archived, so exact historical source equivalence cannot be reconstructed from Git. This prevents a byte-level semantic proof, though the feature-mode/visit metadata and model architecture agree.
5. **No timeout issue.** Observed per-round maximum inference times were 16.377/10.771/18.056 ms, with zero over-limit events, far below 500 ms.

## Model-selection verdict

The experiment preregistered acceptance of the **seed mean**, not a rule for choosing one deployment model. Seed 12 has the highest observed score, but freezing it as “best” would be post-hoc winner selection. Seed 13 is the median-score model, but a median rule was also not preregistered. Seed 11 was listed first, but “first seed wins” was not explicitly declared either.

Therefore no specific one of 11/12/13 is scientifically established as superior. Defensible options are:

- preserve/freeze all three as the successful multiseed family and make no individual superiority claim; or
- before inspecting another dataset, preregister a deployment-selection rule and a genuinely held-out confirmation. If an immediate single practical artifact is required with no new selection experiment, seed 11 is the least selection-biased convention (the first fixed seed), but it must be described as an arbitrary representative, not the best model.

## Final verdict

**Pass, narrowly scoped.** The preregistered multiseed criteria pass across every individual model and in the seed mean. The defensible conclusion is: “5,000-episode active-opponent continuation reproducibly produces safer, higher-scoring policies against three coin collectors, with the score gain coming from coins.” It is not evidence of improved killing, not a complete Task-3 claim against other opponent types, and not a license to select seed 12 from these same results as the winner.

## Follow-up audit: preregistered peaceful selector

### Was the rule actually preregistered?

Yes, in the conversation before any peaceful-selector command was supplied or result inspected. The exact rule was: evaluate each model for 1,000 rounds against three `peaceful_agent`; choose a model only if its score is clearly better than both others; otherwise choose neutral first seed 11; retain suicides and maximum think time as guards. Only after this statement did the user run the three evaluations. This resolves the model-selection gap identified above.

Documentation caveat: the rule was not appended to `BEN.md` before the runs; its current top entry still says that individual selection is open. The conversation establishes temporal preregistration, but the durable experiment ledger should have contained it before execution. This is a documentation omission, not evidence that the rule was invented after seeing results.

### Files and provenance

- Each selector CSV is complete: 4,000 rows, including exactly 1,000 `ben_task3` rows covering rounds 0--999. Each has a matching metadata file with `classic`, 1,000 rounds, base seed 20260731, the same callback SHA (`bf26...bcc`), three `peaceful_agent`, and the expected trained-model switch/seed.
- Recorded model hashes match the actual seed-11/12/13 model files exactly and are the same hashes audited above. Thus no wrong-model or artifact-selection error was found.
- All settings and callback hashes match across the three selectors. Maximum observed think times were 10.20/13.05/25.08 ms for seeds 11/12/13, with zero 500-ms overruns.

### Recomputed selector results

| seed | score | coins | kills | suicides | survived |
|---:|---:|---:|---:|---:|---:|
| 11 | 4.867 | 3.387 | 0.296 | 5.2% | 94.8% |
| 12 | 5.447 | 3.732 | 0.343 | 8.3% | 91.7% |
| 13 | **7.713** | **5.108** | **0.521** | 5.6% | 94.4% |

Seed 13 beats seed 11 by +2.846 score (reported 95% CI +2.422 to +3.270, sign-flip p < 0.0001, non-fragile) and seed 12 by +2.266 (+1.780 to +2.760, p < 0.0001, non-fragile). It also has more kills than both. Against seed 12 it has fewer suicides by 2.7 percentage points; versus seed 11 safety is statistically indistinguishable. Its invalid-action count is higher, but still only 0.046 per round and was an additional metric rather than the preregistered selector.

Peaceful opponents are not fully synchronized: 1,948/3,000 opponent-slot summaries and 350/1,000 complete opponent rounds match across all three runs. Consequently the paired CIs are not literal controlled-policy counterfactuals and may be too narrow. However the score margins of 2.27--2.85 are enormous relative to the project's measured opponent-noise floor, agree with large kill and coin differences, and are nowhere near the significance boundary. Unsynchronized opponents do not plausibly reverse “clearly better” here.

### Selector verdict

**Seed 13 legitimately satisfies the preregistered rule and can be frozen as the concrete Task-3 candidate.** This is no longer post-hoc highest-seed selection: the independent peaceful context and fallback rule were fixed before its results. The selected peaceful performance is necessarily selection-set performance and should not be presented as an untouched final estimate; a frozen-agent reproduction or later held-out confirmation should be labeled accordingly.

The scope remains important: seed 13 is a defensible representative of the already-successful active-opponent family and is clearly strongest among the three against passive opponents. This does not yet prove a finished tournament agent, performance against `rule_based_agent`, or a general training-method effect beyond these three seeds.

## Final frozen-agent follow-up: `agent_code/dqn_task3`

### Independence and identity

- `dqn_task3_seed13.pt` is byte-for-byte identical to the selected development source `ben_task3_mixed_safety_multiseed_v1_5000ep_seed13.pt`, SHA-256 `1285ac5c...d8b63`.
- `features.py` and `model.py` are byte-identical to the development versions. The model remains an 11-channel `CoinCollectorDQN`; visit encoding and reachable-safe escape semantics are unchanged.
- The callback diff removes development arm/environment/checkpoint selection and fixes seed 13 plus the local model filename. The action/feature/masking/inference implementation below that configuration is unchanged. Inference is explicitly CPU-only and uses `torch.inference_mode()`.
- The fixed path is relative and local. No reference to `ben_task3`, `results/`, `tools/`, checkpoints, MPS/CUDA, or an absolute path exists. There is no `train.py`, so the frozen folder has no hidden training dependency.
- Required submission files are the callback, features, model, and `.pt`. Evaluation generated three `.pyc` files, `__pycache__`, and an empty `logs/dqn_task3.log`; these are harmless runtime artifacts but are not needed and should be removed/excluded before packaging.

### Frozen measurement audit

Both frozen evaluations record the actual callback SHA `3a601738...d11f`, actual model hash, fixed seed 13, expected feature metadata, CPU-compatible model, correct scenario/settings, and three coin collectors. The quick run has 400 rows/100 own rounds; the final run has 4,000 rows/1,000 own rounds covering rounds 0--999 without gaps.

Recomputed final frozen metrics are score 2.605, coins 2.215, kills 0.078, suicides 16.5%, opponent kills 4.0%, survival 79.5%, invalid actions 0.281 per round, and maximum think time 9.337 ms with zero overruns. Score and death identities hold exactly in every aggregate (`score = coins + 5*kills`; `died = suicides + killed_by_opponent`).

Against the development seed-13 evaluation, differences are unresolved: score −0.087 (95% CI −0.239 to +0.066), kills −0.010, suicides +0.002, survival −0.013, and invalid actions +0.022; none is marked fragile or significant. Only 541/3,000 opponent-slot summaries and 18/1,000 complete opponent rounds match, confirming that exact row reproduction is impossible because opponent randomness is not synchronized. The observed −0.087 score shift is below the documented ~0.12 opponent-noise floor and supplies no evidence of a freezing regression.

### Frozen verdict

**Pass. `dqn_task3` is a faithful, standalone frozen copy of the selected seed-13 Task-3 policy.** No model, channel, callback-semantic, path, CPU, provenance, runtime, or result mismatch was found. The differing aggregate result is consistent with unsynchronized opponent trajectories rather than changed policy code.

This verdict is deliberately scoped to faithful freezing as the concrete Task-3 candidate. It does not claim that Task 4 is solved, that the agent beats `rule_based_agent`, or that runtime cache/log artifacts belong in the final archive.
