# Adversarial audit: Task-2 coin reward +1.5 fine-tune (2026-08-28)

## Scope and claim under attack

This audit attempts to falsify the claim that `ben_task2_escape_crate_wait003_coin_reward15_finetune2000_from_wait003_v1_seed11.pt` is a clearly stronger practical Task-2 policy candidate than `ben_task2_escape_crate_wait003_finetune2000_from10000_v1_seed11.pt`. It does not assume that the reward change caused any observed policy difference.

## Verdict

**The narrow policy-selection claim survives this audit:** on the supplied no-opponent Task-2 evaluation, the candidate is clearly and practically stronger than the stated `wait003` baseline. The effect is large, consistently paired, non-fragile under the repository's decision rule, and accompanied by better safety rather than a score/suicide trade-off.

**The causal reward claim does not survive:** these data do not establish that changing `COIN_COLLECTED` from +1.0 to +1.5 caused the improvement. The candidate also received 2,000 additional episodes with a fresh replay buffer, Adam optimizer and target network. There is no otherwise-identical +1.0 continuation control, and only training seed 11 was used.

**The broad/global claim is not established:** the candidate is the strongest *evaluated Ben Task-2 DQN artifact found in `results/eval/task2_crates` at n >= 1000*, but this is not evidence that it is the best model anywhere in the repository, the best possible Task-2 policy, or robust across training seeds. It has not yet been frozen into a submission-style agent folder.

## Checks performed and findings

### 1. Evaluation pairing and raw CSV integrity — passed

- Both CSVs contain exactly 1,000 rows and 1,000 unique `(round, seed)` pairs.
- Row-by-row pairing is exact: rounds 0..999 and seeds 20260731..20261730 occur in the same order in both files. There are no duplicate round/seed pairs.
- Both runs use `classic`, no opponents, the same platform/Python/framework settings, the same base seed, and identical callback SHA-256 `98d3c1b87fb83dbb21baccd9594d9f494e1d954bdfccb5b9dfce03d8b8cc0e7f`.
- The current callback file still has that exact SHA. Thus the later code state has not silently changed since evaluation.
- With no opponents, `score == coins` in every raw row for both models. `survived + suicides == 1` in every row, as expected in this setting. No invalid actions or timeout overruns are recorded.
- Framework counters are copied from `Agent.statistics`; `agents.py` maps `CRATE_DESTROYED`, `BOMB_DROPPED`, movement events and `KILLED_SELF` to the reported `crates`, `bombs`, `moves` and `suicides`. No custom reward value enters evaluation score.

### 2. Recomputed policy results — passed

`tools/analyze.py --compare ... --preset task2 --markdown` reproduces:

| Metric | wait003 baseline | +1.5 candidate | Paired difference (candidate - baseline) | Audit result |
|---|---:|---:|---:|---|
| Score / coins | 6.046 | 7.699 | +1.653 [1.468, 1.836] | BETTER, sign-flip p reported 0.0000, not fragile |
| Suicides | 0.082 | 0.024 | -0.058 [-0.077, -0.039] | BETTER, p 0.0000, not fragile |
| Crates | 99.948 | 109.835 | +9.887 [7.822, 11.913] | BETTER, p 0.0000, not fragile |
| Bombs | 31.490 | 35.964 | +4.474 [3.449, 5.464] | BETTER, p 0.0000, not fragile |
| Survival | 0.918 | 0.976 | +0.058 [0.039, 0.077] | BETTER, p 0.0000, not fragile |

Additional raw checks: the candidate scores higher on 693 paired arenas, lower on 166, and ties on 141. It averages 5.57 more movement actions and 12.30 more alive steps. Those latter changes are descriptive here, not preregistered primary verdicts. The higher crate/bomb totals are partly enabled by longer survival, but that does not undermine practical policy selection: survival and official score both improve.

The analyzer uses paired seed maps, a fixed-seed percentile bootstrap, a 20,000-draw sign-flip test, and repeats the CI verdict over additional bootstrap seeds for the `(fragile)` flag. None of the five Task-2 preset rows is flagged fragile. The printed `p=0.0000` means no sampled null statistic was as extreme; it is not literally proof of probability zero.

### 3. Model identity and load selection — passed

- Baseline evaluation metadata names `ben_task2_escape_crate_wait003_finetune2000_from10000_v1_seed11.pt`, SHA-256 `e49a2a40198b413e3a0c28656eca98bfd1029255ddb2cd4e64ce626452cec24a`.
- Candidate metadata names `ben_task2_escape_crate_wait003_coin_reward15_finetune2000_from_wait003_v1_seed11.pt`, SHA-256 `b3f2756552d015adecd4bcb1d16a02ca26b97e9a891222d0c4ce909da19671e0`.
- Both current files match those recorded hashes.
- Each final development model is tensor-for-tensor identical to its own episode-2000 checkpoint. Different serialization bytes explain why the candidate checkpoint's file SHA differs from its final model SHA.
- Callback selection requires `reachable`, 2,000 episodes, fine-tune enabled and WAIT penalty -0.03. With `BM_TASK2_FINETUNE_COIN_REWARD=1`, inference selects the candidate filename; without it, inference selects the baseline filename. Metadata records these distinct environment configurations.

### 4. Actual training configuration and completeness — passed with a warning

The candidate training metadata records:

- source/load model: the wait003 baseline;
- `coin_collected_reward = 1.5`;
- `potential_reward_scale = 0.0`, target disabled;
- `safe_crate_wait_penalty = -0.03`;
- 10 input channels, `reachable_safe_tiles`, linear visit-count encoding;
- epsilon fixed at 0.05, training seed 11, MPS training device;
- planned 2,000 episodes.

The training CSV contains exactly episodes 1..2000 with monotonically increasing elapsed time, and checkpoints 100..2000 exist at every 100 episodes. The final replay buffer reaches its configured 200,000 capacity. The first 37 losses are `nan`, consistent with waiting for minimum replay size rather than evidence of numerical failure.

Warning: the final 100 exploratory training episodes show 92% `KILLED_SELF`/`GOT_KILLED`, while greedy evaluation shows only 2.4% suicides. This large train/eval gap is plausible because training retains epsilon 0.05 and an exploratory bomb can be fatal, but it makes the training curve unsuitable evidence. The greedy evaluation—not training reward—is the valid result.

### 5. Runtime and submission constraints — passed for the evaluated policy, not yet packaged

- Inference explicitly forces `torch.device("cpu")` and uses `torch.inference_mode()`.
- Mean think time is about 0.176 ms; average per-round maximum is 0.285 ms for the candidate, with zero 0.5-second overruns. These are far below the limit, though the metadata's `think_max_ms` summary previously quoted as 13.8 ms should be understood as a worst observed step, not the mean of per-round maxima recomputed above.
- Agent code uses a relative model filename. The framework changes into the agent directory around callbacks, so the development callback loads without an absolute path.
- The development agent depends on environment switches to select the artifact. It is therefore not yet a robust frozen submission agent. Freeze it into a dedicated self-contained directory and test that directory separately before calling it tournament-ready.

### 6. Provenance risks — unresolved but not fatal to the narrow comparison

- Both evaluation metadata files say `39c770a-dirty`. The working tree contains extensive modified and untracked experimental state. The matching callback SHA and model SHA make the two supplied evaluations internally traceable, but the commit alone cannot reproduce them.
- Training metadata says clean `39c770a`, while the evaluated callback is a later dirty version. This does not invalidate the policy comparison because both models were evaluated through byte-identical callbacks, but full training reproduction requires preserving the exact current source files and artifact hashes, not just the commit.
- Training metadata records the source model filename but not its SHA at training start. The current source artifact is internally consistent and later evaluations repeatedly reproduce the same baseline result, but there is no cryptographic proof in the training metadata that those exact source bytes were loaded at episode 1.

### 7. Confounds and generalization limits — claim must be narrowed

The candidate differs from the baseline after **both** the reward change and 2,000 more learning episodes. Fine-tuning resets replay, Adam state and target state. Therefore:

- Supported: “This trained candidate policy beats this baseline policy on the paired 1,000-round Task-2 evaluation.”
- Unsupported: “Increasing the coin reward to +1.5 caused +1.653 score.”
- Required causal control: repeat the same 2,000-episode continuation from the identical wait003 source with coin reward +1.0 and otherwise identical resets/configuration.

Only seed 11 trained the candidate. A single deterministic training outcome cannot show robustness to DQN initialization/data-order variance. At minimum, train matched +1.0/+1.5 continuations on additional seeds before presenting the reward setting as generally superior. Task 2 has no opponents, so evaluation pairing itself is strong; the principal uncertainty is training-seed variance, not opponent randomness.

The candidate is the highest-scoring n>=1000 Ben Task-2 CSV found in this result directory (7.699 versus the next 6.148), and it beats the named wait003 predecessor. That directory scan is not an exhaustive global model comparison and does not compare all team agents under one current callback/evaluation protocol.

## Final disposition

1. **Confirmed:** select the +1.5 artifact as the current *provisional Ben Task-2 DQN policy candidate* over the specified wait003 baseline.
2. **Rejected:** describe the experiment as proof that +1.5 coin reward itself is the cause.
3. **Not yet confirmed:** cross-seed robustness, global-best status, or submission readiness.
4. **Before freezing:** create a dedicated fixed CPU inference folder, copy the audited tensor-identical artifact, remove environment-dependent selection, run its tests, and perform a provenance-recorded evaluation.
5. **Before making a scientific reward claim:** run a +1.0 continuation control and preferably matched additional training seeds.
