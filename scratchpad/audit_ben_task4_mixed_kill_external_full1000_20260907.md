# Adversarial audit: mixed-kill Task 4 vs Task-3 incumbent, external Full1000

Date: 2026-09-07  
Scope: independent reconstruction of the two named CSV/.meta.json pairs, model/callback provenance, assignment, completeness, score decomposition, death causes, opponent randomness, timing, and the borderline kills result. No agent code or result file is modified.

## Claim under attack

`ben_task4_mixed_kill_v1_2000ep_seed11.pt` has a robust score advantage over `dqn_task3_seed13.pt` in the external-top-3 Full1000 evaluation.

## Audit status (incremental)

- Both requested CSV files and both metadata files exist.
- Each metadata file declares 1,000 rounds, base seed 20260731, classic scenario, the same three external opponents, and the same game settings.
- The evaluations were separate runs about 2 h 19 min apart, not a simultaneous A/B game. Arena seeds align, but opponent action streams are not guaranteed to align because external agents may use randomness not controlled identically across separate processes.
- Both runs were made at git state `3dac0c5-dirty`; exact reproduction therefore depends on hashes and retained dirty source state, not the commit alone.
- Numerical, methodological and provenance checks are complete; the final verdict appears below.

## Evidence and reconstruction

### 1. Completeness and assignment

Both CSVs contain exactly 4,000 data rows: 1,000 rounds numbered 0--999 and four slots per round. The seed is exactly `20260731 + round`; there are no missing or duplicate `(round, slot)` pairs. Each named agent occurs exactly 1,000 times and the evaluated policy is always in slot 0 (`dqn_task3` in the incumbent file, `ben_task4` in the candidate file). The three external agents and their order are identical. Thus no missing-round, wrong-slot, or wrong-agent explanation was found.

Both metadata files declare the same classic settings (`17x17`, 9 coins, crate density 0.75, bomb power/timers 3/4/2, 400 steps, kill reward 5, coin reward 1, timeout 0.5 s), the same base seed and the correct labels. Each CSV agrees with its metadata.

### 2. Raw reconstruction of the target rows

| Metric | Task-3 Seed 13 | Mixed-kill Seed 11 | Difference |
|---|---:|---:|---:|
| score | 1,919 / 1.919 | 2,192 / 2.192 | +273 / **+0.273** |
| coins | 1,724 / 1.724 | 1,902 / 1.902 | +178 / +0.178 |
| kills | 39 / 0.039 | 58 / 0.058 | +19 / +0.019 |
| suicides | 478 / 0.478 | 429 / 0.429 | -49 / -0.049 |
| killed by opponent | 185 / 0.185 | 127 / 0.127 | -58 / -0.058 |
| died | 663 / 0.663 | 556 / 0.556 | -107 / -0.107 |
| survived | 337 / 0.337 | 444 / 0.444 | +107 / +0.107 |
| won | 118 / 0.118 | 163 / 0.163 | +45 / +0.045 |

Every one of the 8,000 rows satisfies `score = coins + 5*kills`. Hence the target score difference decomposes exactly as

`+0.273 = +0.178 coins + 5 * (+0.019 kills)`.

About 65% of the score gap is therefore coin score and 35% is the point value of the observed kill gap. The result is not primarily evidence of more kills.

For all 2,000 target-agent rows, `died = 1 - survived`. `suicides` and `killed_by_opponent` are disjoint and sum to `died`; there are no double-labelled target deaths. The lower mortality is split between 49 fewer suicides and 58 fewer opponent-caused deaths. No `GOT_KILLED`/`KILLED_SELF`-style double counting is present in these evaluation columns.

### 3. Independent statistical checks and the borderline kill row

On the 1,000 arena-indexed score differences, the mean is +0.273, SD 2.026 and SE 0.0641. An independently computed paired t interval is `[+0.147, +0.399]` with two-sided p = 0.0000223. `tools/analyze.py` gives bootstrap CI `[+0.150, +0.399]`, t = 4.26 and sign-flip p reported as 0.0000. Treating the two samples as unpaired is deliberately conservative about trajectory matching; the Welch interval is still `[+0.128, +0.418]`, p = 0.000219. Thus the numerical score gap in these two files survives both paired-by-arena and unpaired analyses.

The kill row does not. Its difference is only 19 events per 1,000 rounds. Of the paired arena differences, 909 are zero, 56 are +1, 33 are -1 and two are -2; the candidate never records a double kill, whereas the incumbent does twice. The paired t interval is `[-0.00030, +0.03830]`, p = 0.0537. `analyze.py` prints bootstrap `+0.019 [+0.000, +0.039]`, but its sign-flip p is 0.0688 and its verdict is `no effect shown`; alternative bootstrap seeds can put the lower endpoint at about -0.001. The displayed lower `+0.000` is a boundary/rounding result, not stable evidence that zero was excluded. The unpaired Welch interval also crosses zero (`[-0.00024, +0.03824]`, p = 0.0529). A claim of a kill advantage is therefore refuted/not demonstrated by this Full1000.

The other paired t checks agree directionally with the raw counts: suicides -0.049 (`[-0.091, -0.0069]`), opponent-caused deaths -0.058 (`[-0.089, -0.027]`), survival +0.107 (`[+0.065, +0.149]`), and wins +0.045 (`[+0.016, +0.074]`). These are secondary diagnostics; tournament score remains primary.

### 4. Opponent randomness and causal scope

The evaluations are separate processes, not common-random-number replications. `evaluate.py` reseeds `world.rng` and NumPy per round, but does not seed Python's stdlib `random`. At least the external Xiaoxiae callbacks import stdlib `choice`; the external Lijesse agent also imports `random` and uses NumPy tie-breaking. Therefore the same arena seed does not guarantee the same external-agent random stream in the two files.

The observed opponent outcomes confirm that trajectories are far from identical. Across the two runs, equal opponent scores occur in only 23.2%, 28.0% and 25.3% of rounds for Lijesse, Bindist and Binary respectively; equality of a broader outcome vector is rarer still. Their mean scores all happen to be lower in the candidate run (3.985 to 3.675, 3.199 to 3.129, 3.459 to 3.224). Some of that can be a genuine interaction with the candidate, so it cannot be subtracted as mere noise; conversely, without controlled opponent RNG it cannot be cleanly attributed to the model either.

Arena blocking remains useful and the score result also survives an unpaired analysis. The limitation is interpretive: this is strong evidence for a difference between these two realized Full1000 runs, not a clean common-random-number estimate of the policy-only counterfactual. It is also one trained seed per policy (different training seeds 11 and 13), so it cannot establish a reproducible mixed-kill training-family advantage.

### 5. Model and callback provenance

No model swap, symlink, default-model fallback, or agent-name mismatch was found.

- Task 3 metadata records `dqn_task3_seed13.pt`, SHA-256 `1285ac5c...4fd8b63`; the retained regular file matches byte-for-byte. Its callbacks hash `3a601738...25d11f` also matches both the current file and commit `3dac0c5`.
- Candidate metadata records `BM_TASK4_TRAINING_ARM=mixed_kill_v1`, total episodes 2000, training seed 11 and `BM_TASK4_MODEL_VARIANT=trained`. That loader branch selects `ben_task4_mixed_kill_v1_2000ep_seed11.pt` and rejects a missing trained file. The retained regular file matches metadata SHA-256 `d50ae3d1...cf0806c6` byte-for-byte.
- Both model files load as 11-channel PyTorch state dictionaries with the same architecture/key shapes and 118,790 parameters, but are not tensor-identical.
- The candidate training metadata names the same output file, arm, seed, planned 2,000 episodes and training opponents `peaceful_agent,rule_based_agent,rule_based_agent`. It does not contain a final model hash, so training-run-to-current-bytes provenance rests on the filename/tracked artifact and earlier records; evaluation-to-model-bytes provenance is directly protected by the Full1000 model hash.

There is nevertheless a material reproducibility gap. Both evaluations say `git_commit: 3dac0c5-dirty`. The candidate callback hash at evaluation was `4270ab56...12f4ef`; today's callback is `069ac573...d4ab6`, and commit `3dac0c5` contains a third version (`d0745ac1...dda742`). Earlier 2026-09-04 notes record the evaluation hash, supporting that it existed then, but the exact old callback bytes are not retained as a Git blob or current file. Moreover `evaluate.py` hashes `callbacks.py` and the model only, not `features.py`, `model.py`, the legal-action-mask implementation, or the complete dirty diff. Consequently the selected weights and loader branch are well evidenced, while the full inference function used on 2026-09-04 is not exactly reconstructible from Git plus metadata. This blocks the strongest reproducibility wording.

### 6. Timing

There are zero `think_over_limit` events in either target sample. Mean per-action think time is 0.2409 ms versus 0.2364 ms. The mean of each round's maximum is 1.8427 ms versus 1.3089 ms; the global observed maxima are 40.0093 ms and 47.7238 ms. Both are far below 500 ms. The candidate is tournament-safe in this sample, but the lower average `think_max_ms` should not be oversold: its single worst call is actually higher. Metadata wall clocks (4,562.2 s and 4,698.1 s) are total evaluation runtimes dominated by the whole four-agent game, not inference latency.

## Adversarial verdict

**The strongest claim is rejected; the narrow empirical claim survives.**

I could not refute that, in these two specific external-top-3 Full1000 files, the mixed-kill checkpoint scored more: +0.273 points/round is internally consistent, non-fragile in the repository analysis, and remains statistically separated from zero even under an unpaired check. There is no evidence of wrong model loading, incomplete data, wrong slot assignment, score arithmetic error, death double counting, or timeout failure.

I can refute calling this a generally **robust/model-causal mixed-kill advantage** without qualification. Only one training seed per policy was tested; external opponent randomness was not shared across runs; the candidate evaluation came from a dirty source tree whose complete feature/model code was not hashed and whose exact callback version is no longer retained. Most importantly, the tempting kill narrative is unsupported: the +0.019 kill row is exactly the borderline row on which the t/bootstrap display and sign-flip evidence fail to demonstrate an effect.

Defensible wording: **“The frozen mixed-kill Seed-11 checkpoint outscored the frozen Task-3 Seed-13 checkpoint by 0.273 points/round in one 1,000-arena external-top-3 evaluation; the gap is driven mainly by coins and lower mortality, while a kill-rate increase was not demonstrated. Because opponent RNG was not shared, only one training seed per policy was tested, and the dirty callback/feature provenance is incomplete, this is confirmation of these concrete checkpoints rather than evidence for a reproducible mixed-kill training advantage.”**

Before promotion or a report-level general claim, require at least multiple independently trained seeds, fresh evaluations with stdlib and NumPy opponent RNG explicitly controlled or repeated across several opponent-RNG seeds, and a provenance snapshot hashing all inference-relevant source files.
