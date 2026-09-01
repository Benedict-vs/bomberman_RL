# Adversarial audit: Task 3 `mixed_curriculum_v1`

Date: 2026-08-30

## Audit question

Find what is wrong with the apparent conclusion that the mixed-curriculum candidate should replace the safe `coin_collector_finetune_v1` baseline. In particular, verify the artifacts and then distinguish the preregistered decision from a practical score/safety tradeoff.

## Provisional result (written before detailed interpretation)

The candidate must **not replace the safe baseline under the preregistered rule**. Its score improvement is large and statistically non-fragile in these evaluations, and survival passes the floor, but suicides are 25.9%, exceeding the explicitly preregistered maximum of 25%. It may be retained as a distinct higher-score/lower-safety candidate, not relabelled as the safe successor.

## Artifact checks

- The training CSV exists and contains exactly 2,000 unique episodes, numbered 1 through 2,000.
- Twenty expected checkpoints exist at 100-episode intervals through episode 2,000.
- The final model is SHA-256 `d5c083f918756376abbdd91fb9e50c506a5b5632c3a0dbdd6f8f63d78eea1680`.
- The episode-2,000 checkpoint has a different file SHA (`6ff492...`) but its ten state-dict tensors are exactly equal to the final model (same keys, maximum absolute difference 0). Thus the differing container bytes are not differing policies.
- The source model recorded in training metadata is `ben_task3_coin_collector_finetune_v1_2000ep_seed11.pt`, whose actual SHA is `a9bd850a8b3eb4052b3c56aa87d39235b674e3b01cd5532325e3805e15f6c8f5`, matching the audited safe source.
- The mixed evaluation metadata records the actual mixed model path and matching SHA. It contains 4,000 rows: 1,000 rounds for `ben_task3` and 1,000 for each of three coin-collector slots, with 1,000 unique rounds for our agent.
- The baseline evaluation is likewise complete (4,000 rows, same 1,000 arena seeds and slots).
- Both evaluations use `classic`, seeds 20260731 through 20261730, and unchanged recorded game settings.

## Implementation and training provenance

`mixed_curriculum_v1` selects the safe model as `LOAD_MODEL_FILE`. Source inspection and metadata agree that rewards and features were unchanged: 11 channels, reachable-safe escape feature, visit count, safety potential 1.0, opponent potential 0, coin reward 1.5, crate reward 0.3, kill reward +5, `GOT_KILLED` -5 and `KILLED_SELF` 0. The training score column is exactly `coins + 5*kills` for all 2,000 episodes.

The metadata says the intended training opponents were `peaceful_agent,coin_collector_agent,coin_collector_agent`, consistent with the supplied launch command. This is not independently proven by the framework artifact: `training_opponents` is a self-recorded configuration string and the quiet `game.log` is empty. Therefore the lineup provenance is consistent but not independently reconstructable from per-round training records.

The evaluation callbacks hashes differ (`567f77...` baseline versus `035996...` mixed), so callback identity cannot be claimed cryptographically. Current source inspection indicates the intervening additions concern arm/model/checkpoint selection rather than a feature or action-semantic change, and both metadata files record the same feature modes. Still, “same callback binary” is false; equivalence is source-level, not hash-level.

## Recomputed evaluation results

All score values exactly satisfy `score = coins + 5*kills`.

| Metric | safe baseline | mixed candidate | paired difference |
|---|---:|---:|---:|
| score | 1.801 | 2.193 | +0.392, 95% CI [+0.225, +0.558] |
| coins | 1.261 | 1.593 | +0.332 |
| kills | 0.108 | 0.120 | +0.012, CI includes 0 |
| suicides | 18.3% | 25.9% | +7.6 pp, 95% CI [+4.0, +11.2] |
| killed by opponent | 3.9% | 3.0% | -0.9 pp |
| survival | 77.8% | 71.1% | -6.7 pp, 95% CI [-10.5, -2.8] |
| bombs | 13.202 | 15.430 | +2.228 |
| crates | 30.791 | 35.887 | +5.096 |

The score result is not marked fragile: its bootstrap CI excludes zero, its sign-flip permutation p is 0.0001, and the t statistic is 4.62. The survival and suicide regressions are likewise non-fragile. The gain is predominantly coins (+0.332 of +0.392 score), not demonstrated additional kills (+0.060 score contribution from the point estimate).

The candidate's own marginal 95% intervals are suicides [23.2%, 28.7%] and survival [68.3%, 73.9%]. The preregistered thresholds were defined on the reported point estimates, so 25.9% fails `<=25%` and 71.1% passes `>=70%`. One must not retroactively relax the suicide boundary because the estimate is close; conversely, the interval crossing 25% means the true-policy suicide rate is not known to be above 25% with 95% confidence. The strict experimental decision and population uncertainty are different statements.

## Pairing, noise, and confounds

The arena seed and slot match for every paired row, but the complete opponent outcomes match in **0/1,000** rounds. The provided opponents also use unseeded stdlib randomness. Therefore the nominal paired CI treats arena-aligned but trajectory-unsynchronised observations as paired; it does not cancel opponent noise. The known approximately +/-0.12 repeat-evaluation score noise floor is smaller than +0.392, so the effect is practically substantial rather than a tiny noise-floor result, but it is not an exact same-trajectory policy contrast.

Only training seed 11 was used. There is no same-length control that continues the safe source for 2,000 episodes against three coin collectors, nor a control with another lineup. The experiment therefore compares two resulting policies and does **not** isolate “one peaceful plus two coin collectors” as the cause: another 2,000 episodes, a fresh replay buffer, optimizer/target reset, opponent randomness, and the changed training distribution all travel together. Generalisation across training seeds is untested, contrary to the normal rung-3 standard (the user explicitly declined expensive additional seeds earlier).

The 100-round quick screen preceded the 1,000-round confirmation. The larger run is appropriate confirmation, but the candidate was selected through a development sequence involving several failed seed-11 arms. Its result should therefore be described as a selected single-seed candidate, not an unbiased estimate of curriculum design effectiveness.

The candidate still scores below every coin-collector opponent in this evaluation (their means are 2.657 to 2.805). Thus it is progress relative to our safe baseline, not evidence that Task 3 is finished or that the agent beats this opponent field.

## Verdict

1. **Strict preregistered verdict: reject as replacement.** It fails one mandatory gate: suicides are 25.9%, above the maximum 25%. Survival passes narrowly at 71.1%, and score improves convincingly, but the rule required all three conditions.
2. **Practical interpretation: retain as a distinct tradeoff candidate.** It offers a real-looking +0.392 score gain at the cost of demonstrably worse own-bomb safety and survival. Because tournament score is primary, this policy is scientifically and practically interesting; deleting it would discard useful evidence. It must not overwrite or displace the frozen safe agent.
3. **Do not claim curriculum causality or robustness.** The evidence supports only that this concrete seed-11 mixed-trained policy scored higher in this evaluation while being less safe. It does not prove the mixed lineup caused the change, that it will reproduce across training seeds, or that it is the best Task-3 policy.

The defensible handling is therefore to preserve both policies with explicit labels: the existing safe baseline and the mixed higher-score/safety-regression candidate. Any later promotion would require a predeclared decision that explicitly prioritises score over the old safety gate, plus independent confirmation (ideally another training seed or at minimum a repeated held-out evaluation acknowledging opponent noise). The original preregistration itself must remain recorded as failed, not rewritten.
