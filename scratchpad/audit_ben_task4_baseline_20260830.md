# Adversarial audit: frozen `dqn_task3` Task-4 baseline (2026-08-30)

## Scope and attempted refutation

I independently inspected the frozen agent, the 100- and 1,000-round CSV/meta pairs, the evaluator and framework death/score semantics. I specifically tried to invalidate a claim that `dqn_task3` "beats `rule_based_agent`" by checking completeness, provenance, score arithmetic, death attribution, timing, opponent randomness, within-game dependence, and slot/start-position assignment.

## Artifact and provenance checks

- Quick run: 400 rows, 100 distinct rounds, exactly 100 rows for our agent and each of three rule-based instances.
- Main run: 4,000 rows, 1,000 distinct rounds, exactly 1,000 rows per participant. Seeds are the prescribed `20260731..20261730`, scenario is `classic`, and the lineup is one `dqn_task3` plus three `rule_based_agent` copies.
- Both metadata files identify model `dqn_task3_seed13.pt` with SHA-256 `1285ac5cb78a25b0e6cc6a0e0a68fdd86e153db537832938f940cb1ad4fd8b63`; this exactly matches the current frozen model. The callbacks hash also matches (`3a601738...925d11f`). The model is loaded with `map_location=cpu`, the network and input tensor remain on CPU, and inference uses `torch.inference_mode()`.
- Every row satisfies `score = coins + 5*kills`. Every death row satisfies `died = 1-survived` and `killed_by_opponent = died-suicides`. This matches the framework: own-bomb death increments both `KILLED_SELF` and `GOT_KILLED`, while another owner's blast only causes `GOT_KILLED` for the victim.
- Metadata records `39c770a-dirty`; `tools/evaluate.py` is modified and the frozen agent/results are currently untracked. Hashes preserve the model/callback identity, but the commit alone cannot reproduce the exact working tree or evaluator. This is a provenance limitation until the user commits the reviewed state.

## Independent 1,000-round reconstruction

| participant | score | coins | kills | suicides | killed by opponent | survived |
|---|---:|---:|---:|---:|---:|---:|
| `dqn_task3` | 3.248 | 2.568 | 0.136 | 0.474 | 0.132 | 0.394 |
| rule-based instance 0 | 3.022 | 2.117 | 0.181 | 0.438 | 0.112 | 0.450 |
| rule-based instance 1 | 3.064 | 2.089 | 0.195 | 0.473 | 0.081 | 0.446 |
| rule-based instance 2 | 3.024 | 2.164 | 0.172 | 0.454 | 0.077 | 0.469 |

Against the per-round mean of the three rule-based opponents, our differences are:

- score `+0.211`, bootstrap 95% CI `[+0.034,+0.390]`, `t=2.31`, sign-flip `p=0.0208`, not flagged fragile;
- coins `+0.445` `[+0.334,+0.555]`;
- kills `-0.047` `[-0.074,-0.018]`;
- suicides `+0.019` `[-0.016,+0.055]` (no demonstrated difference);
- killed by opponents `+0.042` `[+0.018,+0.066]`;
- survival `-0.061` `[-0.097,-0.026]`;
- win-rate difference `+0.015` `[-0.021,+0.051]` (secondary and not demonstrated).

Thus the small score advantage is entirely a coin advantage and exists despite demonstrably fewer kills, more opponent-caused deaths, and lower survival. The agent is not generally stronger on the combat/safety dimensions.

## What breaks the broad claim

1. **The defensible claim is narrower than "beats rule_based_agent".** This run supports: in this one-versus-three lineup and this 1,000-round realization, our expected per-round score exceeded the mean score of the three opponents by 0.211. It does not show superiority to every rule-based instance robustly: separate comparisons are borderline/non-significant for two instances, and one is explicitly fragile (`+0.226`, CI lower bound `+0.001`, sign-flip `p=0.0523`).
2. **Opponent trajectories are only partly reproducible.** `evaluate.py` reseeds NumPy each round, but `rule_based_agent` also uses stdlib `random.shuffle`, which is not reseeded. Arenas are matched; full opponent behavior is not. The project measured roughly a `0.12` score noise floor for repeated 1,000-round Task-4 evaluation of the same table. The observed `+0.211` exceeds that heuristic floor but is close enough to the significance boundary that one draw should not be promoted to a final "beats" claim without replication.
3. **The comparison is a valid matchup measurement, not an isolated agent-quality experiment.** Scores of the four agents in a round are dependent: they kill, block, and deny coins to each other. Comparing ours with the mean of the three opponents is meaningful for total-score tournament performance in exactly this field, but the three opponent rows are not three independent replications. The analysis correctly reduces them to one per-round opponent mean before inference.
4. **Start corners are not fixed to slot 0.** The framework permutes all four symmetric starting corners every round, so ours does not always occupy the same corner. However, ours is always agent-list slot 0 and therefore retains a fixed activation/order identity; the three rule-based copies occupy slots 1-3. Their pooled scores are close (`3.022/3.064/3.024`), which gives no obvious large rule-based slot gradient, but it cannot rule out an interaction specific to slot 0 or agent type.
5. **Timing is safe in this run.** Our per-round maximum-step latency averages `0.270 ms`, no step exceeded 500 ms, and the largest observed per-round maximum is far below the limit. The CSV summary's `think_max_ms` row is the mean/CI of per-round maxima, not the single global maximum; any final report should state the actual observed global maximum separately.

## Verdict

The data and arithmetic are sound, provenance identifies the exact frozen policy, and the pre-specified primary metric favors `dqn_task3` over the opponent mean in this particular 1,000-round matchup. The direct paired result is internally non-fragile by the repository's CI/sign-flip checks. Nevertheless, **one partially reproducible evaluation is insufficient for the broad statement that the agent beats `rule_based_agent`**, especially with a small advantage, fixed list slot, materially worse kills/survival, and individual-opponent contrasts that do not all hold.

The correct current statement is: **a promising Task-4 baseline with a demonstrated +0.211 score advantage over the within-round mean of three rule-based opponents in one 1,000-round draw; replication is pending.** It is not yet a Task-4 completion or stopping result.

## Next measurement designed to refute

Before changing or training anything, run a fresh 1,000-round evaluation of the identical frozen model and lineup under a new result label. Require the score advantage over the per-round opponent mean to remain positive and non-fragile; report both draws rather than replacing the first. If it replicates, perform an agent-list-order check with `dqn_task3` in another slot (the framework already randomizes physical corners) to separate policy strength from activation-order identity. Only after those checks should the project claim that the frozen Task-3 policy beats this rule-based field or use the baseline to justify Task-4 reward/feature changes.

## Replication follow-up (`retry1`)

The independent retry contains 4,000 rows, 1,000 complete rounds and exactly 1,000 observations for each participant. It uses the same prescribed arena seeds, `classic` rules, lineup, callbacks hash and exact model SHA as the first run. Score and death arithmetic are valid in every row; there are no timeout overruns, and the largest observed DQN step time is `10.162 ms`.

Retry means are:

- `dqn_task3`: score `3.409`, kills `0.153`, suicides `0.478`, killed by opponents `0.120`, survival `0.402`;
- rule-based slots 1/2/3: score `3.006/2.962/3.025`, kills `0.182/0.181/0.176`.

Against the within-round mean of the three rule-based opponents, retry1 gives score `+0.411` with bootstrap 95% CI `[+0.219,+0.601]` and sign-flip `p<0.0001`; it is not fragile. The effect again comes from coins (`+0.545`), not clearly from kills (`-0.027`, CI includes zero). Our survival remains lower (`-0.041`, CI `[-0.076,-0.005]`), and opponent-caused death remains higher (`+0.024`, lower CI approximately `+0.001`).

The original and retry DQN scores differ by `+0.161`, CI `[-0.023,+0.351]`; no run-to-run score change is demonstrated. Zero of 1,000 rounds has an identical full four-agent outcome vector across the two executions, directly confirming that identical arena seeds do not reproduce the stdlib-shuffled opponent trajectories. This variability did not overturn the direction: the primary score-over-opponent-mean result is positive, non-fragile and practically larger in both independent draws (`+0.211`, then `+0.411`).

### Updated verdict

The narrow claim **replicates**: in two independent 1,000-round draws with our agent in list slot 0, frozen `dqn_task3` outscores the within-round mean of three `rule_based_agent` opponents. It is now reasonable to describe this as replicated evidence for this exact lineup, while reporting both estimates rather than selecting retry1's larger value.

The broader slot-independent claim remains untested. Physical corners are randomized, but agent activation/list order is unchanged in both runs. Therefore the next measurement should indeed be a slot rotation before training. For a clean check, place `dqn_task3` once in each of list slots 1, 2 and 3 over the same 1,000 arena seeds, keep all runs regardless of outcome, and aggregate the four slot identities rather than selecting the best. This tests whether the replicated advantage survives activation order; it is more diagnostic than modifying the policy now.

## List-slot rotation follow-up

All three pre-specified rotations are present and complete. Each CSV contains 4,000 rows, 1,000 distinct rounds, exactly 1,000 observations per participant, and the prescribed seeds `20260731..20261730`. Metadata orders are correct:

- slot 1: `rule_based_agent, dqn_task3, rule_based_agent, rule_based_agent`;
- slot 2: `rule_based_agent, rule_based_agent, dqn_task3, rule_based_agent`;
- slot 3: `rule_based_agent, rule_based_agent, rule_based_agent, dqn_task3`.

Every DQN row carries the intended list slot. All three runs use the exact frozen model SHA and callbacks SHA from the slot-0 runs. Score/death identities hold for every row, no action exceeded the 500 ms limit, and the largest observed DQN step time across the rotations is `13.851 ms`.

### Per-slot score result

The comparison remains one DQN score minus the **within-round mean** of the three rule-based scores; the rule-based copies are not treated as independent samples.

| DQN list slot | DQN score | rule-based pooled score | difference (95% bootstrap CI) | sign-flip p | fragile |
|---:|---:|---:|---:|---:|---:|
| 0, original | 3.248 | 3.037 | `+0.211 [+0.034,+0.390]` | 0.0208 | no |
| 0, retry | 3.409 | 2.998 | `+0.411 [+0.219,+0.601]` | <0.0001 | no |
| 1 | 3.351 | 2.970 | `+0.381 [+0.190,+0.583]` | 0.00015 | no |
| 2 | 3.280 | 3.029 | `+0.251 [+0.064,+0.440]` | 0.0096 | no |
| 3 | 3.372 | 2.940 | `+0.432 [+0.236,+0.633]` | <0.0001 | no |

No slot reverses or makes the primary result fragile. To avoid falsely treating 5,000 repeated uses of the same 1,000 arena seeds as independent, I aggregated **within arena seed first**. I averaged the two slot-0 replications, then weighted slots 0/1/2/3 equally. That position-balanced estimate is `+0.344`, 95% CI `[+0.257,+0.431]`, sign-flip `p<0.0001`, non-fragile. The rotations alone give `+0.355 [+0.247,+0.465]`. Consequently neither the extra slot-0 replication nor a naive 5,000-row calculation is needed to obtain the conclusion.

### Components and attempted refutation

The position-balanced aggregate again exposes a tradeoff rather than general dominance:

- coins: `+0.506 [+0.455,+0.558]`;
- kills: `-0.033 [-0.046,-0.019]`;
- suicides: `+0.039 [+0.022,+0.055]` (worse);
- killed by opponents: `+0.014 [+0.004,+0.024]` (worse);
- survival: `-0.053 [-0.069,-0.037]` (worse);
- secondary win rate: `+0.056 [+0.039,+0.074]`.

The score advantage therefore survives every attempted slot refutation, but it is a **coin-efficiency advantage that overcomes worse combat and safety**, not evidence that the DQN hunts or survives better. This distinction matters for choosing the Task-4 training objective.

The RNG limitation remains: all rotations reuse the same arena seeds, while stdlib `random.shuffle` leaves rule-based trajectories only partly reproducible. Repeated arenas are why the aggregate above clusters by arena rather than presenting 5,000 independent rounds. The consistency across all slots and both slot-0 draws makes opponent noise an implausible explanation for the positive score direction, but these measurements still describe this exact one-DQN/three-rule-based field. They do not establish performance against heterogeneous tournament entrants or a universal head-to-head ordering.

### Final baseline verdict and training signal

The claim can no longer be broken on list position: **frozen `dqn_task3` position-robustly outscores the within-round mean of three `rule_based_agent` opponents in this evaluation field.** This is replicated in all four list slots, with a position-balanced score advantage of about `+0.344`.

Together the five runs justify beginning controlled Task-4 development, but not because baseline score is inadequate. The diagnostic signal is specifically that roughly 60% of DQN rounds end in death, around half in suicide, while kills trail the rule-based mean. The safest first Task-4 hypothesis is therefore to reduce deaths—especially own-bomb deaths—without sacrificing the demonstrated coin advantage and total score. Any candidate must be evaluated in the same rule-based field across training seeds and compared against this frozen baseline; success remains higher total score, with suicides/killed-by as diagnostics and regression guards rather than substitute objectives.
