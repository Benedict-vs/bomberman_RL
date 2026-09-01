# Adversarial audit: Task 3 opponent-potential candidate (2026-08-29)

## Scope and attempted refutation

This audit asks only whether the concrete Seed-11 policy
`ben_task3_coin_collector_opponent_potential_v1_2000ep_seed11.pt` should replace
the safe `coin_collector_finetune_v1` baseline against three
`coin_collector_agent` opponents. It deliberately does not assume that the
potential helped and does not infer general effects from the training curve.

## Artifact status

- The training CSV has exactly 2,000 episode rows plus its header. All 20
  expected 100-episode checkpoints exist, including episode 2,000.
- The final model SHA-256 is
  `397f356e104fdaaeba45bd83e63b93ba42983f69b4d58eb9fd6ccf54058fff22`,
  matching the model recorded by both candidate evaluation metadata files.
- The final model and episode-2,000 checkpoint have different container-file
  hashes, but every state-dict key and tensor is exactly equal (`maxdiff=0`).
- The 1,000-round candidate and baseline CSVs each contain exactly 4,000 rows,
  1,000 rows for `ben_task3`, and all seeds 20260731 through 20261730.

## Implementation semantics

- The arm loads `ben_task3_coin_collector_finetune_v1_2000ep_seed11.pt`, keeps
  `KILLED_OPPONENT=+5`, and sets only `OPPONENT_POTENTIAL_SCALE=1.0` relative
  to that configured source. Its metadata agrees (11 channels, seed 11,
  gamma 0.99, scale 1.0, source filename, and separate destination filename).
- The shaping expression is genuinely state-potential based:
  `scale * (gamma * Phi(new_state) - Phi(old_state))`. `Phi` is negative
  shortest-path distance divided by 32, so decreasing the distance produces a
  positive transition reward. It does not return or encode a best action.
- Terminal transitions use `Phi(None)=0`; deaths missing from
  `game_events_occurred` are explicitly appended as terminal transitions, and
  surviving last transitions are marked terminal. Thus the algebra is the
  standard episodic potential form. A practical subtlety is that a negative
  potential produces a positive terminal boundary term; this is not by itself
  an implementation error, but it makes correct terminal bookkeeping essential.
- “Reachable” is weaker than the name may suggest. The BFS respects stone walls
  and crates and can terminate on an opponent tile, but ignores bombs,
  explosions, dynamic collision with other opponents, and time-to-arrival.
  Therefore it measures static crate-free proximity, not safe or temporally
  reachable hunting distance. That limitation can plausibly reward dangerous
  approaches and prevents interpreting the arm as a demonstrated safe-chase
  mechanism.
- The focused test checks source/destination selection, reward/scale, distance,
  and the signs of approach/retreat shaping. It does not test terminal shaping,
  unreachable targets, bombs/explosions, target switching, or policy invariance.

## Evaluation findings

The repository comparison (candidate minus safe baseline, 1,000 nominally
paired arenas) reports:

| metric | safe baseline | candidate | difference (95% CI) | audit reading |
|---|---:|---:|---:|---|
| official score | 1.801 | 1.828 | +0.027 [-0.126, +0.178] | no effect shown |
| coins | 1.261 | 1.383 | +0.122 [+0.057, +0.188] | higher in these runs |
| kills | 0.108 | 0.089 | -0.019 [-0.047, +0.009] | no effect shown |
| suicides | 18.3% | 21.7% | +3.4 pp [-0.1, +6.9] | no effect shown |
| killed by opponent | 3.9% | 3.3% | not reported above | small nominal change |
| survived | 77.8% | 75.0% | -2.8 pp [-6.7, +1.0] | no effect shown |
| invalid actions | 0.399 | 0.222 | -0.177 [-0.227, -0.129] | lower in these runs |

The score row is non-fragile but plainly includes zero (sign-flip p=0.7380).
The candidate satisfies the preregistered guardrails of at least 70% survival
and at most 25% suicides, but satisfying guardrails is not evidence that it is
better. Its extra coins are offset by fewer kills in the official score.

The quick-100 result (score 2.27 versus 1.95) did not persist as a demonstrated
score advantage at 1,000 rounds. It was correctly only a screen.

## Comparability and threats to inference

- Arena seeds, scenario, framework rules, opponent callback hash, slots, and
  evaluation size match. Actual candidate provenance is correct.
- The `ben_task3` callback hashes differ between the two 1,000 evaluations
  (`567f...` baseline versus `c9bb...` candidate), because the later code adds
  experiment selection/implementation. The inspected inference path appears
  unchanged for the safe arm, but byte-identical callback semantics are not
  proven by metadata. Any claim that these two files used identical callbacks
  would be false.
- More importantly, zero of 1,000 rounds had identical aggregate opponent
  outcomes across the two runs. The supplied opponents use stdlib randomness
  that the evaluator does not fully synchronize. “Paired” therefore means
  common arena seeds/slots, not common opponent trajectories. Small differences
  must not be treated as clean per-arena causal effects; the documented roughly
  +/-0.12 Task-3 score noise floor is much larger than +0.027.
- Only training seed 11 exists. There is no same-duration continuation control
  from the safe model with potential scale zero. The candidate simultaneously
  adds 2,000 learning episodes and starts fresh replay, optimizer, target state,
  and training trajectories. Consequently neither the extra coins nor any
  other behavioral change can be causally assigned to the potential alone.

## Verdict

**Do not replace the safe baseline with this concrete candidate.** The required
primary score improvement is not demonstrated, while kills, suicide rate, and
survival move nominally in the wrong direction. The candidate remains within
the safety guardrails and has interesting secondary gains in coins and invalid
actions, but those do not outweigh the absent score evidence.

This verdict is deliberately narrow. It does not show that opponent potentials
are generally harmful, nor that this candidate is definitively worse. It shows
only that the available single-seed, opponent-nonpaired evidence is insufficient
to select it over the already-safe model.

## Follow-up: preregistered checkpoint screen

### Provenance and selection discipline

The logbook selected exactly episode 1,200 (first higher-training-score region)
and episode 1,800 (a later second region) before greedy checkpoint evaluation,
with episode 2,000 already measured. This is a defensible bounded screen and is
materially less selection-biased than inspecting all 20 stored checkpoints.

Both evaluation metadata files point through the evaluation-only relative path
to the intended original checkpoints, not copied models:

- episode 1,200: recorded and actual SHA-256
  `aecae436a309803c923528f4756d05276d0b4585669853443a615382c8f6667f`;
- episode 1,800: recorded and actual SHA-256
  `72565d11e3d8a9ae0e9d25bf428707b6fee19b2de317ac625775fcec075164b5`.

The metadata also records the requested checkpoint episode, arm, seed, 2,000
episode run, 100 rounds, standard arena seeds, and identical opponent callback
hash. Thus checkpoint identity is sound.

There are actually two completed 100-round outputs for each checkpoint. The
first pair has an accidental newline/indentation in the label and filename; the
later episode-1,200 output is explicitly `retry1`, while the later 1,800 output
uses the clean intended name. These are independent opponent-randomness repeats,
not duplicated rows, and should be retained as such rather than silently merged.

### Results

Against the previously measured safe quick-100 reference (score 1.95, suicides
18%, survival 79%):

- Episode 1,200 scores 1.66 and 1.75 in its two runs. Kills are 0.06 in both;
  safety varies (23%/14% suicides and 72%/79% survival) but remains inside the
  preregistered 25%/70% guardrails. The clean retry comparison gives score
  difference -0.20 with 95% CI [-0.68,+0.28]. Nothing here motivates a costly
  confirmation, and its repeat pattern is directionally unfavorable.
- Episode 1,800 scores 2.07 in **both** independent runs. It also satisfies the
  safety guardrails in both (15%/19% suicides; 82%/79% survival). Relative to
  the single safe quick reference, the clean run has score +0.12 with 95% CI
  [-0.40,+0.63]; the first run gives +0.12 [-0.41,+0.65]. Kills are not higher
  (0.10/0.11 versus 0.12); the nominal gain comes from coins.

No checkpoint run shares even one of 100 complete opponent-outcome aggregates
with the safe reference. The nominal pairing is again arena/slot pairing only.
The broad intervals and unsynchronized opponent randomness mean n=100 can
easily hide a practically meaningful gain: for episode 1,800, effects as large
as roughly +0.63 score are still compatible with either quick comparison. The
observed +0.12 also sits exactly at the project's documented Task-3 opponent
noise floor. Therefore “not significant at 100” is not evidence of absence.

### Follow-up verdict

The proposed joint conclusion is **partly refuted**:

- **Episode 1,200 does not warrant 1,000-round confirmation.** Two independent
  screens are unfavorable, with no kill signal and no compensating primary
  score indication.
- **It is too strong to say episode 1,800 does not warrant confirmation.** It
  passes the preregistered guardrails twice, produces the same positive score
  screen twice, and the 100-round interval plainly cannot exclude a useful
  gain. Under the stated protocol (“at most one” checkpoint may advance), it is
  the one defensible checkpoint for a single 1,000-round confirmation. This is
  not a claim that it is better; only that the screen has not answered that
  question.
- **Stop screening additional checkpoints.** The two checkpoints were bounded
  and preregistered precisely to prevent post-hoc checkpoint fishing. Advancing
  episode 1,800 does not justify opening the other 17 unevaluated checkpoints.

Any 1,000-round episode-1,800 result remains a single-training-seed policy
comparison with non-common opponent randomness and cannot establish causal
benefit of the potential. Replacement still requires a demonstrated primary
score gain while retaining the safety guardrails.

## Final follow-up: episode-1,800 confirmation

The sole promoted checkpoint has now completed its preregistered 1,000-round
confirmation. The CSV is complete (4,000 rows, 1,000 rounds, seeds 20260731
through 20261730). Metadata selects `BM_TASK3_CHECKPOINT_EPISODE=1800` and the
relative original checkpoint path; its recorded SHA-256
`72565d11e3d8a9ae0e9d25bf428707b6fee19b2de317ac625775fcec075164b5`
matches the checkpoint on disk. Scenario, rules, opponent callback, arena seed
range, and inference callback match the checkpoint quick runs.

Independent reproduction of the comparison against the safe 1,000-round
baseline gives:

- official score `1.801 -> 1.779`, difference `-0.022`, 95% CI
  `[-0.169,+0.123]`, sign-flip p `0.7798`: no gain demonstrated;
- kills `0.108 -> 0.075`, difference `-0.033 [-0.060,-0.007]`, p `0.0171`:
  fewer kills in these runs;
- coins `1.261 -> 1.404`, difference `+0.143 [+0.077,+0.209]`, while the lost
  kill points more than cancel the coin increase in tournament score;
- suicides `18.3% -> 18.1%` and survival `77.8% -> 78.3%`: essentially
  unchanged and safely within the preregistered guardrails;
- invalid actions `0.399 -> 0.274`, lower, but again a secondary result that
  does not produce primary score improvement;
- observed maximum think time is 11.413 ms, far below 500 ms.

The quick-100 indication of +0.12 score did not replicate at 1,000 rounds.
As before, zero of 1,000 rounds have identical aggregate opponent outcomes, so
the opponent randomness is not truly paired and small deltas should not be read
as exact causal effects. This caveat weakens a claim that episode 1,800 is
globally worse, but it cannot rescue it as a replacement: its score evidence is
centered slightly negative, its CI includes zero broadly, and kills decline.

### Final checkpoint verdict

**Do not replace the safe baseline with episode 1,800, and stop this checkpoint
screen now.** Episode 1,200 failed the bounded screen; episode 1,800 was the only
checkpoint promoted and failed its full confirmation; episode 2,000 had already
failed to show a score gain. Evaluating any of the other checkpoints would break
the preregistered two-checkpoint bound and turn the exercise into post-hoc
checkpoint fishing.

The safe `coin_collector_finetune_v1` policy therefore remains the selected
Task-3 intermediate against these opponents. This does not establish that every
opponent-potential policy is ineffective: all candidates come from one training
seed, the potential effect is confounded with continuation training state, and
opponent trajectories are unsynchronized. It establishes only that none of the
three prospectively considered policies from this concrete run (episodes 1,200,
1,800, and the already measured 2,000 endpoint) earns replacement.
