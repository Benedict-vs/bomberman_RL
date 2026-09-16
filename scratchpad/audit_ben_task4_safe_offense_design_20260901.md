# Adversarial pre-run audit: Task 4 `safe_offense` (2026-09-01)

## Scope and verdict

Audited `safe_offense_zero_v1` against `safe_offense_reward02_v1`, planned as a
1000-episode Seed-11 pilot against three `rule_based_agent`s. I tried to falsify
the intended one-variable comparison. The implementation is sufficiently
isolated for a pilot: both arms use the same frozen 11-channel source, reward
table, hyperparameters and declared opponent lineup; the enabled arm alone adds
the `+0.2` safe-offense helper. The source SHA-256 is
`c9677540849340e915f74dec7cf9c2d24e9c6c27f9d5132453798035a1f8a9ec`.

This is **not yet a clean mechanism measurement**, however. The helper rewards a
hand-designed *opportunity*, not a hit or kill, and its occurrences are not
logged separately. A policy can therefore repeatedly place aligned bombs after
bomb recharge while opponents evade them. Bomb availability bounds the rate,
but does not prevent farming. Before a long or multi-seed run, add an episode
counter for helper activations (without changing the policy input); otherwise
interpret bombs-per-kill and total bombs as only indirect diagnostics. This does
not block the short pilot if that limitation is recorded in advance.

## Isolation, paths and provenance

- Both arms load
  `ben_task4_rule_based_continue_control1000_v1_1000ep_seed11.pt`, use 11 input
  channels, disable opponent-alignment, and retain identical ordinary rewards:
  coin `+1.5`, crate `+0.3`, kill `+5`, invalid `-1`, `GOT_KILLED=-5`,
  `KILLED_SELF=0`, step `-0.05`, plus the same inherited shaping.
- `SAFE_OFFENSE_BOMB_REWARD` is the only audited arm-dependent training value:
  `0.0` in the placebo and `0.2` in the candidate. Model, CSV, meta and checkpoint
  names contain the selected Task-4 arm, 1000 episodes and Seed 11. The checked
  targets and ten expected checkpoints were absent, so the proposed run does not
  overwrite them.
- Metadata records the arm, source model, source hash, input channels, ordinary
  rewards, helper scale and declared opponents. `TRAINING_OPPONENTS` is metadata,
  not enforcement: the actual command must explicitly pass
  `--agents ben_task4 rule_based_agent rule_based_agent rule_based_agent`.
- No dependency on `ben_task3` data was found in this pair. The source is an
  internal Task-4 model and outputs remain in Task-4 locations.

## Reward timing and semantics

The helper is called from `game_events_occurred` for ordinary transitions. A
lethal transition is not sent there by the framework and is handled in
`end_of_round`; the helper is called there only in the `GOT_KILLED` branch. For a
surviving terminal transition, `end_of_round` only marks the already stored
transition terminal. Thus code inspection finds no normal/terminal double
reward: each actual transition receives the helper at most once.

Eligibility requires all of the following in the pre-action state: action is
`BOMB`, the agent reports bomb availability, an opponent's current tile belongs
to the hypothetical blast, and the existing escape feature reports at least one
reachable safe tile after adding the hypothetical bomb. This correctly avoids
rewarding `WAIT`, unavailable bombs, distant opponents and positions with no
represented escape. It does **not** guarantee that the policy subsequently
takes the escape, that the opponent stays in line, or that the bomb causes a
hit. A terminal bomb action may therefore earn `+0.2` even if the later outcome
is death; that is consistent with opportunity shaping but must not be described
as a safe-bomb or kill reward.

The geometry uses `(x,y)` board coordinates and the same orthogonal,
stone-wall-stopped blast helper used by the Task-4 features. Crates do not stop
the blast in this framework implementation, so the helper is consistent with
the actual engine rather than a contrary informal Bomberman assumption.
Opponent tiles are passed as blocked tiles to the reachability calculation;
the source tile is the agent's legal current tile. The check inherits the
escape feature's approximations: it establishes an existing modeled route, not
safety under future opponent motion or every possible bomb interaction.

## ML rule and external-agent boundary

This is an action-dependent auxiliary reward, not a feature that outputs the
best action and not potential-based shaping. The DQN still has to learn a policy
from its legal 11-channel state, so the hard ML rule is not violated. The reward
will not exist in official games and its farming risk must be evaluated. Agents
from older years may be used as external training/evaluation opponents where
the project rules permit that, but none of their code, heuristics, features or
reward logic may be copied into the submitted agent. This experiment uses the
provided `rule_based_agent`, not copied old-year code.

## Tests and missing coverage

The actual selected configuration passed all 12 scaffold tests separately for
both arms. Tests confirm the expected positive case, `WAIT=0`, a distant
opponent, no modeled escape, identical source/channels/base rewards/opponents,
and the sole `0.0` versus `0.2` difference.

Coverage is weaker than the claim in three places: there is no explicit test
for `bomb_available=False`, no callback-level test proving exactly-once reward
at normal and lethal terminal boundaries, and no explicit wall/crate geometry
test for this helper. These are small targeted tests worth adding before a
multi-hour follow-up. Most importantly, tests cannot expose reward farming
without an activation counter or behavioral evaluation.

## Preregistered pilot gates

Compare candidate minus placebo within the same evaluation convention. Because
the opponents use unseeded stdlib randomness, pairing is on arenas only and a
single trained Seed-11 model is a pilot, not a general Task-4 conclusion.

1. Primary score effect must be positive, non-fragile, have a paired 95% CI
   excluding zero, agree with sign-flip `p<.05`, and exceed the known `0.12`
   opponent-noise floor in point estimate.
2. The intended mechanism also requires kills to be positive, non-fragile, with
   CI excluding zero and sign-flip agreement. A score gain without a kill gain
   does not validate safe offense.
3. Safety guards: suicides delta at most `+0.03`, `killed_by` at most `+0.02`,
   and survival at least `-0.03`; a demonstrated regression overrides the score
   gate. Inspect bombs and bombs-per-kill for farming, while acknowledging that
   the missing helper-activation counter prevents a direct audit.
4. `think_max_ms` must remain below 500 ms globally with zero timeout breaches;
   retain ample margin for the slower reference CPU rather than accepting a
   near-limit local maximum.

Use quick100 only as a smoke test and the full1000 for the pilot verdict. Do not
select checkpoints after seeing results. Only if all gates pass should the same
predeclared design be repeated across independent training seeds; one Seed-11
run cannot establish a rung-4 improvement.

## Post-run audit: Seed-11 pilot

### Artifact integrity

Both arms completed exactly 1000 sequential training rows (`1..1000`) and wrote
all ten scheduled checkpoints (`100..1000`). For each arm, the final model's
state tensors are byte-for-byte tensor-identical to its episode-1000 checkpoint
(the surrounding serialization files need not be byte-identical). The training
metadata selects the intended arm, Seed 11, 1000 episodes, three declared
`rule_based_agent`s, the same 11-channel source and the same ordinary rewards and
hyperparameters. The sole intended difference remains helper scale `0.0` versus
`0.2`. Final model hashes are:

- zero: `9adb0911b3dca54a0b7f80c40bb918d876375587c40308e840d4b6912751fbd8`
- reward02: `b8f3910e4261a6c35d965dcf3cc54bf1e9cda24187282f21b012ccb379c7cfdd`

Each full evaluation contains 4000 rows, 1000 complete rounds and exactly one
`ben_task4` in slot 0 plus three rule-based opponents in slots 1--3. The eval
metadata records the correct arm environment and corresponding final-model
hash, default arena seeds `20260731..20261730`, `classic`, and identical rules.
Score arithmetic is exact in every audited DQN row (`score = coins + 5*kills`),
and the death partition is exact (`survived + suicides + killed_by = 1`). The
quick100 files are complete but are smoke tests only; their candidate result was
not promising (score `-0.02`, kills `-0.01`) and illustrates why it cannot drive
selection.

### Full-1000 causal result and orientation

All differences below are **reward02 minus zero**, paired only on the common
arena seed. The primary analysis reports:

- score `3.884 - 3.531 = +0.353`, 95% CI `[+0.148,+0.567]`, sign-flip
  `p=0.0009`, non-fragile;
- kills `0.188 - 0.133 = +0.055`, CI `[+0.021,+0.090]`, `p=0.0026`,
  non-fragile;
- suicides `0.471 - 0.433 = +0.038`, CI `[-0.006,+0.081]`, `p=0.0943`;
- killed by opponent `0.087 - 0.081 = +0.006`, CI `[-0.019,+0.031]`,
  `p=0.6937`;
- survived `0.442 - 0.486 = -0.044`, independently bootstrapped CI
  `[-0.086,0.000]`, sign-flip `p=0.0514`; the boundary verdict is fragile.

Thus the score and kill mechanism is demonstrated in this evaluation, and the
score point effect is well beyond the `0.12` evaluation noise floor. It is not
valid to reverse the subtraction or call lower deaths an improvement here. The
opponents' stdlib RNG remains unseeded, so this is arena-paired rather than fully
behavior-paired evidence; one trained seed cannot establish generality.

### Trigger and farming audit

The new counter resolves the pre-run observability concern: zero logged exactly
0 triggers; reward02 logged 1655 triggers. Reward02 placed 12,829 training bombs,
so only 12.9% of training bombs collected the helper (1.655 triggers per
episode). Zero placed 12,278 bombs. The training arms had 65 versus 69 kills,
which is not itself evidence of learned benefit because training includes
exploration and different opponent trajectories.

At deterministic evaluation, reward02 placed `27.010` bombs/round versus
`21.824`, a clear `+5.186` (`[+4.173,+6.213]`, sign-flip `p<0.0001`) while also
raising kills by `0.055`. This does not prove farming: bombs per evaluated kill
actually improve from about 164.1 to 143.7. But the marginal scale is expensive
(roughly 94 additional bombs per additional kill), and the evaluation CSV
cannot say which bombs met the helper condition. The honest conclusion is
“no decisive farming signature, but substantially more bombing and opportunity
shaping remain plausible contributors”, not “farming ruled out”.

### Timing

There are zero timeout breaches. DQN per-round mean think time is about
`0.191 ms` (zero) and `0.188 ms` (reward02); mean row maxima are `0.631 ms` and
`0.457 ms`. The true global maxima across all four agents and all steps are
`27.423 ms` and `44.780 ms`, respectively, still far below 500 ms. The helper is
training-only, so no tournament inference-cost regression is expected or seen.

### Gate verdict and next experiment

This is a **formal preregistered fail with a useful offense--safety trade-off**,
not a full pass. Score and kills pass strongly, and killed-by plus timing pass.
However, the point-estimate guards were explicitly absolute: suicides `+0.038`
exceeds `+0.03`, and survival `-0.044` is below `-0.03`. Their intervals do not
establish a regression, but “CI includes zero” cannot retroactively replace the
predeclared point guard. Two of three safety guards therefore fail.

Running Seeds 12/13 of reward02 now would answer whether the *trade-off*
replicates, but it would spend the multi-seed budget on an arm that already
failed its acceptance rule. A smaller reward is scientifically motivated as a
dose-response attempt because the observed mechanism is positive and the harms
increase alongside much heavier bombing. It would nevertheless be a new,
post-result hypothesis, not a rescued preregistration. The clean next step is a
new controlled dose pilot from the same source, with a predeclared lower value
(for example `+0.1`) against the same zero control, identical 1000 episodes and
the same gates, while retaining trigger logging. Choose the dose once before
running; do not sweep values or checkpoints and select the winner. Only a dose
that passes offense **and** all safety gates should advance to Seeds 12/13.

## Pre-run check: preregistered `+0.1` dose

`safe_offense_reward01_v1` is a valid single-dose follow-up against the existing
`safe_offense_zero_v1` control. Code inspection confirms that both select the
exact same frozen source
`ben_task4_rule_based_continue_control1000_v1_1000ep_seed11.pt` (11 channels),
the same feature modes, ordinary rewards, optimizer/replay/DQN hyperparameters,
1000-episode naming convention, Seed 11 and declared three-rule-based lineup.
The only intended training difference is
`SAFE_OFFENSE_BOMB_REWARD = 0.1` instead of `0.0`. The existing trigger counter
and `safe_offense_bombs` train-log column remain active, so dose usage is still
auditable.

All 13 scaffold tests pass under the actual reward01 environment and separately
under the zero environment. The explicit cross-process test verifies identical
source, input channels, kill/death rewards and opponent metadata, followed by
the sole `0.0` versus `0.1` difference. The candidate final model, training CSV,
training meta, log directory and all ten checkpoint targets were checked and
are free. No reward01 evaluation artifact currently exists either. The run
command must still enforce the real lineup explicitly with `ben_task4` followed
by three `rule_based_agent`s; the metadata string alone cannot do so.

Reusing the completed zero training and evaluation is scientifically valid and
preferable here: the zero arm is unchanged, came from the identical source and
configuration, was selected before observing reward01, and its default-seed
full1000 evaluation is the fixed control to which the new arm should be paired.
There is no requirement to retrain a stochastic control merely to repeat the
same comparison; doing so would add training/opponent noise and change the
estimand. This reuse still pairs arenas only, because the rule-based opponents'
stdlib RNG is not fully controlled.

The earlier reward02 result motivated exactly this one intermediate `+0.1`
dose. It must be described as a new post-result dose-response hypothesis, not a
preregistered rescue of reward02. Apply the already recorded score, kill,
safety and timing gates unchanged. This is the final dose test: do not try
`0.05`, `0.15`, extra checkpoints or another value after seeing reward01. If it
fails any gate, stop dose tuning; only if it passes all gates should reward01 be
replicated on training Seeds 12/13.

## Post-run audit: `+0.1` dose

The reward01 arm completed 1000 sequential training rows and all ten scheduled
checkpoints; its final state tensors equal the episode-1000 checkpoint. Training
metadata records the intended source, 11 channels, `safe_offense_reward=0.1`,
Seed 11 and three rule-based opponents. The final model SHA-256 is
`42dbee1e0d33d8ae2efdc0f7c6133bbbc0b58b446897ed385e840c89bd816ac9`,
and the full-eval metadata attributes that exact model to `ben_task4`. The full
evaluation has 4000 rows/1000 complete rounds with the DQN in slot 0 and the
expected three opponents.

Trigger logging is intact: reward01 recorded 1702 eligible bombs out of 12,353
training bombs (13.8%, 1.702/episode), versus zero's 0 triggers. It recorded 66
training kills; these exploratory training totals are diagnostics, not outcome
evidence. At evaluation reward01 uses 23.073 bombs/round versus 21.824, a paired
increase of `+1.249` (CI `[+0.405,+2.095]`, sign-flip `p=0.0044`).

Candidate minus the fixed zero control on full1000 is:

- score `+0.038` (`[-0.151,+0.231]`, `p=0.7014`): no effect;
- kills `-0.006` (`[-0.037,+0.026]`, `p=0.7544`): no effect;
- suicides `-0.077` (`[-0.119,-0.034]`, `p=0.0004`): clear improvement;
- killed by opponent `-0.026` (`[-0.048,-0.004]`, `p=0.0258`): clear improvement;
- survival is consequently `+0.103`, with direct paired CI `[+0.060,+0.147]`
  and sign-flip `p<0.0001`: clear improvement.

The death arithmetic is coherent: the two death reductions sum to the survival
gain. But the preregistered mechanism required positive, non-fragile score and
kills (and score magnitude above `0.12`); both attack gates fail. Reward01 is
therefore **rejected as a safe-offense arm**. It may be retained as a single-seed
safety finding/hypothesis, but it is not a Task-4 attack improvement and cannot
replace the control on these data.

Timing is safe despite `analyze.py` calling the per-round maximum statistically
worse: DQN mean think is `0.232 ms`, its true global maximum is `30.288 ms`, and
there are zero 500-ms overruns (zero: `0.191 ms`, global `27.423 ms`). The
`+2.136 ms` mean-of-round-max difference is operationally immaterial and is not
the actual global maximum.

The predeclared dose sequence is now exhausted. Do not run reward01 Seeds 12/13,
because the causal attack gates already fail, and do not try `0.05`, `0.15`, a
checkpoint or another dose. The scientifically defensible conclusion is a
dose-dependent trade-off: `+0.2` improved attack but missed safety guards;
`+0.1` improved safety but showed no attack effect. Further work needs a new
mechanism, not post-hoc dose search.
