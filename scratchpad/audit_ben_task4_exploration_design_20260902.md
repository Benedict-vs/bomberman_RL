# Adversarial pre-run audit: epsilon exploration (2026-09-02)

## Verdict

`exploration_eps05_v1` versus `exploration_eps15_v1` is a clean
one-hyperparameter 1000-episode Seed-11 pilot. Both arms share the exact frozen
11-channel source, rewards, features, optimizer/replay/target settings and three
rule-based training opponents. The sole intended difference is epsilon start
`0.05` versus `0.15`; both end at `0.05` with a 100,000-environment-transition
linear decay.

No implementation or overwrite blocker was found. The scientific limitation is
expected: exploration deliberately changes the visited-state and replay
distribution, and Python-random action/opponent behavior is not paired by the
arena seed. One training seed is therefore a pilot only.

## Isolation and source

- Both arms load
  `ben_task4_rule_based_continue_control1000_v1_1000ep_seed11.pt`, SHA-256
  `c9677540849340e915f74dec7cf9c2d24e9c6c27f9d5132453798035a1f8a9ec`.
- Both use the same 11 input channels, reachable escape feature, visit count,
  disabled optional opponent channel, legal-action mask and symmetry
  augmentation.
- Learning rate `1e-4`, gamma, replay capacity `200000`, batch 64, minimum
  replay 5000, soft target tau `1e-4`, optimizer and one-update-per-transition
  logic are shared.
- Rewards and shaping are identical, including coin `+1.5`, crate `+0.3`, kill
  `+5`, `GOT_KILLED=-5`, `KILLED_SELF=0`, step `-0.05`, and no experimental
  auxiliary reward.
- Both declare and must actually run against exactly three
  `rule_based_agent`s, with Seed 11 and 1000 episodes. The lineup metadata does
  not replace the CLI, so preserve/verify the actual command.
- The only arm branch affecting this pair is `EPSILON_START`: `0.05` for control
  and `0.15` for candidate. `EPSILON_END=0.05` and
  `EPSILON_DECAY_STEPS=100000` are identical and logged in metadata.

Exploratory actions are sampled only from the existing legal-action mask, so
higher epsilon does not directly introduce invalid moves. It does increase
random legal movement/bombing and necessarily changes later states, episode
length, replay contents and optimization samples. Those are consequences of
the intended intervention, not hidden hyperparameter differences.

## Actual schedule and expected endpoint

The schedule advances once after every stored environment transition:

`epsilon(step) = start + min(step/100000, 1) * (0.05 - start)`.

Therefore control remains exactly `0.05`. Candidate starts at `0.15`, reaches
`0.10` at transition 50,000 and reaches/floors at `0.05` from transition
100,000 onward. It is inaccurate to describe candidate as “15% exploration for
1000 episodes”; it is a high-to-low early-training schedule.

Recent comparable 1000-episode three-RB runs produced roughly 120,000--133,000
transitions, so candidate is expected to hit `0.05` around episode 750--850 and
finish at `0.05`. This is a forecast, not a guarantee: extra exploration may
alter deaths and episode lengths. The train CSV must be checked after the run.
If total transitions are `N<100000`, the endpoint will instead be
`0.15 - 0.10*N/100000`; that is correct implementation behavior, not an
incomplete decay bug.

Over the first 100,000 transitions candidate averages epsilon `0.10` versus
control `0.05`, approximately 5,000 extra exploratory decisions in expectation.
Afterward they share `0.05`. The causal question is thus whether extra early
legal exploration produces a better final greedy policy.

## Randomness and interpretation

`act()` uses Python's stdlib `random` for the epsilon coin flip and random legal
action. The project seed does not fully pair stdlib randomness across opponent
processes/runs. Even the control has `0.05` exploration, so the two training
histories are not action-by-action paired. Increased trajectory diversity is
the intervention, but single-run differences also contain stochastic training
and opponent variation.

Final evaluation at `self.train=False` uses no epsilon exploration, so the
measurement compares learned greedy policies rather than one policy acting more
randomly. Evaluation still pairs arenas only because rule-based opponents also
use incompletely seeded stdlib randomness. Clear single-seed results must be
replicated across training seeds before a Task-4 family claim.

## Tests and overwrite safety

Both actual arm environments independently pass all 19 scaffold tests. The
focused cross-process test verifies equal source, input channels, learning rate,
epsilon end/decay, kill/death rewards and three-RB lineup, with only `0.05`
versus `0.15` start differing. Existing schedule code is linear, clamped and
transition-based as metadata states.

For both arms, the final model, training CSV, training meta, log directory and
all ten checkpoints (`100..1000`) were checked and are absent. The planned run
will not overwrite existing artifacts.

## Preregistered gates and funnel

Compare eps15 minus eps05 on deterministic inference evaluations using default
arena seeds.

1. Score: positive, point estimate `>+0.12`, paired 95% CI excluding zero,
   sign-flip `p<.05`, non-fragile.
2. Kill mechanism: kills positive with CI excluding zero, sign-flip agreement
   and non-fragile. Also report coins; score improvement without kills is useful
   performance but does not prove the intended Task-4 combat deficit improved.
3. Safety: suicides `<=+0.03`, killed-by-opponent `<=+0.02`, survival
   `>=-0.03`. Inspect bombs and bombs-per-kill because added exploration may
   teach either useful aggression or bomb spam.
4. Timing: zero timeout breaches and true global `think_max_ms<500`, with ample
   slower-hardware margin. Epsilon is training-only, so inference should remain
   equal-cost.

Quick100 is a smoke/futility screen. Stop early for a clear safety regression or
when the favorable score interval does not reach the practical `+0.12` gate and
the kill mechanism points negative with no compensating benefit. Otherwise run
full1000 and apply all formal gates. Only a full pass should advance eps15 to
training Seeds 12/13. Do not tune an intermediate epsilon, decay length or
checkpoint after viewing results; that would begin a new, separately justified
experiment rather than rescue this one.

## Quick100 stop/go audit

### Artifacts and actual schedule

Both arms completed exactly 1000 training rows and all ten checkpoints; final
model tensors equal episode-1000. Final SHA-256s are
`589dd419082bdd6907eebe726c6ead094bb1a3dca02b94aae146dbeabbacc0a3`
(eps05) and
`ad5d15cdc06a84c78351cad4ad7df78e4a2b0639c1a30541b1e30e94ef313892`
(eps15). Metadata records the same source, rewards/hyperparameters, three-RB
lineup, Seed 11 and intended `.05/.05` versus `.15/.05` schedule.

The endpoint forecast was wrong, but the implementation was not. Eps05 produced
119,997 transitions and remained `.05`; eps15 produced only 56,349 transitions,
so the formula gives `.15 - .10*(56349/100000) = .093651`, exactly the logged
endpoint. Higher exploration substantially shortened episodes. Because one
optimization occurs per transition after replay warm-up, eps15 also received
roughly half as many replay updates as eps05. This is a downstream consequence
of the intended epsilon intervention under a fixed 1000-episode budget, not a
hidden code difference, but it prevents a narrow interpretation as “same amount
of learning with more diverse actions.” The measured treatment is higher early
exploration **plus its induced reduction in transitions/updates**.

Do not now continue eps15 until 100,000 transitions or force epsilon to `.05`:
that would be a new transition-budget/annealing experiment chosen after seeing
the outcome, not completion of this preregistered episode-budget arm.

Each quick eval is complete (400 rows/100 rounds, default seeds, Ben slot 0 and
three rule-based opponents). Eval environment and provenance select the correct
arm and exact final hash; callback hashes match. No attribution or completeness
problem was found.

### Candidate minus control

- score `-0.210`, CI `[-0.880,+0.430]`, sign-flip `p=.5507`;
- kills `0.000`, `[-0.110,+0.110]`, `p=1.0`;
- coins `-0.210`, `[-0.550,+0.130]`, `p=.2484`;
- suicides `-0.060`, `[-0.190,+0.060]`;
- killed by opponent `+0.010`, `[-0.060,+0.090]`;
- survival `+0.050`, `[-0.070,+0.160]`;
- bombs `-0.730`, `[-3.080,+1.600]`.

Nothing is demonstrated at n=100. The point estimates do not suggest the
registered score/kill mechanism, but the favorable score interval extends to
`+0.430`, comfortably beyond the practical `+0.12` gate. Safety point guards
all pass (`-.060`, `+.010`, `+.050` respectively), and there is no clear bomb-
spam or harm signal. Therefore the preregistered futility condition—favorable
score bound below `+.12` together with negative kills and no compensation—is
not met.

### Timing and RNG

Inference timing is effectively unchanged: mean think about `.195/.189 ms`,
true global maxima 17.76/16.77 ms, and zero timeout breaches. Epsilon affects
training only.

Rule-based stdlib RNG remains incompletely seeded, so Quick100 shares arenas but
not opponent behavior. Wide intervals and the known score-noise floor forbid a
go/stop conclusion based on the `-.21` point estimate alone. Training randomness
is even less paired because epsilon itself changes stdlib-random draws and
trajectories.

### Decision

I attempted both an early go and an early stop; neither is supported. The arm is
not presently promising—score points down and kills are flat—but it has no clear
safety regression, and Quick100 still permits a practically useful score effect.
Under the registered funnel, **run both models on full1000** and apply the formal
score/kill/safety gates. This is a measurement continuation, not endorsement.

If full1000 fails, stop this epsilon arm without extending training, changing
decay or matching transition counts post hoc. If it passes every gate, the
unreached endpoint and unequal update counts must still be reported as part of
the treatment before considering Seeds 12/13; a future transition-matched study
would answer a different question.

## Final full1000 audit and long-training question

### Measurement integrity

Both full evaluations contain exactly 4000 rows/1000 rounds, default seeds
`20260731..20261730`, and 1000 observations in each slot. Ben remains slot 0
against three rule-based opponents in both. Metadata selects the correct arm,
1000-episode schedule and exact previously audited final-model hashes
`589dd4...0a3` (eps05) and `ad5d15...3892` (eps15); callback hashes and rules
match. Score arithmetic and the death partition are exact in every Ben row.

### Candidate minus control

- score `3.367 - 3.465 = -0.098`, CI `[-0.277,+0.083]`, sign-flip
  `p=.2979`;
- kills `.111 - .128 = -.017`, `[-.046,+.012]`, `p=.2838`;
- coins `2.812 - 2.825 = -.013`, `[-.112,+.083]`, `p=.8081`;
- suicides `.359 - .470 = -.111`, `[-.153,-.069]`, sign-flip
  `p<.0001`, clear and non-fragile;
- killed by opponent `.071 - .096 = -.025`, `[-.049,-.001]`, nominal
  `p=.048`, but **fragile**, so no demonstrated effect under project rules;
- survival `.570 - .434 = +.136`, `[+.092,+.179]`, sign-flip
  `p<.0001`, clear and non-fragile;
- bombs `20.626 - 21.313 = -.687`, `[-1.503,+.116]`, no effect.

The survival arithmetic is coherent: the robust suicide reduction plus the
point killed-by reduction equals the survival gain. This is strong evidence for
a safety effect in this trained Seed-11 pair. It is not evidence of Task-4
offense: score is negative and its upper CI bound `+.083` is below the required
`+.12`; kills also point negative. Thus eps15 fails both primary replacement
gates despite passing safety.

### Timing and randomness

Mean inference time differs trivially (`.202 ms` versus `.192 ms`).
`analyze.py` reports a statistically higher mean-of-round maximum, but the true
global maxima are only 23.49 ms (eps15) and 19.53 ms (eps05), with zero timeout
breaches. Timing passes comfortably.

Evaluation remains arena-paired rather than fully opponent-behavior-paired due
to stdlib RNG. The `-.098` score difference is smaller than the known `.12`
noise floor and cannot support a harm claim. Conversely, noise cannot turn it
into the registered positive `>+.12` improvement. One training seed also limits
generality; the clear suicide/survival result is a strong single-seed finding,
not yet a multi-seed family claim.

### Formal 1000-episode verdict

**Reject eps15_v1 as the 1000-episode replacement and stop that registered
experiment.** Do not continue its saved model, run its Seeds 12/13, choose a
checkpoint or reinterpret safety as fulfillment of the score/kill gates. This
is exactly the failure condition recorded before full1000.

### Is a fresh 3000-episode comparison justified?

There is nevertheless a scientifically defensible *new* hypothesis. Eps15
received only 56,349 transitions and ended at `.093651`, whereas eps05 received
119,997 transitions. The higher-exploration policy shows a large safety gain
despite roughly half the replay/optimization exposure, while its score cost is
not demonstrated. It is plausible—not established—that reaching transition
100,000 and then training at `.05` for many further episodes could retain safety
while recovering score/kills.

A fresh 3000-episode comparison from the same common `c967...a9ec` source can
test that practical fixed-episode-budget hypothesis. It must be registered as
new arms before running, train **both** `.05` and `.15 -> .05` schedules from
the common source, and never initialize from either 1000-episode result. This
does not “complete” or rescue eps15_v1; it changes the training horizon and
estimates a different treatment. The earlier stop rule prohibits simple
continuation/post-hoc relabeling, not all future hypotheses motivated by a
documented mechanism.

The longer design still will not equalize transitions/updates: fixed episodes
are the budget and episode length is an outcome. At the observed early rate,
eps15 should cross 100,000 transitions around episode 1775, leaving substantial
post-anneal training, but the actual endpoint and transition totals must be
logged. If the desired question is instead “same number of optimizer updates,”
an explicitly transition-matched design is required and is a different
experiment.

Given the robust suicide/survival signal, one predeclared 3000-episode Seed-11
pilot is justified if compute budget permits. Preserve the same primary gates:
score positive/clear/non-fragile and `>+.12`, kills positive/clear, and the
existing safety/timing guards. Safety alone again cannot select the replacement.
Use quick100 only for catastrophic safety/futility, then full1000 if it passes;
no further horizon, epsilon or decay search after this single test. Only a full
pass would justify additional training seeds.

## Final pre-run audit: fresh 3000-episode comparison

The implemented `exploration_eps05_long3000_v1` and
`exploration_eps15_long3000_v1` correctly instantiate the new hypothesis rather
than continuing the rejected 1000-episode models.

- Both load the common frozen source
  `ben_task4_rule_based_continue_control1000_v1_1000ep_seed11.pt`, SHA-256
  `c9677540849340e915f74dec7cf9c2d24e9c6c27f9d5132453798035a1f8a9ec`.
  Neither source name contains an exploration arm, and neither loads the eps05
  or eps15 1000-episode output.
- Both use 11 channels, the identical feature set, rewards/shaping, learning
  rate, optimizer, replay capacity/warm-up/batch, target update, legal mask,
  augmentation, three-rule-based opponent metadata, Seed 11 and 3000 episodes.
- Both use epsilon end `.05` and transition decay 100,000. The only intended
  difference is start `.05` versus `.15`.
- Model names and run labels include the distinct long3000 arm, `3000ep` and
  Seed 11. For each arm the final model, train CSV, train meta, log directory
  and all 30 checkpoint targets (`100..3000`) were checked. All 34 targets per
  arm are absent; no prior artifact will be overwritten.

Both actual 3000-episode configurations independently pass all 20 scaffold
tests. The focused subprocess test verifies the identical common source (and
explicitly rejects an exploration source), distinct 3000-episode output model,
same input channels/learning rate/end/decay/opponents and sole `.05` versus
`.15` start difference. Shared code inspection confirms unchanged rewards,
replay and target-network settings beyond the values printed by that focused
test.

This remains a fixed-episode, not fixed-transition, comparison. Episode length
is an outcome of exploration, so the arms may again receive unequal transition,
replay-update and wall-clock budgets. At the prior eps15 rate it would reach
transition 100,000 around episode 1775, but that is only a forecast; changing
policy quality changes episode length. Report total transitions, the episode at
which `.05` is first reached, final epsilon and optimization exposure for both
arms. Unequal transitions do not invalidate the practical question “best policy
after 3000 episodes”, but they prohibit claiming equal-data sample efficiency.

The 1000-episode eps15 result remains rejected. These fresh models must not be
described as continuation, completion or rescue of it, and no 1000-episode
checkpoint should enter either run. Similarly, 3000 was selected before this
new run; do not select an intermediate checkpoint, extend the horizon again or
change decay after viewing the curves.

Apply the existing gates without modification: eps15-long minus eps05-long
score positive, clear/non-fragile and `>+.12`; kills positive and clear as the
offense mechanism; suicides `<=+.03`, killed-by `<=+.02`, survival `>=-.03`;
zero timeouts and global think below 500 ms. Quick100 is only the already
defined catastrophic safety/futility screen. Only a full1000 evaluation passing
**all** score, kill, safety and timing gates may advance this long schedule to
training Seeds 12/13. A repeated safety-only result remains scientifically
useful but is not a replacement pass.

**Pre-run verdict: GO for the single fresh Seed-11 long pilot.** The source,
causal contrast, tests and overwrite boundaries are correct; the remaining
unequal-transition issue is the registered estimand, not an implementation
surprise.

## Long3000 Quick100 stop/go audit

### Training and evaluation integrity

Both fresh long arms completed exactly 3000 rows and all 30 checkpoints; final
model tensors equal episode-3000. Final SHA-256s are
`1a5356cf3882dba91689c82ccea193c434197822eda2b4bf3853482b2eae8337`
(eps05-long) and
`ef9599e6f902e4dc88ccba618ee6f84f322a6630700af74a2c21eb650950c324`
(eps15-long). Metadata records the same common `c967...a9ec` source, 3000
episodes, Seed 11 and `.05/.05` versus `.15/.05` schedules. No 1000-episode
exploration model entered either run.

The expected fixed-episode inequality occurred but is much smaller than at
1000 episodes: eps05 accumulated 377,050 transitions; eps15 accumulated
306,191. Candidate first reached logged epsilon `.05` at episode 1461 and then
trained for another 1539 episodes at `.05`. It therefore received substantial
post-anneal experience, while still having about 19% fewer transitions/updates
than control. This is the registered fixed-episode treatment, not an incomplete
run or equal-data comparison.

Each Quick100 file has 400 rows/100 complete default-seed rounds, Ben in slot 0
against three rule-based agents. Eval metadata selects the correct 3000-episode
arm and exact final hash; callback hashes, settings and lineup match.

### Candidate minus control

- score `+0.300`, CI `[-0.210,+0.820]`, sign-flip `p=.2742`;
- kills `+0.040`, `[-0.040,+0.120]`, `p=.4533`;
- coins `+0.100`, `[-0.210,+0.410]`, `p=.5707`;
- suicides `+0.030`, `[-0.080,+0.140]`, `p=.7268`;
- killed by opponent `-0.020`, `[-0.100,+0.060]`, `p=.8097`;
- survival `-0.010`, `[-0.140,+0.120]`, `p=1.0`;
- bombs `+2.750`, `[+0.720,+4.820]`, `p=.0098`, clear and non-fragile.

At n=100 the primary effects are appropriately inconclusive but both score and
kills point in the registered direction, and the score interval permits a
practically substantial improvement. The only demonstrated behavioral change
is more bombing. Because kills also point higher, this is compatible with the
intended increased combat behavior, but it could still become unproductive bomb
spam at full sample size.

The suicide guard was explicitly written `<= +0.03`; the observed `+0.030`
therefore passes **exactly at the boundary**, not fails. It has zero safety
margin. Killed-by and survival also pass their point guards, and no safety CI
excludes zero. It would be improper to tighten the suicide inequality after
seeing the data, but equally improper to call a boundary value reassuring.

### Timing, randomness and decision

Inference timing passes: mean think about `.201/.198 ms`, true global maxima
19.58/14.48 ms, and zero timeout breaches. The statistically fragile
mean-of-round-max row is operationally irrelevant here.

Rule-based stdlib RNG still makes this arena-paired rather than fully
behavior-paired, and Quick100 is too small for a result. I attempted to refute a
go using the suicide boundary and increased bombs, but neither is a
preregistered stop condition: safety is not demonstrated worse, primary point
effects are favorable, and the score interval reaches far beyond `+.12`.

**Proceed to full1000 for both long models.** This is a valid go under the
registered funnel, with the explicit warning that suicide has no guard margin
and bombs already increased. Full1000 must meet the unchanged score/kills gates
and all three safety guards; a score gain with suicide even slightly above
`+.03`, or without a clear kill gain, is a trade-off/failure. Do not inspect
checkpoints or reinterpret the exact Quick100 boundary. Only a complete
full1000 pass may advance the schedule to Seeds 12/13.

## Final long3000 full1000 audit

### Integrity

Both full evaluations are complete: 4000 rows, 1000 rounds `0..999`, default
seeds `20260731..20261730`, and 1000 rows per fixed slot. Both use Ben in slot 0
against three rule-based opponents. Metadata selects the correct 3000-episode
arm and exact already-audited final hashes `1a5356...8337` (eps05-long) and
`ef9599...c324` (eps15-long); callback hashes, rules and evaluation environment
match. Every Ben row satisfies score and death-partition arithmetic.

The training estimand remains as preregistered: 3000 episodes from the common
source, with 377,050 versus 306,191 transitions and unequal replay updates. The
candidate reached `.05` at episode 1461 and received 1539 post-anneal episodes.
It is therefore not credible to call this an unfinished annealing run, although
it is not an equal-transition comparison.

### Candidate minus control

- score `3.530 - 3.550 = -0.020`, CI `[-0.196,+0.158]`, sign-flip
  `p=.8379`;
- kills `.125 - .110 = +.015`, `[-.012,+.043]`, `p=.3330`;
- coins `2.905 - 3.000 = -.095`, `[-.197,+.006]`, `p=.0702`;
- suicides `.341 - .232 = +.109`, `[+.069,+.148]`, sign-flip
  `p<.0001`, clear and non-fragile;
- killed by opponent `.059 - .079 = -.020`, `[-.043,+.002]`, no effect;
- survival `.600 - .689 = -.089`, `[-.131,-.047]`, sign-flip
  `p=.00005`, clear and non-fragile;
- bombs `20.865 - 18.376 = +2.489`, `[+1.832,+3.176]`, sign-flip
  `p<.0001`, clear.

The small kill point gain cannot rescue the arm: it is not demonstrated, score
is flat/slightly negative and far below the required positive `>+.12`, and the
suicide/survival guards are violated by large clear effects. The candidate
learned to place more bombs, not to convert them into sufficiently more kills
or score while escaping safely.

### Quick100 non-replication

Quick100 suggested score `+.30`, kills `+.04`, suicides `+.03` and survival
`-.01`. Full1000 changes these to `-.02`, `+.015`, `+.109` and `-.089`.
Therefore the apparent positive score and boundary-safe suicide picture did not
replicate. The only stable directional effect is more bombs (`+2.75` quick,
`+2.489` full), which becomes an adverse mechanism once paired with the full
suicide result. Quick100 was correctly used only as a go screen, not a result.

### Timing and stochastic caveat

Inference remains fast: mean think `.189/.199 ms` (candidate/control), true
global maxima 21.07/30.77 ms, and zero timeout breaches. `analyze.py`'s large
mean-of-round-max improvement is operationally immaterial; both are far below
500 ms.

Rule-based stdlib RNG remains only partly controlled, and the score difference
is below the known `.12` opponent-noise floor. Thus no score harm should be
claimed from `-.02`. The safety effects are much larger, have clear paired
intervals/sign-flip agreement and coherent arithmetic. Partial RNG weakens
generalization from one evaluation but does not supply evidence of passing the
registered guards. One training seed similarly cannot establish a population
effect; multi-seed work is conditional on a Seed-11 gate pass, which did not
occur.

### Final decision

**Reject eps15-long3000 and stop.** It fails score, kill, suicide and survival
requirements. Do not run Seeds 12/13, select a checkpoint, extend the horizon,
try another epsilon/decay, or equalize transitions after seeing this result.
Those actions would be additional search after both the 1000- and 3000-episode
versions failed their registered replacement criteria.

The unequal transition totals must remain in the experiment interpretation,
but cannot be used as a rescue argument: unequal exposure was explicitly
accepted as part of the fixed-episode design, the candidate completed ample
post-anneal training, and a fresh longer common-source comparison was already
the one permitted follow-up. The defensible conclusion is that this exploration
schedule changes bomb/safety behavior but did not improve the Task-4 policy
under either tested horizon.
