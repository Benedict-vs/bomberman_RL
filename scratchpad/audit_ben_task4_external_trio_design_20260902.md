# Adversarial pre-run audit: external-trio training (2026-09-02)

## Verdict

`external_trio_control_v1` versus `external_trio_v1` is a valid controlled
Seed-11 pilot. Both arms start from the exact same frozen 11-channel model and
retain the same features, rewards, DQN/replay/optimizer hyperparameters and
1000-episode schedule. The intended intervention is the training opponent field:
three provided rule-based agents in the control versus Li-Jesse, bindist and
binary in the candidate.

No submission-isolation or ML-rule blocker was found. The main threat is
statistical rather than implementation-related: the training seed controls the
arena but not all opponent stdlib RNG, and the two lineups cannot share opponent
trajectories. One run per arm is therefore a pilot, not a causal family-level
result. External-agent CPU cost also makes sequential, power-safe execution
important.

## Source and one-variable contrast

- Both arms load
  `ben_task4_rule_based_continue_control1000_v1_1000ep_seed11.pt`, SHA-256
  `c9677540849340e915f74dec7cf9c2d24e9c6c27f9d5132453798035a1f8a9ec`.
- Both use 11 channels, reachable-safe-tiles, disabled opponent alignment and
  escape pressure, the same visit-count input, legal action mask and symmetry
  augmentation.
- Ordinary rewards are identical (coin `+1.5`, crate `+0.3`, kill `+5`, invalid
  `-1`, `GOT_KILLED=-5`, `KILLED_SELF=0`, step `-0.05`) with the same inherited
  safety shaping and no safe-offense auxiliary reward.
- Both use Seed 11, 1000 episodes and arm-specific Task-4 model/log/checkpoint
  names. The control is a new matched 1000-episode continuation, not accidental
  reuse of the source model.
- The only audited arm-dependent value is declared opponent lineup:
  `rule_based_agent,rule_based_agent,rule_based_agent` versus
  `ext_lijesse_featureeverything,ext_xiaoxiae_bindist_v2,ext_xiaoxiae_binary_v6`.

`TRAINING_OPPONENTS` is descriptive metadata. It does not configure `main.py`.
The actual control CLI must list Ben followed by three rule-based agents; the
candidate CLI must list Ben followed by Li-Jesse, bindist and binary in exactly
that order. A mismatch between CLI and metadata would invalidate the experiment,
so preserve the commands with the run record and verify the first log/state
before leaving the long run unattended.

The same numerical training seed does not make the two training histories
paired: opponent type is the intervention, and provided/external agents retain
partly unseeded behavior. That is expected, but it limits a one-seed conclusion.

## Submission and external provenance

`agent_code/ben_task4` contains no import of any external agent. Their names
appear only in the training metadata selection and tests. The trained policy
continues to depend only on its own 11-channel input and learned weights; the
external agents are environmental opponents, not copied features, rules or
teachers. The final submitted `ben_task4` directory therefore does not require
the external directories.

The external installations were separately pinned and loader/hash-verified:
Li-Jesse is the high-score coin/survival opponent, while bindist and binary are
the more aggressive bomb/kill opponents. They may be used as training/evaluation
opponents, but their code and learned logic must remain outside the submission
and—especially for the no-license copies—outside any public redistributed
artifact. Their callback hashes/model hashes should be recorded with the
experiment because standard eval metadata does not currently capture every
external model file.

This remains a machine-learning agent: changing the environment's opponents
changes experience distribution, while the DQN must learn its own policy. No
external best-action oracle, imitation label or code path is added.

## Tests and overwrite safety

Both actual environments independently pass all 16 scaffold tests. The focused
cross-process test verifies equal source, 11-channel configuration, escape and
opponent-feature modes, coin/kill/death rewards, followed by only the intended
lineup difference.

For both arms, the final model, training CSV, training meta, log directory and
all ten checkpoint targets (`100..1000`) were checked and are absent. The run
does not overwrite an existing Task-4 artifact under the planned names.

## Runtime risks

The prior external-trio Quick100 required about 470 seconds, dominated by
Li-Jesse's feature computation. Training additionally performs replay updates
and feature/augmentation work, so candidate wall time will be much longer than
the three-RB control and should not be inferred from episode count alone.

Run the arms sequentially, with the laptop powered and sleep prevention active;
do not run simultaneous `ben_task4` training/evaluation jobs. Parallel runs
would contend for CPU/GPU and the framework's same-agent log path can overwrite.
Use `BM_QUIET_LOGS=1`, distinct precreated log directories and the training CSV
as the durable progress record. An interrupted arm is not comparable to a
completed 1000-episode arm unless resumed under a separately audited mechanism.

## Preregistered evaluation gates

Quick100 is a liveness/safety smoke test. Apply decisions to candidate minus
matched control on full1000 default-seed evaluations. Opponent RNG is only
arena-paired; one training seed remains a pilot even when the evaluation CI is
clear.

### Primary external-trio evaluation

1. Score must be positive, exceed `+0.12` in point estimate, have paired 95% CI
   excluding zero, sign-flip `p<.05`, and be non-fragile.
2. Kill conversion is the registered mechanism: kills must be positive with CI
   excluding zero, sign-flip agreement and non-fragile verdict. A coin-only
   score gain does not validate the intended combat adaptation; report coins
   separately to identify substitution.
3. Safety guards remain suicides `<= +0.03`, killed-by-opponent `<= +0.02`, and
   survival `>= -0.03`. A demonstrated safety regression overrides score/kills.
   Inspect bombs and bombs-per-kill for uncontrolled aggression.
4. Zero timeouts and true global `think_max_ms < 500`, with substantial slower-
   hardware margin.

### Three-rule-based regression evaluation

The external adaptation must not destroy the established broad baseline:

- score must be no worse than `-0.12` in point estimate and must show no clear,
  non-fragile negative effect;
- kills must show no demonstrated regression;
- the same suicide, killed-by, survival and timing guards apply.

Passing the external trio while failing the 3RB guard is a distribution-specific
trade-off, not an accepted improvement. Conversely, a 3RB-safe model without a
clear external score and kill gain does not demonstrate the intended mechanism.
Only a full pass on both fields should advance to training Seeds 12/13. Do not
select checkpoints, external-agent subsets or slot orders after seeing partial
results.

## Final post-run audit: external-trio full1000

### Measurement integrity

Both full files contain exactly 4000 rows, 1000 rounds `0..999`, default seeds
`20260731..20261730`, and 1000 observations in each fixed slot. Agent order is
identical in both: Ben slot 0, Li-Jesse slot 1, bindist slot 2, binary slot 3.
Metadata selects the correct arm and attributes the already-audited final hashes
`51536d...b031` (control) and `c2aaaf...bb4e` (candidate), with identical Ben
callback hash `94d87f...f87d`, trained variant, Seed 11 and 1000 episodes.
External callback hashes also match across files and the pinned installation.

Every Ben row satisfies `score = coins + 5*kills` and
`survived + suicides + killed_by = 1`. No row, round, slot or seed is missing.
The fixed slot is shared across arms, so it is controlled for this narrow model
comparison, but neither result establishes position-robust generality.

### Candidate minus matched control

- score: `2.060 - 2.025 = +0.035`, CI `[-0.098,+0.168]`, sign-flip
  `p=0.6268`, non-fragile;
- kills: `.062 - .043 = +.019`, `[-.001,+.039]`, `p=.0882`;
- coins: `1.750 - 1.810 = -.060`, `[-.140,+.020]`, `p=.1415`;
- suicides: `.480 - .519 = -.039`, `[-.083,+.005]`, `p=.0960`;
- killed by opponent: `.177 - .133 = +.044`, `[+.011,+.076]`,
  `p=.0076`, clear and non-fragile;
- survival: `.343 - .348 = -.005`, `[-.046,+.037]`, `p=.8536`;
- bombs: `20.692 - 17.254 = +3.438`, `[+2.395,+4.491]`,
  sign-flip `p<.0001`.

The most favorable attempted refutation is that kills rise by 44% relative to a
very low control rate and narrowly miss significance. That does not rescue the
arm: the absolute gain is small, its CI includes zero/sign-flip disagrees with a
mechanism claim, score gains only `.035` (well below the `.12` practical/noise
floor), and the policy substitutes substantially more bombs plus more enemy-
bomb deaths for slightly fewer suicides. This is not successful kill conversion
or a favorable safety trade-off.

### Timing and RNG

Ben mean think time is `0.263 ms` candidate versus `0.277 ms` control; true Ben
global maxima are 32.55 and 37.48 ms. Across all four agents global maxima are
325.28 and 320.16 ms, with zero 500-ms overruns. The external field remains
computationally heavy but timing passes and does not explain the result.

Evaluation pairing controls arenas, not the external agents' stdlib randomness.
The known approximately `.12` score noise floor makes the observed `+.035`
especially unreadable, and one training seed cannot establish a population
effect. This caveat weakens any positive claim; it does not create evidence that
the candidate met its gates. Fixed slots similarly limit generality but are
identical between arms.

### Preregistered verdict and remaining measurements

The candidate **fails** the external primary gate (score is neither clear nor
`>.12`), the kill-mechanism gate, and the killed-by safety guard (`+.044` versus
allowed `+.02`, with a clear adverse interval). Suicide and survival guards
pass, as does timing, but partial gate success cannot override three required
failures. Reject `external_trio_v1` as a Task-4 improvement and do not run
training Seeds 12/13 or inspect checkpoints.

The preregistered funnel explicitly made the 3RB full1000 guard conditional on
passing the external field. A 3RB result cannot rescue failure on the target
field or prove the intended mechanism; therefore **skip the 3RB guard** for the
decision. It would be scientifically permissible only as optional descriptive
characterization for a report question such as catastrophic forgetting, with a
newly stated purpose and without reclassifying the arm. Given the clear target-
field failure, it is not necessary measurement and the saved budget should go
to a genuinely different, pre-audited mechanism rather than more opponent-field
continuation.

## Quick100 stop/go audit

### Artifacts, attribution and lineup

Both training arms completed 1000 sequential rows and all ten scheduled
checkpoints; each final model's state tensors equal its episode-1000 checkpoint.
Final SHA-256s are
`51536db93061e5b49dd5b2a65973a5f0fa65bb342a8f0b162a66bb58ab88b031`
(control) and
`c2aaaf4b68e6124fc26f23bcf48cbe3b84b70288da4ccb9650ddab701207bb4e`
(external trio). Training metadata records the shared `c967...a9ec` source,
correct arms and declared lineups. Candidate training took about 2688 seconds
versus 1604 seconds for control, consistent with the slower external field.

Quiet training logs are empty and `TRAINING_OPPONENTS` remains metadata rather
than CLI enforcement, so the stored artifacts alone cannot independently prove
which opponent command was typed. The recorded arm and wall-time difference
support, but do not replace, preserving the actual CLI in the experiment ledger.

Both evaluation files are complete: 400 rows/100 rounds, default seeds, and the
same fixed evaluation order (`ben_task4`, Li-Jesse, bindist, binary). Eval
metadata selects the correct arm and exact matching final-model hash. Thus the
evaluated policies are correctly attributed, and their evaluation opponent
lineup is proven directly by CSV slots rather than metadata alone.

### Candidate minus matched control

The direct Ben-policy comparison reports:

- score `+0.030`, CI `[-0.450,+0.510]`, sign-flip `p=0.9375`;
- kills `-0.020`, `[-0.100,+0.060]`, `p=0.8036`;
- suicides `+0.000`, `[-0.150,+0.140]`;
- killed by opponent `+0.010`, `[-0.090,+0.110]`;
- survival `-0.010` by exact death arithmetic;
- bombs increase from 15.66 to 19.62 per round (`+3.96` point effect).

No primary effect is demonstrated at n=100. The intended kill mechanism points
slightly the wrong way, and the small score gain comes from coins (`2.06` versus
`1.93`), not kills. This is not positive evidence for adaptation. However, the
registered safety point guards all pass: `0.000 <= +0.03`, `+0.010 <= +0.02`,
and `-0.010 >= -0.03`. Unlike the escape-pressure quick failure, there is no
large coherent candidate-minus-control harm that justifies a safety stop.

Absolute performance remains poor for both policies: candidate suicides in
0.56 of rounds, survives 0.29, and still trails each external opponent in score
and kills. Candidate-minus-opponent raw kill gaps are `-0.11` (Li-Jesse),
`-0.14` (bindist), and `-0.17` (binary). Its score gaps narrow slightly versus
the control primarily through coins. These are important diagnostics, but the
acceptance design compares candidate with the matched continuation; a shared bad
absolute baseline is not evidence that the opponent-distribution intervention
worsened safety.

### Timing and stochastic caveat

There are zero timeout breaches. Candidate/control DQN mean think times are
about `0.279/0.310 ms`; mean-of-round maxima `4.00/4.20 ms`, and true DQN global
maxima `20.90/18.18 ms`. Across all agents the global maxima are 144.08 and
82.05 ms, still below 500 ms and attributable to the external field rather than
Ben inference. Run longer evaluations alone because the four agent processes
may contend for CPU.

The two evaluation files share arenas, but external agents use incompletely
seeded stdlib randomness. Consequently these are not behavior-paired opponent
trajectories, and the wide n=100 intervals are the expected limitation rather
than evidence that the true effect is zero.

### Stop/go decision

I attempted to support an early stop but cannot do so under the preregistered
logic. Quick100 was defined as a smoke test, the primary score/kill intervals
still include practically useful positive effects, and no safety/timing guard is
violated relative to control. **Proceed with the paired external-trio full1000
evaluations.** This is a measurement go, not a claim that the candidate is
promising; current mechanism evidence is absent.

Use a sequential funnel to conserve time. First evaluate both models full1000 on
the exact external trio and apply the registered score, kill and safety gates.
If score or kills fail—or safety regresses—stop the arm and do not run Seeds
12/13. Run the separate 3RB full1000 guard only if the external full1000 passes
all external gates; otherwise the guard cannot rescue a failed primary
mechanism and is not worth the extra run. Do not reinterpret the present coin
point gain as success or inspect checkpoints.
