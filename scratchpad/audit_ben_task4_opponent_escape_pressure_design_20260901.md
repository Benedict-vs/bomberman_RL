# Adversarial pre-run audit: opponent escape pressure (2026-09-01)

## Verdict

`opponent_escape_pressure_zero_v1` versus
`opponent_escape_pressure_v1` is a mostly clean 1000-episode Seed-11 feature
pilot. Both arms load the exact same converted 12-channel source, use the same
ordinary features/rewards/hyperparameters and declare the same three
`rule_based_agent` lineup. The only intended arm difference is channel 11 being
identically zero versus containing opponent escape pressure.

I did not find a hard-ML-rule violation or a timing blocker. I did find one
avoidable geometry nuisance: source tiles exclude walls, crates and existing
bombs, but not tiles currently occupied by an opponent. Such a tile is not a
currently legal bomb placement, yet it receives a pressure value. Excluding all
opponent-occupied source tiles and adding a test would make the feature's
meaning cleaner before training. If left unchanged, record it as a known source
mask limitation and do not later describe every nonzero cell as an immediately
actionable bomb position.

## Isolation and provenance

- Both arms load
  `ben_task4_safe_seed11_12ch_alignment_source.pt`, SHA-256
  `f0e153902c930770739a29fc5bdf4d755405534f3ca3a46f4bf9b59837cceb65`.
  That source was previously verified as the conversion of the same safe
  11-channel model with the first 11 input weights preserved and the new input
  channel zero-initialized.
- Both use 12 channels, reachable-safe-tiles, disabled opponent alignment,
  identical DQN/replay/optimizer settings, identical base rewards and no
  safe-offense helper. `OPPONENT_ESCAPE_PRESSURE_MODE` alone is `zero` in the
  control and `enabled` in the candidate.
- Both metadata paths will record arm, input-channel count, model source and the
  declared lineup. The actual training command must still explicitly contain
  `ben_task4 rule_based_agent rule_based_agent rule_based_agent`; the metadata
  constant cannot enforce the CLI lineup.
- The final model, train CSV, train meta, log directory and all ten checkpoint
  paths (`100..1000`) were checked for both arms and are absent. No existing
  artifact will be overwritten by the planned commands.

Because channel 11 starts with zero source weights, neither arm inherits an old
alignment behavior through that channel. In the zero control it also receives
no input gradient; the candidate alone can learn weights driven by pressure.

## Feature semantics and geometry

For every board cell with `field[x,y] == 0` and no existing bomb, the candidate
examines each opponent. It first requires that opponent to be in the hypothetical
orthogonal bomb blast. Blast power is three, coordinates consistently convert
framework `(x,y)` to CNN `[y,x]`, stone walls stop propagation, and crates do
not stop it, matching this repository's engine semantics.

For an aligned opponent it performs a four-step static BFS, treating walls,
crates, the hypothetical source, existing bombs and the other opponents as
blocked. It counts distinct endpoints outside the hypothetical blast and emits
`1/(1+n_endpoints)`, taking the maximum pressure across opponents. A trapped
opponent therefore yields 1.0 and more modeled exits reduce pressure.

Important limits:

- “Safe” here means only outside this one hypothetical blast. Endpoints may be
  inside an existing bomb's future blast or a current explosion; timers,
  chain reactions and future opponent motion are not modeled. The code's
  docstring correctly calls these *static* endpoints.
- Counting unique endpoints does not measure path multiplicity or opponent
  policy. An opponent with several routes to one endpoint looks more pressured
  than one with several distinct endpoints.
- Existing bomb tiles are excluded as sources, but opponent-occupied source
  tiles are not. The target opponent can even equal the hypothetical source;
  the BFS begins there despite the source being added to the blocked set. This
  produces an internally calculable but presently impossible bomb placement.
- Bomb availability and whether our agent can reach a high-pressure source are
  deliberately not baked into this channel. They remain separate learned inputs
  (bomb-available, self position, board and escape tiles), which prevents this
  channel from becoming a complete action oracle.

The feature is a spatial situation map, not a returned best action: it neither
chooses a movement nor guarantees that bombing is safe for our agent. The DQN
must learn how to combine it with position, reachability, availability and own
escape information. It therefore complies with the hard ML rule, although it
is strong hand-engineering that must be described transparently.

## Runtime consistency and cost

Callbacks inject the selected mode before inference; training injects it for
both old and next states. All transitions therefore keep the same 12-channel
shape, and existing symmetry augmentation operates on the already extracted
spatial tensor. The zero/candidate model and replay dimensions agree.

On a representative local classic-like state, feature extraction measured
about `0.027 ms` for zero and `1.205 ms` for enabled (200-call microbenchmark).
This is not a reference-hardware guarantee, but leaves large margin under
500 ms. Training pays this cost repeatedly, including augmented/next-state
processing, so wall time will rise; it is not a tournament blocker. The final
evaluation must still report true global `think_max_ms` and timeout count.

## Tests

All 15 scaffold tests pass independently under the actual zero environment and
the actual enabled environment. Tests cover 12-channel shape, high pressure for
a trapped aligned opponent, reduction after opening an escape corridor, an
identically zero control channel, and cross-process equality of source,
channels, alignment mode, death/kill rewards and opponent metadata with only the
pressure mode changed.

Missing targeted coverage mirrors the limitations above: no test excludes an
opponent-occupied source, no existing-danger/chain-reaction case, and no explicit
crate-versus-wall blast test for this new channel. At minimum the occupied-source
case should be decided and tested before interpreting cells as legal bomb sites.

## Preregistered pilot gates

Compare enabled minus zero after deterministic full1000 evaluation on default
arena seeds; quick100 is smoke only. Opponent stdlib RNG means pairing remains
arena-only, and one training seed is only a pilot.

1. Primary score: positive, point estimate greater than the known `0.12` noise
   floor, paired 95% CI excluding zero, sign-flip `p<.05`, and non-fragile.
2. Intended offense mechanism: kills positive with CI excluding zero,
   sign-flip agreement and non-fragile verdict. Score without kills does not
   validate “escape pressure” as an attack feature.
3. Safety guards: suicides no more than `+0.03`, killed-by-opponent no more than
   `+0.02`, and survival no less than `-0.03`; a demonstrated safety regression
   overrides offense. Also inspect bombs and bombs-per-kill for indiscriminate
   bombing.
4. Timing: zero timeout breaches and true global maximum below 500 ms, with
   substantial margin for slower tournament hardware.

Do not choose a favorable checkpoint. Only a candidate passing both attack
gates and all safety/timing guards should advance to Seeds 12/13. Failure should
end this feature experiment rather than trigger unregistered variants of the
pressure formula.

## Follow-up: occupied-source nuisance resolved

The implementation now skips a hypothetical source when it belongs to
`opponent_positions`, in addition to the existing field and bomb checks. The
target-opponent/source alias described above can therefore no longer produce a
nonzero value. The focused trapped-opponent test now also asserts that channel
11 is exactly zero at the opponent's own CNN tile `[y=7,x=5]`, while preserving
the valid pressure value at the agent's source tile.

Both actual configurations again pass all 15 scaffold tests independently
(`opponent_escape_pressure_zero_v1` and `opponent_escape_pressure_v1`, Seed 11,
1000 episodes). This removes the identified pre-run nuisance without changing
the causal contrast: both arms share the code and legal-source mask, while only
zero versus enabled channel content differs. The static-danger/path-counting
approximations remain documented limitations, not run blockers. **The pre-run
blocker is resolved and the pilot may proceed under the registered gates.**

## Quick100 stop/go audit

### Integrity

Both training runs are complete: 1000 sequential rows, ten checkpoints each,
and final state tensors identical to episode-1000. Final-model SHA-256 is
`965ae09debc9a175d9252da39d2d53d98cf4eec38a22c7e93ef0fbe36542e567`
for zero and
`25764bca353bebcecb999b3cec980c5362bd67a38b05d347f492a9536bcd4de8`
for enabled. Each quick evaluation has exactly 400 rows/100 complete rounds,
default seeds `20260731..20260830`, DQN slot 0 and three rule-based opponents.
Eval provenance attributes the matching final-model hash, trained variant and
Seed 11 to each arm; environment snapshots select the correct arm and 1000
training episodes. The callback hash is identical, as expected for an
environment-selected feature mode.

One metadata weakness remains: the train hyperparameter snapshot does not
contain an explicit `opponent_escape_pressure_mode` key (the queried value is
absent), although `training_arm`, input count/source and `bm_env` identify the
mode unambiguously. This is a traceability omission, not evidence that the wrong
arm ran, because evaluation provenance and output hashes also differ correctly.

### Gate evidence (enabled minus zero)

The primary tool reports score `+0.130` with CI `[-0.510,+0.770]`, sign-flip
`p=0.7178`, and kills `+0.070` with `[-0.040,+0.180]`, `p=0.2854`. The score
point estimate only barely clears the `0.12` practical floor and neither attack
effect is demonstrated. Quick100 was not powered to require significance, but
these values provide no positive evidence capable of offsetting safety harm.

Safety is internally coherent and much larger than the registered margins:

- suicides `+0.150`, CI `[+0.020,+0.280]`, sign-flip `p=0.0338`, non-fragile,
  versus the `+0.03` maximum;
- killed by opponent `+0.040`, CI `[-0.040,+0.120]`, versus the `+0.02` point
  guard;
- survival `-0.190`, direct paired CI `[-0.320,-0.060]`, sign-flip `p=0.0100`,
  non-fragile, versus the `-0.03` minimum;
- bombs `+4.910` per round, CI `[+1.930,+7.900]`, sign-flip `p=0.0010`.

The death changes sum correctly to the survival loss. More bombs, more own-bomb
deaths and lower survival support one mechanism—unsafe increased aggression—
rather than an isolated noisy suicide row. The pre-gate stated that demonstrated
safety regression overrides offense; both suicide and survival point effects
are five to six times their allowed margins, and survival clearly excludes zero.

Timing is operationally safe. Mean DQN think rises from `0.199 ms` to `1.084 ms`;
mean-of-round maxima rise from `1.44` to `3.69 ms`. True global maxima are
`36.84 ms` (zero) and `24.45 ms` (enabled), with zero timeout breaches across
both four-agent files. The apparent `analyze.py` “WORSE” timing row is far from
the 500-ms limit and does not drive the stop decision.

### Decision

This is **STOP for advancement**. The full1000 is not justified as a candidate
go measurement: the smoke test already detects a large, coherent violation of
the preregistered safety guards while attack remains unproven. Opponent stdlib
RNG means the runs are only arena-paired, and n=100 is not a final report-sized
estimate, but that caveat does not turn a large clear survival loss into evidence
for continuing.

Because the pre-audit described quick100 as smoke only, distinguish two claims.
Quick100 is sufficient for a resource/futility stop and rejection from further
development; it is not sufficient to quote the exact effect as the final report
result. A full1000 would be defensible only if the explicit goal is to document
this negative feature experiment at final measurement quality—not to rescue it,
advance it, run Seeds 12/13 or inspect checkpoints. If measurement budget is
scarce, stop now and record the quick safety failure with its sample-size caveat.
