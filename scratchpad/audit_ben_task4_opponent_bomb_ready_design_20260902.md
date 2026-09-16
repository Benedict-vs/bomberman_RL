# Adversarial pre-run audit: opponent bomb readiness (2026-09-02)

## Verdict

`opponent_bomb_ready_zero_v1` versus `opponent_bomb_ready_v1` is a clean
one-feature 1000-episode Seed-11 pilot. Both arms share the exact converted
12-channel source, rewards, hyperparameters, training field and tensor/replay
paths. The only intended difference is whether channel 11 remains zero or marks
opponents that can currently place a bomb. No ML-rule, geometry, runtime or
overwrite blocker was found.

The feature is primarily threat context, not an attack oracle. A positive kill
effect is still required for an “offense improvement” claim, while lower
`killed_by` would support a safety mechanism. A score-only effect must be
decomposed rather than retroactively assigned to either interpretation.

## Source and causal isolation

- Both arms load
  `ben_task4_safe_seed11_12ch_alignment_source.pt`, SHA-256
  `f0e153902c930770739a29fc5bdf4d755405534f3ca3a46f4bf9b59837cceb65`.
  Prior conversion tests establish that the first 11 input-channel weights are
  identical to the frozen safe source and channel 11 begins at exactly zero.
- Both use 12 inputs, reachable-safe-tiles, disabled opponent alignment and
  escape-pressure features, identical visit-count/action-mask/symmetry settings,
  the same DQN, optimizer and replay hyperparameters, Seed 11 and 1000 episodes.
- Base rewards and shaping are identical, including `KILLED_OPPONENT=+5`,
  `GOT_KILLED=-5`, `KILLED_SELF=0`, and no safe-offense helper.
- Both declare and must actually run against exactly three
  `rule_based_agent`s. As elsewhere, the metadata string does not enforce the
  CLI; preserve the command with the experiment record.
- `OPPONENT_BOMB_READY_MODE` alone is `zero` versus `enabled`. Metadata records
  this mode explicitly alongside source, input channels, rewards and lineup.

In the zero arm channel 11 is spatially and temporally zero, so its new source
weights remain behaviorally inert. The candidate alone supplies signal and can
learn channel-11 weights. No old alignment behavior is inherited through that
channel.

## Tuple semantics and coordinates

The framework defines each agent state tuple as
`(name, score, bombs_left, (x,y))`. The implementation correctly reads
`bool(other[2])` and coordinates from `other[3]`. It writes a 1 only for a ready
opponent at CNN position `[channel, y, x]`; an unready opponent remains present
in the ordinary opponent channel but is zero in bomb-ready. This distinguishes
“opponent exists” from “opponent can currently bomb” without losing location.

`bombs_left=True` means availability now, not an intent to bomb or probability
of bombing. The feature does not predict the opponent policy, future recharge,
blast geometry or whether Ben is in danger. Those remain learned relationships
with the existing opponent, bomb, danger, self and escape channels. Claims must
therefore say “bomb availability”, not “incoming bomb”.

## State, replay and augmentation consistency

Callbacks inject the selected mode before every `act` feature extraction. The
training path injects it for old/cached action states, next-state preview and
the helper used when visit counts are unavailable. Both arms therefore always
produce `(12,17,17)` arrays, and replay cannot mix 11- and 12-channel samples.

The ready map is an ordinary spatial channel. Existing rotation/reflection
augmentation transforms it with the other board channels; the Boolean value is
orientation-independent and the position rotates correctly. Next-state
readiness is recomputed from that state's `others` tuples rather than copied
from the old state. No action-dependent coordinate remapping is required.

## ML rule and cost

The channel reports a directly observed game-state fact at opponent positions.
It does not output a recommended move, bomb target, path or best action. The DQN
must learn whether and how bomb availability affects movement, attack and
escape. It therefore satisfies the hard ML rule and remains a learned agent.

Feature cost is `O(number of opponents)`—at most three Boolean/coordinate checks
and writes—negligible relative to the CNN and existing safety/pathfinding
features. It adds no search and no training-only dependency. Final evaluation
must still inspect global timing because the tournament limit applies to the
whole agent, not individual features.

## Tests and output safety

Both actual environments independently pass all 18 scaffold tests. Focused
coverage verifies the official tuple layout, a ready and unready opponent at
distinct nonsymmetric coordinates, correct `[y,x]` placement, exact zero-mode
sum, 12-channel shape, and cross-process equality of source, feature modes,
rewards and three-RB lineup with only zero/enabled differing.

For each arm, the final model, training CSV, training meta, log directory and
all ten checkpoints (`100..1000`) were checked and are absent. Planned execution
will not overwrite an existing artifact.

## Preregistered gates

Use Quick100 only as smoke; apply the decision to candidate minus zero on
full1000 default-seed evaluations. Rule-based stdlib RNG remains incompletely
seeded, so pairing is on arenas only and one training seed is a pilot.

1. Primary score must be positive, exceed `+0.12` in point estimate, have a
   paired 95% CI excluding zero, sign-flip `p<.05`, and be non-fragile.
2. For an offense interpretation, kills must be positive with CI excluding zero,
   sign-flip agreement and non-fragile verdict. Score without kills is not
   evidence that readiness improved attack.
3. Safety guards: suicides `<= +0.03`, killed-by-opponent `<= +0.02`, survival
   `>= -0.03`. Because readiness is naturally a threat feature, a clear negative
   `killed_by` effect is useful mechanism evidence, but safety alone does not
   satisfy the registered offense/score gates. Inspect bombs and coins to
   distinguish behavioral substitution.
4. Zero timeout breaches and true global `think_max_ms < 500`, with substantial
   margin for slower reference hardware.

Only a model passing score, kill, all safety and timing gates should advance to
Seeds 12/13 as an offense candidate. If it instead produces a clear safety gain
without score/kills, retain that as a separately named safety finding rather
than changing the success definition. Do not select checkpoints or combine this
channel post hoc with failed feature arms.

## Quick100 stop/go audit

### Integrity and attribution

Both arms completed 1000 sequential training rows, all ten checkpoints, and
their final model tensors equal episode-1000. Final SHA-256s are
`7b57d5d363bf725ea857cf698253ce1522cf21ebba2d802d44dc2ac7c193afd0`
(zero) and
`51e0b1070a088db5d60bc051f6a7961950b352481682d5e313ef6bb76efe4d61`
(enabled). Training metadata records the same converted source, three-RB lineup,
Seed 11/1000 episodes and explicit `zero` versus `enabled` mode.

Each quick evaluation has exactly 400 rows/100 complete rounds, default seeds
`20260731..20260830`, Ben in slot 0 and three rule-based opponents. Eval metadata
selects the correct arm and exact matching final-model hash; callback hashes are
identical, as expected for an environment-selected channel. No provenance or
lineup mismatch was found.

### Candidate minus zero

- score `3.56 - 4.05 = -0.49`, CI `[-1.08,+0.07]`, sign-flip
  `p=.1157`;
- kills `.09 - .15 = -.06`, `[-.15,+.02]`, `p=.2673`;
- coins `3.11 - 3.30 = -.19`, `[-.53,+.14]`, `p=.3023`;
- suicides `0.00`, killed-by `0.00`, and survival `0.00`, each with broad but
  symmetric intervals around zero;
- bombs `24.05 - 19.84 = +4.21`, CI `[+1.56,+6.80]`, sign-flip
  `p=.0023`, clear and non-fragile.

The score/kills rows do not individually prove harm at n=100, but they provide
strong futility evidence. Most importantly, the upper score bound is only
`+0.07`, below the preregistered practical requirement `>+0.12`; the kill upper
bound is just `+0.02`. A practically acceptable offense effect is not supported
even at the favorable edge of these Quick100 intervals. Meanwhile the one clear
behavioral change is substantially more bombing, without more kills, score or
safety. That pattern contradicts both an offense mechanism and the natural
threat-awareness interpretation.

### Timing and stochastic uncertainty

Timing passes: mean think is about `0.195 ms` in both arms, true Ben/global
maxima are 21.96 ms (zero) and 11.78 ms (enabled), and there are zero timeout
breaches. Feature cost does not drive the decision.

Quick100 remains arena-paired only because rule-based stdlib RNG is incompletely
seeded. The known opponent-noise floor and small sample mean the exact `-0.49`
should not be quoted as a final effect. But stochastic uncertainty is already
represented by wide intervals; invoking it cannot turn the observed data into
positive mechanism evidence. A fresh opponent sequence could differ, which is
why this is a futility decision rather than a final population claim.

### Decision

I attempted to justify Full1000 but the preregistered practical gate makes the
case weak: unlike a merely inconclusive smoke test, the score CI does not even
reach the required `+0.12`, kills point downward, and the sole clear change is
unproductive extra bombing. **Futility stop for advancement** is justified. Do
not run Seeds 12/13, inspect checkpoints or combine this channel with another
failed arm.

A Full1000 remains scientifically permissible only to document this negative
feature experiment at report-grade sample size. It is not warranted as a go or
rescue measurement when budget is scarce. If skipped, record Quick100 explicitly
as an early futility screen and do not present its exact numerical effect as a
final reported result.
