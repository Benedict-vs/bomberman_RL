# Audit: external-agent quick evaluations (2026-09-01)

## Integrity and attribution

The four separate versus-3RB files and the direct top-four file each contain
exactly 400 rows: 100 rounds, four slots per round, round indices `0..99`, and
default seeds `20260731..20260830`. Metadata agent lists agree with every CSV
slot. Each separate game puts the external candidate in slot 0 and three
`rule_based_agent`s in slots 1--3. The direct game uses, in fixed slot order,
`bindist_v2`, `binary_v6`, `aielka_ql_atom`, and `lijesse_featureeverything`.

The install ledger pins source commits (`50b682f`, `8d85731`, `a7fe504`) and its
loader verification demonstrates that trained parameters/tables were actually
loaded rather than fresh initializations. Current model hashes still match its
recorded xiaoxiae SHA-1s (`bb9f8953c1051ada`, `87cb349c029563ea`); current
SHA-256s are respectively `4fcccb...5221`, `adabec...bb6`, Aielka table
`917724...144f`, and Li-Jesse model `0fb1df...0577`. Evaluation metadata records
matching current callback hashes, but does **not** record external model-file
hashes. Attribution therefore rests on the pinned installation audit plus the
unchanged installed files, not on eval metadata alone. Future reported external
measurements should snapshot those model hashes in their experiment record.

These external agents are legitimate evaluation/training opponents only; their
code or learned logic must not be copied into the submitted agent. Two sources
have no license, strengthening the need to keep their folders outside the
submission/public deliverable.

## What quick100 does and does not show

Against separate 3RB fields, own mean scores rank bindist `5.81`, Aielka `5.43`,
binary `5.39`, Li-Jesse `4.97`. In the direct top-four game the order changes to
Li-Jesse `3.43`, bindist `3.14`, binary `2.79`, Aielka `2.76`. Direct kills also
differ substantially (bindist `.23`, Aielka `.22`, binary `.19`, Li-Jesse
`.07`). The reversal refutes any claim that the separate quick scores identify
the best agent in their mutual matchup: opponent composition and interaction
matter strongly.

All four quick samples are too small for a reported rank. Opponent behavior is
only partly reproducible because stdlib `random` is unseeded; common arena seeds
do not make separate games behavior-paired. The fixed slot allocation is another
confound: each candidate is slot 0 in the separate games, while each occupies a
different fixed start in the direct game. Consequently neither differences
between the four separate files nor direct top-four gaps of only a few tenths
support a final superiority claim.

## Timing

There are zero 500-ms overruns. Worst single-step maxima in the separate games
are 17.8 ms (Aielka), 38.4 ms (Li-Jesse), 81.9 ms (binary), and 45.5 ms
(bindist); in the direct game they are 19.5, 44.2, 54.8, and 67.4 ms. Li-Jesse
has by far the largest mean think time (11.6 ms separately, 15.2 ms directly).
All remain locally safe, but timing is not portable to the slower reference CPU.

Metadata timestamps minus wall-clock durations indicate the five evaluations
ran sequentially rather than overlapping, so these quick timings were not
inflated by separate evaluation jobs competing in parallel. Inside a direct
four-agent game, however, the framework's agent processes may contend for CPU;
the all-four timing is the relevant system measurement, and a full run should
be executed alone with no simultaneous training/evaluation. Score comparisons
remain usable because no timeout occurred, but timing differences between the
separate and direct fields should not be attributed solely to model complexity.

## Next measurement

The quick results justify **one predeclared direct all-four 1000-round
evaluation**: it measures the actual interaction that the separate 3RB screens
cannot. Run it alone, retain default evaluation seeds, fixed model hashes and
the exact agent order, and treat score as primary with deaths and global timing
as diagnostics. Do not choose a checkpoint or reorder agents after inspecting
partial results.

One fixed-order full1000 remains lineup- and slot-specific, so it can confirm a
leader in that matchup but cannot establish position-robust ranking. If the
result will be used to select a supposedly strongest external opponent, follow
with preregistered slot rotations (or explicitly limit the claim to this fixed
lineup). The present quick100 evidence supports spending measurement budget on
the direct full run; it does not itself rank the four.

## Post-run audit: direct all-four full1000

### Completeness and provenance

`external_top4__task4_head_to_head_eval1000_v1.csv` is complete: 4000 rows,
1000 rounds numbered `0..999`, four rows per round, and default arena seeds
`20260731..20261730`. CSV slots exactly match metadata: bindist in slot 0,
binary in slot 1, Aielka in slot 2 and Li-Jesse in slot 3. The meta records
`classic`, 1000 rounds, the expected callback hashes and unchanged framework
rules.

As in quick100, metadata does not hash external model files. Current installed
model SHA-256s nevertheless remain exactly those recorded in the installation
audit: bindist `4fcccb...5221`, binary `adabec...bb6`, Aielka
`917724...144f`, Li-Jesse `0fb1df...0577`. Together with matching callback
hashes and the prior loader verification, this supports attribution to the
pinned trained models, but future measurement metadata should capture their
model hashes directly.

All 4000 rows satisfy `score = coins + 5*kills` and
`survived + suicides + killed_by = 1`. No row is missing and no death category
is double-counted.

### Result decomposition

Fixed-order means are:

| agent | score | coins | kills | suicide | killed by | survived |
|---|---:|---:|---:|---:|---:|---:|
| bindist | 2.912 | 1.832 | 0.216 | 0.324 | 0.138 | 0.538 |
| binary | 2.975 | 1.965 | 0.202 | 0.442 | 0.162 | 0.396 |
| Aielka | 2.446 | 1.571 | 0.175 | 0.336 | 0.067 | 0.597 |
| Li-Jesse | **3.720** | **3.120** | 0.120 | 0.345 | 0.078 | 0.577 |

Li-Jesse is the highest scorer, but not the strongest killer. Direct within-round
Li-Jesse-minus-other score contrasts are clear and non-fragile: `+0.808`
`[+0.570,+1.052]` against bindist, `+0.745` `[+0.497,+0.994]` against binary,
and `+1.274` `[+1.043,+1.505]` against Aielka (all sign-flip `p<0.0001`). These
gaps exceed the known `0.12` evaluation noise floor.

The mechanism is entirely coin dominance. Li-Jesse gains respectively
`+1.288`, `+1.155`, and `+1.549` coins, while recording **fewer** kills:
`-0.096`, `-0.082`, and `-0.055`; all three kill deficits have CIs excluding
zero and non-fragile sign-flip agreement. Its survival is clearly above binary
(`+0.181`) but not distinguishable from bindist (`+0.039`) or Aielka (`-0.020`).
It is therefore accurate to call Li-Jesse the highest-scoring farmer/survivor in
this lineup, not the most aggressive or universally strongest agent.

The direct quick100 had already put Li-Jesse first (`3.43` versus bindist
`3.14`); full1000 preserves and enlarges that direction. Quick100 is useful only
as this consistency check, not as additional independent evidence because it
uses the first 100 seeds of the same evaluation schedule and likely overlaps the
same arena set.

### Timing and CPU caveat

There are zero recorded 500-ms overruns. Mean think times are about 1.12 ms
(bindist), 1.11 ms (binary), 1.82 ms (Aielka), and 14.51 ms (Li-Jesse). True
global single-step maxima are 190.3, 293.0, 24.3, and 42.5 ms, respectively.
Thus all completed locally within budget, although binary's 293-ms outlier has
poor margin for the slower reference CPU.

The four agents act in separate processes and may compete for CPU inside the
same game. Those timings describe this full system on this machine, not isolated
model inference, and should not be scaled into precise reference-hardware
predictions. The score result is not timeout-driven because no limit breach
occurred. Future timing qualification should run without concurrent unrelated
jobs and distinguish global maximum from mean-of-round maxima.

### Slot bias and selection verdict

The strong within-round contrasts control arena difficulty and directly measure
competition for the same coins, but they do **not** control starting position:
each model occupied one fixed slot for all 1000 rounds. Opponent stdlib RNG is
also not fully seeded. A fixed slot could change access to coins, escape routes
and which opponents are encountered; statistical significance over rounds does
not remove this systematic confound. No claim of a position-robust ranking is
yet justified.

Li-Jesse may now be selected as a **provisional high-score benchmark** and as an
external training opponent when the intended pressure is coin denial, survival
and beating a strong total scorer. The full1000 evidence is much stronger than
the quick screen for that narrow choice. It does not justify calling Li-Jesse
the best combat opponent or using it as the sole Task-4 training distribution:
bindist and binary generate significantly more kills and may provide the more
relevant aggression pressure. Li-Jesse is also the slowest sustained opponent,
so it materially increases training wall time.

Before presenting Li-Jesse as the strongest external agent, rotate its slot (and
preferably all four cyclically) under a preregistered aggregation. If immediate
training budget precludes rotation, state the selection narrowly: “highest
score in one fixed-order 1000-round direct matchup,” retain at least one
aggressive opponent in a mixed lineup, and evaluate the learned agent against
both scorer and attacker archetypes. That is supported by these data; choosing
Li-Jesse alone as the definitive next opponent is not.

## Diagnostic audit: Ben versus external top three (Quick100)

### Integrity and limits

`ben_dqn_task4_rule_based_continue_control1000_v1_1000ep_seed11__task4_external_top3_quick100.csv`
is complete: 400 rows, 100 rounds `0..99`, default seeds
`20260731..20260830`, and exactly four agents per round. The fixed order is our
`ben_task4` in slot 0, Li-Jesse in slot 1, bindist in slot 2 and binary in slot
3. Metadata attributes our frozen control model with the correct SHA-256
`c9677540849340e915f74dec7cf9c2d24e9c6c27f9d5132453798035a1f8a9ec`,
trained variant and Seed 11. External callback hashes match the audited
installations; as before, their model-file hashes are supported by the install
audit but are not embedded in this eval metadata.

Every row satisfies score and death arithmetic. The contrasts below compare Ben
minus each opponent inside the same round, so arena difficulty and direct
competition are shared. They remain exploratory: n=100 is Quick100, stdlib RNG
is not fully controlled, and the fixed slot assignment can systematically
favor a start position. They diagnose a next experiment; they do not establish
a final rank.

### Independent result decomposition

Raw means:

| agent | score | coins | kills | suicide | killed by | survived |
|---|---:|---:|---:|---:|---:|---:|
| Ben | 2.55 | 2.30 | 0.05 | 0.46 | 0.12 | 0.42 |
| Li-Jesse | 3.50 | 2.80 | 0.14 | 0.25 | 0.09 | 0.66 |
| bindist | 2.70 | 1.90 | 0.16 | 0.47 | 0.10 | 0.43 |
| binary | 3.24 | 1.89 | 0.27 | 0.37 | 0.07 | 0.56 |

Against Li-Jesse, Ben loses score by `-0.95` (CI `[-1.57,-0.32]`, sign-flip
`p=0.0041`, non-fragile), coins by `-0.50` (`[-0.94,-0.06]`), and survival by
`-0.24` (`[-0.35,-0.12]`), while suicides are `+0.21`
(`[+0.09,+0.33]`). The kill deficit is `-0.09`; its CI narrowly excludes zero
but sign-flip `p=0.062` disagrees, so it is fragile and not demonstrated.

Against binary, Ben actually has `+0.41` coins (boundary/fragile) but loses
`-0.22` kills (`[-0.31,-0.13]`, `p<0.0001`) and therefore `-0.69` score
(`[-1.28,-0.10]`). Against bindist, Ben has a clear `+0.40` coin advantage
(`[+0.02,+0.80]`) and a clear `-0.11` kill deficit (`[-0.21,-0.02]`); those
effects offset, leaving score `-0.15` with no demonstrated difference. Ben's
suicide/survival differences versus bindist are essentially zero; versus binary
they point worse but are not established at n=100.

The primary diagnosis is therefore **failure to convert combat opportunities
into opponent kills**, not general navigation or coin failure. Ben is the lowest
killer by a wide raw margin and can out-collect both aggressive xiaoxiae agents
without outscoring them. A second, opponent-specific deficit appears against
Li-Jesse: Ben both loses the coin race and self-destructs more, producing a clear
survival gap. `killed_by` is not independently demonstrated against any one
opponent, so enemy-bomb avoidance alone is not the leading diagnosis.

### Timing

There are zero timeout breaches. Mean think times are approximately `0.28 ms`
(Ben), `13.75 ms` (Li-Jesse), and `1.22 ms` for each xiaoxiae agent. True global
maxima are 18.4, 38.9, 48.8 and 60.3 ms respectively, all locally below 500 ms.
This mixed external field is substantially slower because of Li-Jesse, and
training/evaluation should run without other CPU-heavy jobs. These timings are
system measurements with four agent processes, not isolated inference costs.

### Recommended next controlled pilot

The next pilot should target kill conversion **while preserving own-bomb
escape**, not add another unconditional aggression incentive. Prior Task-4
experiments already show that more bombing can raise kills while destroying
safety. Suicides must remain a hard regression guard, with coin score retained
as a secondary guard rather than sacrificed for kills.

The most informative opponent distribution is one Li-Jesse plus the two distinct
xiaoxiae policies (bindist and binary): Li-Jesse supplies high-score coin and
survival pressure, while bindist/binary supply the aggressive kill pressure
where Ben's clearest deficit appears. Training against Li-Jesse alone would
over-focus coin competition and underrepresent the demonstrated combat gap;
training only against bombers would ignore the clear Li-Jesse score/survival
failure.

Make opponent distribution the single causal intervention: start candidate and
control from the same frozen source with identical rewards, features,
hyperparameters, episodes and Seed 11; candidate trains against the exact mixed
external trio, while the control performs the corresponding continuation against
the established three-rule-based field. Evaluate both on the external trio and
also retain the 3RB evaluation as a regression check. Pre-register primary score,
kills as mechanism, and suicide/survival guards before the run. This first
Seed-11 result is only a pilot because opponent RNG is partial; advance to
multiple training seeds only after it passes. Do not select an external opponent,
checkpoint or slot arrangement after seeing partial results.
