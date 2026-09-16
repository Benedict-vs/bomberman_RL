# Independent Task-4 incumbent audit (2026-09-03)

## Scope and independent verdict

I inventoried every complete `n=1000` evaluation containing `ben_task4` under
`results/eval/task4_tournament`, grouping by evaluation field, training seed,
model hash and provenance. All listed 3RB files contain 4000 rows/1000 rounds
with Ben in list slot 0; the two external-trio files use a materially different
lineup and are not part of the same score leaderboard.

**No `ben_task4` policy or family has yet displaced the frozen `dqn_task3`
incumbent under the repository's evidence rules.** The current primary-score
incumbent remains the concrete frozen
`dqn_task3_seed13.pt`, SHA-256
`1285ac5cb78a25b0e6cc6a0e0a68fdd86e153db537832938f940cb1ad4fd8b63`.
It is the only policy here with replicated 1000-round 3RB measurements and all
four agent-list slots: its position-balanced DQN-minus-RB-mean score advantage
is `+0.344` `[+0.257,+0.431]`, non-fragile. That validation outweighs a higher
single slot-0 raw mean selected from many experiments.

Within `ben_task4`, `baseline_v1` Seeds 11/12/13 is the only developed
multi-training-seed family. It is a demonstrated **safety family**, not the
primary-score incumbent: equal-seed mean score is approximately `-0.037` versus
the frozen reference and kills `-0.036`, while suicides/killed-by/survival
improve consistently. Its individual scores `3.408/3.055/3.411` may not be used
to select Seed 11 or 13 post hoc.

## Why the raw-score leaderboard is not model selection

The largest 3RB slot-0 means among `ben_task4` are mixed-kill `3.892`,
safe-offense-02 `3.884`, alignment-mixed-zero `3.685`, the continued source
`c967` at `3.663`, and several controls around `3.5--3.6`. This ordering cannot
be read as a tournament ranking:

- all are fixed list-slot 0, whereas list-order robustness has only been shown
  for frozen `dqn_task3`;
- most are one training seed and one partially reproducible evaluation draw;
- rule-based stdlib RNG gives about a `0.12` same-model score noise floor, so
  close raw gaps such as `3.892` versus `3.884` are meaningless;
- selecting the maximum after many arms creates winner's-curse/multiple-testing
  bias, even when each individual CSV has a valid CI;
- external-trio scores near `2.0` use a different opponent distribution and
  cannot be mixed into the 3RB ranking;
- metadata identifies distinct model hashes, but callback hashes changed as
  code accumulated. Provenance is adequate for each historical run, while a
  fresh evaluation is still needed to prove an older model behaves identically
  with the current packaged callbacks.

Several high raw scorers were also formally rejected by their own preregistered
comparisons: safe-offense-02 violated safety point guards; alignment-mixed-zero
was a placebo/control rather than the feature candidate; later exploration and
external-distribution arms failed primary or safety gates. A rejected arm does
not become accepted merely because it sits high in a retrospective sort.

## Status of `mixed_kill_v1`

`mixed_kill_v1` is a legitimate **separate score/aggression candidate**, not an
incumbent or validated family. Its exact predeclared 2000-episode endpoint has
model SHA-256
`d50ae3d1ce80018a8a834f7df28cff3dfead5c0c99d578d7af1604a3cf0806c6`.
Against its equal-length `rule_based_continue_v1` control in 1000 rounds it
showed:

- score `+0.286` `[+0.084,+0.484]`, sign-flip `p=.0064`, non-fragile;
- kills `+0.065` `[+0.031,+0.099]`, `p=.0003`, non-fragile;
- suicides `+0.143` `[+0.102,+0.183]` and killed-by `+0.029`
  `[+0.010,+0.049]`, both clearly worse;
- survival fell from `.669` to `.497` (about `-.172`).

Thus its score result is stronger evidence than a raw maximum: it has a matched,
preplanned control and the intended kill mechanism. Tournament total score is
the primary objective, so retaining the frozen endpoint as a score-first
candidate is scientifically legitimate. It must not be rewritten as a clean
improvement under the original acceptance rule: it failed explicit safety
guards. It is best described as a demonstrated Seed-11 offense/safety trade-off.

The attempt to consolidate it for another 1000 episodes against 3RB failed to
preserve the advantage (score `-0.310` against its matched continuation control,
kills `-0.031` fragile). That does not invalidate the frozen 2000-episode model,
but it shows the behavior is not robust to further training and argues against
calling it a stable training family. No Seeds 12/13 of the rejected unchanged
arm should be launched merely to rescue it.

Compared with frozen Task-3, mixed-kill's absolute safety is not universally
worse—the frozen baseline itself dies often. “Safety trade-off” here means a
clear regression from the deliberately safe matched source/control, not proof
that mixed-kill is the least safe available policy. This distinction should be
kept when tournament score and catastrophic failure risk are weighed.

## Incumbent and candidate labels

- **Current Task-4 incumbent:** frozen `dqn_task3_seed13.pt`, because its primary
  score advantage over 3RB is replicated and list-position balanced.
- **Current safe development family:** `ben_task4_baseline_v1` Seeds 11/12/13,
  without selecting a best seed and without a primary-score improvement claim.
- **Score-first challenger:** the exact frozen `mixed_kill_v1` Seed-11 model,
  retained with its explicit safety regression. It is not yet a family or
  submission choice.
- Other high single-run controls/placebos remain experiment-specific references,
  not candidates selected from the retrospective leaderboard.

## Minimal fresh validation for mixed-kill

The minimum defensible staged program is:

1. **Current-code compatibility and opponent-noise replication.** Re-evaluate
   the exact frozen mixed-kill hash for a fresh 1000-round 3RB slot-0 run under a
   new label and current callbacks, without replacing the original. Require its
   absolute score/kill advantage to remain consistent with the original and
   report both draws. This tests packaging drift and a new stdlib-opponent draw.
2. **List-slot robustness.** If replication holds, run the same frozen model in
   list slots 1, 2 and 3 for 1000 rounds each, keep every result, and aggregate
   within arena seed with slots equally weighted. Compare the resulting
   position-balanced score, kills and deaths with the already position-balanced
   frozen incumbent; do not select the best slot. A single extra slot cannot
   support a position-robust claim.
3. **Heterogeneous confirmation.** Run mixed-kill first through Quick100 safety/
   liveness and then one predeclared full1000 against the exact external trio
   (Li-Jesse, bindist, binary) in a fixed documented order. This tests whether
   its 3RB aggression transfers to a strong coin/survival opponent and two
   aggressive bombers. Treat it as fixed-lineup evidence; rotate later only if
   making a universal external ranking claim.

For a tournament submission decision, predeclare how these fields are weighted:
official score remains primary, while suicides, killed-by, survival and global
timing are explicit risk diagnostics. Do not silently restore the old safety
guards only when mixed-kill loses, or discard them only because it scores well.
The candidate must have zero timeouts and ample margin below 500 ms; its existing
3RB global maximum (~22 ms) is safe but does not replace fresh packaged timing.

This validation concerns the **existing concrete policy**, so it does not
violate the earlier decision against more training seeds of the failed arm. A
claim about a reproducible mixed-kill *training family* would still require a
new preregistered multi-seed design and would conflict with spending budget to
rescue the rejected original. The economical next step is therefore fresh
evaluation of the frozen challenger, not retraining.

## Fresh mixed-kill slot-0 replication audit

### Integrity and provenance

The original and `retry1` files each contain exactly 4000 rows/1000 complete
rounds, default seeds `20260731..20261730`, Ben in list slot 0 and three
rule-based opponents. Both metadata files identify the exact same frozen
mixed-kill model SHA-256
`d50ae3d1ce80018a8a834f7df28cff3dfead5c0c99d578d7af1604a3cf0806c6`,
trained variant, Seed 11 and classic rules. Score/death arithmetic is valid and
there are zero timeout breaches.

The callback hash changed from `f50de2...17c0` to current `4270ab...f4ef` as
new disabled experiment branches were added. Therefore the runs are not
byte-identical software replications. Retry1 is precisely the requested
current-code compatibility check: the same model remains performant under the
current callback package. Similar outputs are evidence of behavioral
compatibility, although they cannot prove every code addition is semantically
inert.

### Same-policy repeatability

Retry1 versus original changes Ben's own metrics as follows:

- score `3.761 - 3.892 = -.131`, CI `[-.339,+.077]`, sign-flip `p=.226`;
- kills `.176 - .184 = -.008`, `[-.043,+.028]`;
- suicides `.410 - .431 = -.021`, `[-.064,+.023]`;
- killed by opponent `.070 - .072 = -.002`, `[-.024,+.020]`;
- survival `.520 - .497 = +.023` by death arithmetic.

No own-policy component change is demonstrated. The score movement is about the
known `.12` same-model opponent-RNG noise scale and warns against quoting only
the original `3.892`, but does not refute replication. Only one of 1000 full
four-agent outcome vectors is identical across executions, confirming that
common arena seeds do not reproduce stdlib-random opponent trajectories.

True Ben global think maxima are 22.24 ms and 15.58 ms, with no overruns. The
mean-of-round-max timing shift is operationally irrelevant.

### Within-game score versus opponent mean

To avoid treating three opponents as independent replications, each round was
reduced to Ben minus the mean of its three rule-based scores:

- original score advantage `+1.093`, CI `[+.880,+1.306]`, sign-flip
  `p<.0001`, non-fragile;
- retry1 score advantage `+.868`, `[+.670,+1.074]`, `p<.0001`, non-fragile.

Both advantages are primarily coins (`+.973`, then `+.851`). Kill advantages
are small and unproven (`+.024`, then `+.003`). In both runs Ben has clearly
lower `killed_by` than the RB mean (`-.029`, `-.030`) and higher survival
(`+.049`, `+.061`); suicide differences point lower but include zero. Thus the
challenger's absolute behavior in this matchup is not catastrophically unsafe,
even though it clearly regressed from its deliberately safer matched training
control.

The replication is not as strong as a naive “3.892 repeated” claim. Ben drops
`.131` while the RB pooled mean rises from about `2.799` to `2.893`, so its
within-field advantage contracts by roughly `.225`. This is exactly the
opponent-noise mechanism the retry was meant to expose. The key result is that
the direction and a large margin survive, not that the means repeat.

### Challenger decision

I looked for reasons to stop before slots: callback drift, opponent draw,
coin-only mechanism and known safety regression are all real caveats. None
invalidates the predeclared challenger gate. The same frozen model remains near
its original own score/kills/deaths, and its primary tournament score advantage
over the within-round opponent mean is large, clear and non-fragile twice.

The original safety guards still forbid calling mixed-kill an accepted safe
replacement or a successful training family. They do **not** forbid evaluating
a frozen score-first challenger when tournament total score is explicitly the
primary selection metric. In fact, relative to the actual RB field, its death
metrics are competitive; the trade-off is specifically versus its safer
control.

Therefore the preplanned condition is met: **run list-slot 1/2/3 measurements
of the exact unchanged SHA**. Keep all three 1000-round files, use identical
arena seeds and lineups, and aggregate within arena before weighting slots
equally. Do not retrain, choose only favorable slots, or promote it to incumbent
until the position-balanced score comparison is complete. If the advantage
survives slots, proceed to the separately planned external-trio confirmation;
if not, frozen `dqn_task3` remains incumbent without further rescue tests.
