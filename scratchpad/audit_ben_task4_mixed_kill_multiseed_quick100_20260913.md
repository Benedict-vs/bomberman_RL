# Adversarial audit: Mixed-Kill Mehrseed Quick100

## Scope and provenance

The two new policies each completed 2,000 training episodes (2,001-line CSVs,
including header) from the same `ben_task4_baseline_v1_5000ep_seed11.pt` source
in the same `peaceful_agent,rule_based_agent,rule_based_agent` training field.
Only the configured training seed differs: 12 versus 13.  The four evaluation
CSVs each contain 100 rounds (401 rows: four agents per round).  Model hashes
are `d21768...f94e` (seed 12) and `2b768d...5d81` (seed 13), so this is not a
duplicated checkpoint.

## Attempt to refute a seed-13 selection

Against three rule-based opponents, seed 13 exceeds seed 12 by `+0.510` score
with paired 95% CI `[−0.060,+1.080]` (sign-flip `p=0.08575`), hence the apparent
advantage is not demonstrated.  Against the external trio, its score difference
is only `+0.050 [−0.380,+0.510]`.  Its suicides are instead demonstrably worse:
`+0.270 [+0.140,+0.400]`, sign-flip `p=0.00005`.

## Verdict

Neither new seed is a field-robust challenger.  The 3RB point-estimate increase
for seed 13 is compatible with ordinary training/opponent variation, while its
external-field safety regression is clear and its score does not improve there.
Quick100 must not be used to cherry-pick a best seed.  Preserve the audited
seed-11 Mixed-Kill incumbent; run no Full1000 for either replicate and do not
infer that longer unchanged training would help.
