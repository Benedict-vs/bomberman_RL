# Mixed-kill position balance — primary analysis (2026-09-04)

This is the primary analysis, not the required independent adversarial audit.
Reproduce it with:

```bash
.venv/bin/python scratchpad/ben_task4_mixed_kill_position_balance.py
```

## Estimator

The unit of analysis is one of the 1,000 standard arena seeds. For each policy,
results are first averaged within a list slot and arena seed, then the four list
slots are weighted equally. The incumbent's two callback-identical slot-0
replications are averaged before slot weighting. Mixed-kill uses only its fresh
current-callback slot-0 replication: its older slot-0 result has a different
callback SHA and is excluded from the primary estimate. This avoids treating
four repeated uses of each arena as independent observations.

The selected mixed-kill runs all use callback SHA
`4270ab56cf95f358c9b67dfc404d7e437bcee97494bf93df09b935a371c2f4ef`
and model SHA
`d50ae3d1ce80018a8a834f7df28cff3dfead5c0c99d578d7af1604a3cf0806c6`.
The incumbent runs all use callback SHA
`3a601738fef20b2f6101a26bece2908a2b92193cd52c6298c04c1cd5b925d11f`
and model SHA
`1285ac5cb78a25b0e6cc6a0e0a68fdd86e153db537832938f940cb1ad4fd8b63`.
Every selected file contains 4,000 rows over 1,000 standard seeds under the
same game rules, its agent occupies the expected list slot, score equals
`coins + 5 * kills` in every row, and no action exceeded the time limit.

## Result

| Metric | Mixed-kill | Incumbent | Difference | Paired 95% CI | Sign-flip p | Fragile |
|---|---:|---:|---:|---:|---:|---:|
| Score | 3.8070 | 3.3329 | +0.4741 | [+0.3727, +0.5751] | <0.00005 | no |
| Won | 0.4165 | 0.3312 | +0.0853 | [+0.0656, +0.1051] | <0.00005 | no |
| Kills | 0.1875 | 0.1434 | +0.0441 | [+0.0270, +0.0614] | <0.00005 | no |
| Suicides | 0.3935 | 0.4955 | -0.1020 | [-0.1232, -0.0804] | <0.00005 | no |
| Killed by opponent | 0.0732 | 0.1075 | -0.0343 | [-0.0456, -0.0229] | <0.00005 | no |
| Survived | 0.5333 | 0.3970 | +0.1363 | [+0.1160, +0.1561] | <0.00005 | no |
| Coins | 2.8695 | 2.6160 | +0.2535 | [+0.2014, +0.3045] | <0.00005 | no |
| Bombs | 20.5670 | 23.8360 | -3.2690 | [-3.7143, -2.8205] | <0.00005 | no |

The score differences by list slot are `+0.4325`, `+0.5250`, `+0.5010` and
`+0.4380`; no position reverses the direction. Maximum observed inference time
was `46.7317 ms` for mixed-kill, well below the `500 ms` tournament limit.

## Interpretation boundary

This supports the narrow claim that the concrete frozen mixed-kill Seed-11
policy outperforms the concrete frozen Task-3 Seed-13 policy in the 3RB field
across list slots. It does not establish that mixed-kill training is a
reproducible multi-seed training family, nor that the ranking transfers to
external opponent fields. The provided opponents' stdlib randomness is not
seeded, so pairing controls arenas and slots but not opponent trajectories.
Promotion and external confirmation remain pending an independent adversarial
audit that tries to refute this analysis.
