# E40 — results

## Control validation (shipped agent published: score 3.949, kills 0.226, suicides 0.488, crates 33.55)

- score: 4.009
- kills: 0.236
- suicides: 0.477
- crates: 33.514
- won: 0.387

## P1 + P4 — the corrected oracle against the control

### k4sim  vs  ctl   (8000 paired arenas)

| metric | ctl | k4sim | paired difference | sign-flip p | verdict |
|---|---|---|---|---|---|
| score | 4.009 | 4.062 | +0.053 [-0.028, +0.132] | t=+1.24, p=0.2160 | nicht gezeigt |
| margin_mean | 1.024 | 1.121 | +0.097 [-0.003, +0.193] | t=+1.87, p=0.0652 | nicht gezeigt |
|  ^ | | | seed-stability of that verdict: 20% of 15 bootstrap seeds | | |
| margin_best | -1.380 | -1.217 | +0.163 [+0.032, +0.290] | t=+2.48, p=0.0142 | BESSER |
| kills | 0.236 | 0.255 | +0.019 [+0.004, +0.033] | t=+2.53, p=0.0150 | BESSER |
| won | 0.387 | 0.406 | +0.018 [+0.004, +0.033] | t=+2.47, p=0.0154 | BESSER |
| coins | 2.827 | 2.787 | -0.040 [-0.077, -0.006] | t=-2.25, p=0.0226 | SCHLECHTER |
| crates | 33.514 | 33.453 | -0.060 [-0.301, +0.172] | t=-0.50, p=0.6248 | nicht gezeigt |
| suicides | 0.477 | 0.466 | -0.011 [-0.026, +0.004] | t=-1.43, p=0.1644 | nicht gezeigt |
| survived | 0.464 | 0.475 | +0.011 [-0.004, +0.026] | t=+1.40, p=0.1710 | nicht gezeigt |

## P2 — corrected oracle against the broken one (the reason this entry exists)

### k4sim  vs  k4stale   (8000 paired arenas)

| metric | k4stale | k4sim | paired difference | sign-flip p | verdict |
|---|---|---|---|---|---|
| score | 4.084 | 4.062 | -0.022 [-0.104, +0.063] | t=-0.52, p=0.6040 | nicht gezeigt |
| margin_mean | 1.148 | 1.121 | -0.027 [-0.126, +0.075] | t=-0.51, p=0.6174 | nicht gezeigt |
| margin_best | -1.195 | -1.217 | -0.022 [-0.148, +0.107] | t=-0.35, p=0.7476 | nicht gezeigt |
| kills | 0.259 | 0.255 | -0.004 [-0.018, +0.011] | t=-0.49, p=0.6350 | nicht gezeigt |
| crates | 33.217 | 33.453 | +0.236 [+0.007, +0.465] | t=+1.96, p=0.0504 | nicht gezeigt  <-- CI and p disagree |
|  ^ | | | seed-stability of that verdict: 53% of 15 bootstrap seeds | | |

## Bridge — the broken oracle against the control (reproduces E39)

### k4stale  vs  ctl   (8000 paired arenas)

| metric | ctl | k4stale | paired difference | sign-flip p | verdict |
|---|---|---|---|---|---|
| score | 4.009 | 4.084 | +0.075 [-0.009, +0.157] | t=+1.77, p=0.0740 | nicht gezeigt |
| margin_mean | 1.024 | 1.148 | +0.123 [+0.021, +0.222] | t=+2.38, p=0.0168 | BESSER |
| margin_best | -1.380 | -1.195 | +0.185 [+0.055, +0.314] | t=+2.82, p=0.0050 | BESSER |
| kills | 0.236 | 0.259 | +0.022 [+0.007, +0.037] | t=+3.01, p=0.0022 | BESSER |
| crates | 33.514 | 33.217 | -0.296 [-0.526, -0.061] | t=-2.43, p=0.0144 | SCHLECHTER |

## P5 — reach

### k8sim  vs  k4sim   (8000 paired arenas)

| metric | k4sim | k8sim | paired difference | sign-flip p | verdict |
|---|---|---|---|---|---|
| score | 4.062 | 3.985 | -0.076 [-0.164, +0.003] | t=-1.77, p=0.0820 | nicht gezeigt |
| margin_mean | 1.121 | 1.026 | -0.095 [-0.205, +0.002] | t=-1.83, p=0.0696 | nicht gezeigt |
|  ^ | | | seed-stability of that verdict: 7% of 15 bootstrap seeds | | |
| margin_best | -1.217 | -1.340 | -0.122 [-0.258, +0.005] | t=-1.87, p=0.0632 | nicht gezeigt |
| kills | 0.255 | 0.252 | -0.003 [-0.018, +0.010] | t=-0.40, p=0.7070 | nicht gezeigt |
| crates | 33.453 | 32.546 | -0.907 [-1.150, -0.663] | t=-7.28, p=0.0000 | SCHLECHTER |
