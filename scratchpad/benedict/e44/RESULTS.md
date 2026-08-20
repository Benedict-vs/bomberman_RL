============================================================================================================
E44 -- the 2x2.  8 seeds/cell, 300 rounds, validation seed 550731
============================================================================================================

## Cell means (score / margin_mean), for orientation

  cell                           heldout                    indist                     guard
  warm_rb                 2.965 / -1.000            3.342 / -0.045            3.832 / +0.795
  warm_mix                2.850 / -1.043            3.181 / -0.164            3.782 / +0.804
  scr_rb                  1.305 / -2.642            1.406 / -2.206            1.813 / -1.522
  scr_mix                 1.203 / -2.684            1.271 / -2.312            1.432 / -1.966

## P2 GUARD -- is the scratch row even trained? bar = 3.07 on the guard field
     (80 % of warm_rb's 3.832; if this fails, P1 is uninterpretable)

  scr_rb      guard score 1.813   *** FAIL ***   (per-seed [1.85 2.2  1.47 1.97 1.97 1.49 1.76 1.8 ])
  scr_mix     guard score 1.432   *** FAIL ***   (per-seed [1.59 1.44 1.3  1.36 1.12 1.72 1.49 1.43])

## P1 PRIMARY -- the field effect from scratch, on the HELD-OUT field  (bar >= +0.35)

  comparison                                     A       B     diff                95% CI  perm p
  scr_mix - scr_rb  [margin_mean]           -2.642  -2.684   -0.042      [-0.251, +0.181]  0.7260  nicht gezeigt
  scr_mix - scr_rb  [score]                  1.305   1.203   -0.102      [-0.283, +0.090]  0.3459  nicht gezeigt
  scr_mix - scr_rb  [margin_best]           -6.370  -6.260   +0.110      [-0.228, +0.442]  0.5485  nicht gezeigt
  scr_mix - scr_rb  [survived]               0.338   0.417   +0.079      [+0.017, +0.147]  0.0486  BESSER
  scr_mix - scr_rb  [suicides]               0.481   0.416   -0.065      [-0.136, +0.005]  0.1121  nicht gezeigt

## P3 INTERACTION -- did the warm start suppress the field effect?

  heldout    warm effect -0.043   scratch effect -0.042   difference +0.001
  indist     warm effect -0.119   scratch effect -0.106   difference +0.013
  guard      warm effect +0.009   scratch effect -0.443   difference -0.453

## Scratch vs warm, same field -- how much is the parent worth?

  comparison                                     A       B     diff                95% CI  perm p
  scr_rb - warm_rb  [score, heldout]         2.965   1.305   -1.661      [-1.831, -1.498]  0.0000  SCHLECHTER
  scr_rb - warm_rb  [score, indist]          3.342   1.406   -1.936      [-2.111, -1.739]  0.0001  SCHLECHTER
  scr_rb - warm_rb  [score, guard]           3.832   1.813   -2.019      [-2.210, -1.830]  0.0001  SCHLECHTER
