====================================================================================================
E42 -- training against a field that hunts back.  8 seeds/arm, 300 rounds, val seed 550731
====================================================================================================

### HELD OUT: 3x bindist_v2 (never seen in training)

  metric               control     mixed  difference                95% CI   perm p   verdict
  --------------------------------------------------------------------------------------------
  score                  2.965     2.850      -0.115      [-0.243, +0.007]   0.1195   nicht gezeigt
  margin_mean           -1.000    -1.043      -0.043      [-0.174, +0.082]   0.5682   nicht gezeigt
  margin_best           -4.558    -4.582      -0.024      [-0.191, +0.146]   0.8102   nicht gezeigt
  coins                  2.078     2.065      -0.013      [-0.063, +0.038]   0.6252   nicht gezeigt
  kills                  0.177     0.157      -0.020      [-0.042, +0.002]   0.1161   nicht gezeigt
  suicides               0.557     0.568      +0.012      [-0.016, +0.040]   0.4671   nicht gezeigt
  killed_by_opponent     0.175     0.164      -0.011      [-0.027, +0.003]   0.2031   nicht gezeigt
  survived               0.268     0.268      -0.000      [-0.027, +0.025]   0.9677   nicht gezeigt
  crates_per_bomb        2.214     2.053      -0.160      [-0.185, -0.137]   0.0000   SCHLECHTER

### in-distribution: the training field

  metric               control     mixed  difference                95% CI   perm p   verdict
  --------------------------------------------------------------------------------------------
  score                  3.342     3.181      -0.161      [-0.305, -0.014]   0.0644   nicht gezeigt
  margin_mean           -0.045    -0.164      -0.119      [-0.292, +0.058]   0.2354   nicht gezeigt
  margin_best           -2.935    -2.953      -0.018      [-0.211, +0.176]   0.8672   nicht gezeigt
  coins                  2.371     2.298      -0.073      [-0.132, -0.017]   0.0389   SCHLECHTER
  kills                  0.194     0.177      -0.017      [-0.043, +0.010]   0.2410   nicht gezeigt
  suicides               0.446     0.471      +0.025      [-0.005, +0.052]   0.1403   nicht gezeigt
  killed_by_opponent     0.199     0.187      -0.012      [-0.032, +0.010]   0.3276   nicht gezeigt
  survived               0.355     0.342      -0.013      [-0.030, +0.003]   0.1770   nicht gezeigt
  crates_per_bomb        1.825     1.706      -0.120      [-0.165, -0.078]   0.0001   SCHLECHTER

### regression guard: 3x rule_based

  metric               control     mixed  difference                95% CI   perm p   verdict
  --------------------------------------------------------------------------------------------
  score                  3.832     3.782      -0.049      [-0.182, +0.075]   0.4948   nicht gezeigt
  margin_mean            0.795     0.804      +0.009      [-0.152, +0.169]   0.9171   nicht gezeigt
  margin_best           -1.636    -1.532      +0.105      [-0.081, +0.290]   0.3261   nicht gezeigt
  coins                  2.740     2.693      -0.047      [-0.122, +0.030]   0.2782   nicht gezeigt
  kills                  0.218     0.218      -0.000      [-0.021, +0.019]   0.9991   nicht gezeigt
  suicides               0.503     0.687      +0.183      [+0.126, +0.236]   0.0002   SCHLECHTER
  killed_by_opponent     0.062     0.049      -0.014      [-0.024, -0.004]   0.0256   BESSER
  survived               0.434     0.265      -0.170      [-0.218, -0.117]   0.0001   SCHLECHTER
  crates_per_bomb        1.445     1.401      -0.045      [-0.090, +0.001]   0.0908   nicht gezeigt
