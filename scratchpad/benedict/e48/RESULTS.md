
========================================================================================================
### 3x rule_based (GUARD, primary for P1)
========================================================================================================
  cell means:  ctl 3.832   C 2.042   D 3.929   CD 2.273

  -- C vs ctl
     metric                         ctl     arm     diff                95% CI       p  verdict
     score                        3.832   2.042   -1.789      [-2.061, -1.501]  0.0001  SCHLECHTER
     coins                        2.740   1.407   -1.333      [-1.509, -1.152]  0.0000  SCHLECHTER
     kills                        0.218   0.127   -0.091      [-0.117, -0.067]  0.0001  SCHLECHTER
     crates                      32.982  15.247  -17.735    [-19.725, -15.673]  0.0001  SCHLECHTER
     coins_per_crate              0.083   0.093   +0.010      [+0.004, +0.017]  0.0109  BESSER
     suicides                     0.503   0.325   -0.178      [-0.239, -0.118]  0.0001  BESSER
     survived                     0.434   0.595   +0.160      [+0.105, +0.216]  0.0004  BESSER
     invalid                      4.741   3.967   -0.774      [-1.249, -0.317]  0.0076  BESSER

  -- D vs ctl
     metric                         ctl     arm     diff                95% CI       p  verdict
     score                        3.832   3.929   +0.097      [-0.030, +0.221]  0.1819  nicht gezeigt
     coins                        2.740   2.790   +0.050      [-0.031, +0.131]  0.2904  nicht gezeigt
     kills                        0.218   0.228   +0.010      [-0.008, +0.027]  0.3529  nicht gezeigt
     crates                      32.982  32.947   -0.035      [-0.610, +0.562]  0.9121  nicht gezeigt
     coins_per_crate              0.083   0.085   +0.002      [+0.000, +0.003]  0.0486  BESSER
     suicides                     0.503   0.513   +0.010      [-0.049, +0.068]  0.7776  nicht gezeigt
     survived                     0.434   0.415   -0.020      [-0.075, +0.035]  0.5299  nicht gezeigt
     invalid                      4.741   4.766   +0.025      [-0.439, +0.450]  0.9223  nicht gezeigt

  -- CD vs ctl
     metric                         ctl     arm     diff                95% CI       p  verdict
     score                        3.832   2.273   -1.558      [-1.785, -1.303]  0.0000  SCHLECHTER
     coins                        2.740   1.575   -1.165      [-1.323, -1.000]  0.0001  SCHLECHTER
     kills                        0.218   0.140   -0.079      [-0.103, -0.055]  0.0001  SCHLECHTER
     crates                      32.982  16.747  -16.235    [-17.539, -14.951]  0.0001  SCHLECHTER
     coins_per_crate              0.083   0.094   +0.011      [+0.004, +0.018]  0.0072  BESSER
     suicides                     0.503   0.354   -0.150      [-0.200, -0.102]  0.0001  BESSER
     survived                     0.434   0.556   +0.122      [+0.077, +0.171]  0.0005  BESSER
     invalid                      4.741   3.829   -0.912      [-1.393, -0.461]  0.0022  BESSER

========================================================================================================
### 3x bindist_v2 (HELD OUT)
========================================================================================================
  cell means:  ctl 2.965   C 0.738   D 2.966   CD 0.804

  -- C vs ctl
     metric                         ctl     arm     diff                95% CI       p  verdict
     score                        2.965   0.738   -2.228      [-2.423, -2.040]  0.0001  SCHLECHTER
     coins                        2.078   0.563   -1.515      [-1.649, -1.388]  0.0001  SCHLECHTER
     kills                        0.177   0.035   -0.142      [-0.163, -0.122]  0.0001  SCHLECHTER
     crates                      39.096  10.388  -28.708    [-31.060, -26.611]  0.0001  SCHLECHTER
     coins_per_crate              0.053   0.055   +0.001      [-0.002, +0.004]  0.4703  nicht gezeigt
     suicides                     0.557   0.257   -0.300      [-0.349, -0.254]  0.0001  BESSER
     survived                     0.268   0.545   +0.277      [+0.209, +0.349]  0.0001  BESSER
     invalid                      3.462   2.343   -1.119      [-1.740, -0.442]  0.0081  BESSER

  -- D vs ctl
     metric                         ctl     arm     diff                95% CI       p  verdict
     score                        2.965   2.966   +0.000      [-0.162, +0.155]  0.9989  nicht gezeigt
     coins                        2.078   2.095   +0.017      [-0.049, +0.080]  0.6286  nicht gezeigt
     kills                        0.177   0.174   -0.003      [-0.027, +0.021]  0.8177  nicht gezeigt
     crates                      39.096  38.597   -0.498      [-1.002, -0.014]  0.0843  nicht gezeigt (fragile)
     coins_per_crate              0.053   0.054   +0.001      [-0.000, +0.002]  0.1585  nicht gezeigt
     suicides                     0.557   0.565   +0.008      [-0.014, +0.031]  0.5471  nicht gezeigt
     survived                     0.268   0.268   -0.000      [-0.019, +0.017]  0.9496  nicht gezeigt
     invalid                      3.462   3.361   -0.102      [-0.508, +0.354]  0.6764  nicht gezeigt

  -- CD vs ctl
     metric                         ctl     arm     diff                95% CI       p  verdict
     score                        2.965   0.804   -2.162      [-2.310, -2.015]  0.0001  SCHLECHTER
     coins                        2.078   0.633   -1.445      [-1.541, -1.343]  0.0001  SCHLECHTER
     kills                        0.177   0.034   -0.143      [-0.162, -0.125]  0.0001  SCHLECHTER
     crates                      39.096  11.230  -27.866    [-29.723, -25.938]  0.0000  SCHLECHTER
     coins_per_crate              0.053   0.057   +0.004      [+0.000, +0.008]  0.0784  nicht gezeigt (fragile)
     suicides                     0.557   0.237   -0.320      [-0.367, -0.265]  0.0001  BESSER
     survived                     0.268   0.530   +0.261      [+0.200, +0.313]  0.0001  BESSER
     invalid                      3.462   2.672   -0.791      [-1.637, +0.078]  0.1077  nicht gezeigt

========================================================================================================
### P4 interaction: (CD - C) - (D - ctl), on score
========================================================================================================
  guard      C 2.042  D 3.929  CD 2.273  ctl 3.832   interaction +0.133
  heldout    C 0.738  D 2.966  CD 0.804  ctl 2.965   interaction +0.065
