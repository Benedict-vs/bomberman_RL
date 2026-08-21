
========================================================================================================
### 3x rule_based (GUARD, primary for P1)
========================================================================================================
  cell means:  ctl 3.832   K 3.916   R 3.746   KR 3.845

  -- K vs ctl
     metric                         ctl     arm     diff                95% CI       p  verdict
     score                        3.832   3.916   +0.084      [-0.067, +0.230]  0.3195  nicht gezeigt
     coins                        2.740   2.793   +0.053      [-0.039, +0.144]  0.3135  nicht gezeigt
     kills                        0.218   0.225   +0.006      [-0.014, +0.026]  0.5626  nicht gezeigt
     crates                      32.982  32.460   -0.522      [-1.360, +0.283]  0.2598  nicht gezeigt
     coins_per_crate              0.083   0.086   +0.003      [+0.001, +0.005]  0.0231  BESSER
     suicides                     0.503   0.453   -0.050      [-0.109, +0.004]  0.1227  nicht gezeigt
     survived                     0.434   0.485   +0.051      [+0.001, +0.106]  0.0969  nicht gezeigt (fragile)
     invalid                      4.741   5.083   +0.342      [-0.086, +0.741]  0.1555  nicht gezeigt

  -- R vs ctl
     metric                         ctl     arm     diff                95% CI       p  verdict
     score                        3.832   3.746   -0.085      [-0.219, +0.044]  0.2548  nicht gezeigt
     coins                        2.740   2.627   -0.113      [-0.190, -0.029]  0.0243  SCHLECHTER
     kills                        0.218   0.224   +0.005      [-0.011, +0.020]  0.5582  nicht gezeigt
     crates                      32.982  32.162   -0.820      [-1.332, -0.292]  0.0120  SCHLECHTER
     coins_per_crate              0.083   0.082   -0.001      [-0.003, +0.000]  0.1401  nicht gezeigt
     suicides                     0.503   0.635   +0.132      [+0.061, +0.198]  0.0059  SCHLECHTER
     survived                     0.434   0.317   -0.117      [-0.183, -0.048]  0.0097  SCHLECHTER
     invalid                      4.741   5.193   +0.452      [+0.019, +0.861]  0.0716  nicht gezeigt (fragile)

  -- KR vs ctl
     metric                         ctl     arm     diff                95% CI       p  verdict
     score                        3.832   3.845   +0.013      [-0.140, +0.164]  0.8821  nicht gezeigt
     coins                        2.740   2.663   -0.077      [-0.152, +0.009]  0.1019  nicht gezeigt
     kills                        0.218   0.236   +0.018      [-0.002, +0.037]  0.1225  nicht gezeigt
     crates                      32.982  32.372   -0.610      [-1.148, -0.043]  0.0642  nicht gezeigt (fragile)
     coins_per_crate              0.083   0.082   -0.001      [-0.002, +0.001]  0.2958  nicht gezeigt
     suicides                     0.503   0.701   +0.198      [+0.120, +0.263]  0.0006  SCHLECHTER
     survived                     0.434   0.242   -0.192      [-0.258, -0.112]  0.0006  SCHLECHTER
     invalid                      4.741   5.615   +0.873      [+0.408, +1.334]  0.0050  SCHLECHTER

========================================================================================================
### 3x bindist_v2 (HELD OUT)
========================================================================================================
  cell means:  ctl 2.965   K 2.779   R 2.802   KR 2.764

  -- K vs ctl
     metric                         ctl     arm     diff                95% CI       p  verdict
     score                        2.965   2.779   -0.186      [-0.377, +0.004]  0.0998  nicht gezeigt
     coins                        2.078   2.002   -0.076      [-0.155, +0.002]  0.0978  nicht gezeigt
     kills                        0.177   0.155   -0.022      [-0.049, +0.005]  0.1584  nicht gezeigt
     crates                      39.096  37.201   -1.895      [-3.506, -0.340]  0.0648  nicht gezeigt (fragile)
     coins_per_crate              0.053   0.054   +0.001      [-0.001, +0.002]  0.4073  nicht gezeigt
     suicides                     0.557   0.537   -0.020      [-0.041, +0.002]  0.1159  nicht gezeigt
     survived                     0.268   0.303   +0.035      [+0.020, +0.049]  0.0019  BESSER
     invalid                      3.462   3.691   +0.229      [-0.200, +0.807]  0.5090  nicht gezeigt

  -- R vs ctl
     metric                         ctl     arm     diff                95% CI       p  verdict
     score                        2.965   2.802   -0.163      [-0.289, -0.043]  0.0249  SCHLECHTER
     coins                        2.078   2.044   -0.034      [-0.082, +0.016]  0.2266  nicht gezeigt
     kills                        0.177   0.152   -0.026      [-0.047, -0.004]  0.0476  SCHLECHTER
     crates                      39.096  39.995   +0.900      [+0.523, +1.221]  0.0008  BESSER
     coins_per_crate              0.053   0.051   -0.002      [-0.003, -0.001]  0.0083  SCHLECHTER
     suicides                     0.557   0.549   -0.007      [-0.033, +0.021]  0.6371  nicht gezeigt
     survived                     0.268   0.282   +0.014      [-0.011, +0.035]  0.3060  nicht gezeigt
     invalid                      3.462   5.101   +1.638      [+1.141, +2.170]  0.0001  SCHLECHTER

  -- KR vs ctl
     metric                         ctl     arm     diff                95% CI       p  verdict
     score                        2.965   2.764   -0.202      [-0.324, -0.089]  0.0042  SCHLECHTER
     coins                        2.078   2.039   -0.039      [-0.085, +0.009]  0.1606  nicht gezeigt
     kills                        0.177   0.145   -0.032      [-0.053, -0.011]  0.0124  SCHLECHTER
     crates                      39.096  40.020   +0.925      [+0.596, +1.213]  0.0006  BESSER
     coins_per_crate              0.053   0.051   -0.002      [-0.003, -0.001]  0.0057  SCHLECHTER
     suicides                     0.557   0.546   -0.011      [-0.028, +0.008]  0.2997  nicht gezeigt
     survived                     0.268   0.292   +0.024      [+0.010, +0.036]  0.0081  BESSER
     invalid                      3.462   4.491   +1.028      [+0.724, +1.385]  0.0001  SCHLECHTER

========================================================================================================
### P4 scale vs balance: KR (ratio held, magnitude doubled) vs ctl
========================================================================================================
  guard      K 3.916  R 3.746  KR 3.845  ctl 3.832   KR-ctl +0.013 (P4 bar |.|<0.35)   interaction +0.014
  heldout    K 2.779  R 2.802  KR 2.764  ctl 2.965   KR-ctl -0.202 (P4 bar |.|<0.35)   interaction +0.148
