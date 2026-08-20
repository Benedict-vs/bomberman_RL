
========================================================================================================
### field = 3x ext_xiaoxiae_binary_v6
========================================================================================================
  control: score 2.778  margin_mean -0.949  suicides 0.540  survived 0.302

  -- veto 5 vs control
     metric              ctl      arm      diff                95% CI        p   verdict
     score             2.778    2.748    -0.029      [-0.061, +0.002]   0.0624   nicht gezeigt
     margin_mean      -0.949   -1.016    -0.067      [-0.108, -0.025]   0.0014   SCHLECHTER
     coins             2.174    2.150    -0.025      [-0.039, -0.010]   0.0010   SCHLECHTER
     kills             0.121    0.120    -0.001      [-0.006, +0.004]   0.7593   nicht gezeigt
     suicides          0.540    0.536    -0.003      [-0.011, +0.004]   0.4341   nicht gezeigt
     survived          0.302    0.290    -0.012      [-0.019, -0.004]   0.0015   SCHLECHTER
     crates           39.119   38.768    -0.351      [-0.535, -0.171]   0.0001   SCHLECHTER
     bombs            21.590   21.193    -0.398      [-0.595, -0.202]   0.0000   SCHLECHTER

  -- veto 4 vs control
     metric              ctl      arm      diff                95% CI        p   verdict
     score             2.778    2.495    -0.283      [-0.349, -0.212]   0.0000   SCHLECHTER
     margin_mean      -0.949   -1.308    -0.359      [-0.453, -0.265]   0.0000   SCHLECHTER
     coins             2.174    1.942    -0.233      [-0.271, -0.193]   0.0000   SCHLECHTER
     kills             0.121    0.111    -0.010      [-0.020, +0.000]   0.0604   nicht gezeigt
     suicides          0.540    0.409    -0.131      [-0.147, -0.115]   0.0000   BESSER
     survived          0.302    0.352    +0.050      [+0.035, +0.065]   0.0000   BESSER
     crates           39.119   34.707    -4.412      [-4.954, -3.857]   0.0000   SCHLECHTER
     bombs            21.590   19.183    -2.407      [-2.874, -1.921]   0.0000   SCHLECHTER

  -- veto 3 vs control
     metric              ctl      arm      diff                95% CI        p   verdict
     score             2.778    0.523    -2.255      [-2.334, -2.179]   0.0000   SCHLECHTER
     margin_mean      -0.949   -3.608    -2.659      [-2.777, -2.544]   0.0000   SCHLECHTER
     coins             2.174    0.453    -1.722      [-1.766, -1.677]   0.0000   SCHLECHTER
     kills             0.121    0.014    -0.107      [-0.118, -0.096]   0.0000   SCHLECHTER
     suicides          0.540    0.117    -0.422      [-0.441, -0.404]   0.0000   BESSER
     survived          0.302    0.509    +0.207      [+0.186, +0.228]   0.0000   BESSER
     crates           39.119    5.822   -33.297    [-33.815, -32.780]   0.0000   SCHLECHTER
     bombs            21.590    3.815   -17.776    [-18.253, -17.291]   0.0000   SCHLECHTER

========================================================================================================
### field = 3x rule_based_agent
========================================================================================================
  control: score 3.944  margin_mean 0.945  suicides 0.485  survived 0.450

  -- veto 5 vs control
     metric              ctl      arm      diff                95% CI        p   verdict
     score             3.944    3.961    +0.017      [-0.097, +0.130]   0.7770   nicht gezeigt
     margin_mean       0.945    0.947    +0.002      [-0.137, +0.140]   0.9779   nicht gezeigt
     coins             2.822    2.821    -0.001      [-0.048, +0.048]   0.9741   nicht gezeigt
     kills             0.225    0.228    +0.004      [-0.016, +0.023]   0.7481   nicht gezeigt
     suicides          0.485    0.480    -0.005      [-0.026, +0.016]   0.6863   nicht gezeigt
     survived          0.450    0.455    +0.004      [-0.017, +0.025]   0.7174   nicht gezeigt
     crates           33.477   33.562    +0.086      [-0.249, +0.409]   0.6169   nicht gezeigt
     bombs            28.534   28.921    +0.388      [-0.220, +1.002]   0.2177   nicht gezeigt

  -- veto 4 vs control
     metric              ctl      arm      diff                95% CI        p   verdict
     score             3.944    3.788    -0.156      [-0.271, -0.042]   0.0092   SCHLECHTER
     margin_mean       0.945    0.713    -0.232      [-0.373, -0.092]   0.0016   SCHLECHTER
     coins             2.822    2.660    -0.162      [-0.214, -0.112]   0.0000   SCHLECHTER
     kills             0.225    0.226    +0.001      [-0.018, +0.021]   0.9186   nicht gezeigt
     suicides          0.485    0.463    -0.023      [-0.043, -0.001]   0.0413   BESSER
     survived          0.450    0.467    +0.017      [-0.005, +0.037]   0.1312   nicht gezeigt
     crates           33.477   31.367    -2.110      [-2.484, -1.726]   0.0000   SCHLECHTER
     bombs            28.534   27.719    -0.814      [-1.425, -0.206]   0.0098   SCHLECHTER

  -- veto 3 vs control
     metric              ctl      arm      diff                95% CI        p   verdict
     score             3.944    1.426    -2.518      [-2.615, -2.421]   0.0000   SCHLECHTER
     margin_mean       0.945   -2.410    -3.355      [-3.478, -3.232]   0.0000   SCHLECHTER
     coins             2.822    1.114    -1.708      [-1.759, -1.657]   0.0000   SCHLECHTER
     kills             0.225    0.062    -0.162      [-0.178, -0.146]   0.0000   SCHLECHTER
     suicides          0.485    0.251    -0.234      [-0.256, -0.213]   0.0000   BESSER
     survived          0.450    0.464    +0.013      [-0.009, +0.035]   0.2455   nicht gezeigt
     crates           33.477    6.543   -26.933    [-27.227, -26.633]   0.0000   SCHLECHTER
     bombs            28.534    8.959   -19.575    [-20.057, -19.075]   0.0000   SCHLECHTER
