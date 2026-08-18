================================================================================================
E41 -- the shipped table against four third-party agents.  n=1000, ship seed 990731
================================================================================================

## P2 -- the calibration constant: THEIR agent in OUR slot vs 3x rule_based
   (our shipped table in that slot scores 3.949)

  agent                                score   vs ours   coins   kills   suic  crates  inval med thinkmax over%
  ext_xiaoxiae_binary_v6               5.572    +1.623   3.372   0.440  0.377   25.09        2.0    155.7 0.00%
  ext_xiaoxiae_bindist_v2              5.336    +1.387   3.216   0.424  0.276   25.10        2.0    120.3 0.00%
  ext_aielka_ql_atom                   4.690    +0.741   2.770   0.384  0.127   30.40        3.0     32.1 0.00%
  ext_lijesse_featureeverything        5.143    +1.194   3.728   0.283  0.108   39.72        5.0     51.1 0.00%

## P1 -- head-to-head: ours + 3x external.  PRIMARY = margin_mean

  ext_xiaoxiae_binary_v6   (our score 2.611, their mean 3.619)
  margin_mean               -1.008 [-1.197, -0.813]  t=-10.41 p=0.0000  SCHLECHTER
  margin_best               -4.263 [-4.540, -3.978]  t=-30.38 p=0.0000  SCHLECHTER
    P3 crates/bomb 1.797  (guard 1.16 +/- 0.15)  *** OUT OF BAND -- P3 REFUTED ***
    P4 opp deaths 1.819 - opp suicides 1.463 = takeable 0.356   (rule_based 0.333; reopen bar 0.5)  no
       our kills 0.091, share of takeable 25.6%

  ext_xiaoxiae_bindist_v2   (our score 2.986, their mean 3.783)
  margin_mean               -0.797 [-1.012, -0.570]  t= -7.19 p=0.0000  SCHLECHTER
  margin_best               -4.308 [-4.607, -3.991]  t=-27.96 p=0.0000  SCHLECHTER
    P3 crates/bomb 1.960  (guard 1.16 +/- 0.15)  *** OUT OF BAND -- P3 REFUTED ***
    P4 opp deaths 1.675 - opp suicides 1.220 = takeable 0.455   (rule_based 0.333; reopen bar 0.5)  no
       our kills 0.173, share of takeable 38.0%

  ext_aielka_ql_atom   (our score 4.283, their mean 3.371)
  margin_mean               +0.912 [+0.661, +1.165]  t= +7.06 p=0.0000  BESSER
  margin_best               -1.872 [-2.170, -1.566]  t=-12.10 p=0.0000  SCHLECHTER
    P3 crates/bomb 1.429  (guard 1.16 +/- 0.15)  *** OUT OF BAND -- P3 REFUTED ***
    P4 opp deaths 1.717 - opp suicides 1.049 = takeable 0.668   (rule_based 0.333; reopen bar 0.5)  *** FIRES ***
       our kills 0.301, share of takeable 45.1%

  ext_lijesse_featureeverything   (our score 2.226, their mean 2.770)
  margin_mean               -0.544 [-0.684, -0.399]  t= -7.65 p=0.0000  SCHLECHTER
  margin_best               -2.284 [-2.464, -2.095]  t=-24.74 p=0.0000  SCHLECHTER
    P3 crates/bomb 1.249  (guard 1.16 +/- 0.15)  IN BAND
    P4 opp deaths 0.400 - opp suicides 0.239 = takeable 0.161   (rule_based 0.333; reopen bar 0.5)  no
       our kills 0.092, share of takeable 57.1%

## The symmetric bar -- 4x external, no us. Prices the field itself.
  (4x rule_based gives won 0.282; our shipped table reaches won 0.406)

  ext_xiaoxiae_binary_v6             mean score  3.377   best-of-four  7.744   survival 0.432
  ext_xiaoxiae_bindist_v2            mean score  3.428   best-of-four  7.678   survival 0.467
  ext_aielka_ql_atom                 mean score  3.089   best-of-four  6.550   survival 0.550
  ext_lijesse_featureeverything       (pending)
