# Adversarial design audit: auxiliary opponent prediction

`opponent_prediction_v1` adds only an auxiliary head that predicts the opponent
occupancy map of the stored next state from the current shared CNN features.
The Q head is copied exactly from the Mixed-Kill-2000 model; the auxiliary head
is new. The Q loss remains unchanged and receives an added weighted binary
cross-entropy auxiliary loss (scale 0.1, positive weight 32). No auxiliary head
is used during inference and no action is prescribed.

Control and candidate run 2,000 episodes from the same Q policy in the Mixed
field, with equal seed, rewards, replay, optimizer, target update and epsilon.
Quick100 requires non-negative score/kill direction and suicide increase no
greater than 0.03; Full1000 only follows a passed gate. Conversion Q equality,
occupancy-logit shape, shell/config imports and the 35-test suite pass.
