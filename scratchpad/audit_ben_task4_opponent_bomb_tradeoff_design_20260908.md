# Adversarial design audit: opponent-bomb tradeoff pilot (2026-09-08)

## Claim under test

The `opponent_bomb_tradeoff_v1` arm may improve Task 4 combat decisions by exposing
state information about bomb sources that hit an opponent while retaining an escape
option. `opponent_bomb_tradeoff_zero_v1` is the 12-channel inert control.

## Attempt to break the claim

- **Is this a hidden action prescription?** No. The channel is a board-valued state
  feature. It does not select an action, alter the action mask, or inject a reward.
  It marks hypothetical free bomb-source tiles and combines two state-derived
  quantities: reachable own escape tiles and opponent escape pressure.
- **Are the two arms actually comparable?** Yes by configuration: both load the same
  12-channel source model, use seed 11, 1,000 training episodes, and the same mixed
  opponent field (`peaceful_agent, rule_based_agent, rule_based_agent`). The control
  computes the same channel allocation but fills it with zero; the candidate enables
  only the new channel.
- **Could geometry leak the best action?** The feature marks the hypothetical bomb
  source, not the direction or action to take. The network must learn how to use the
  spatial pattern; no action label is derived from it.
- **Could it be too slow for the tournament?** On a synthetic 17x17 Task-4 board,
  200 warmed-up calls measured about 1.15 ms/call with the feature enabled and
  0.014 ms/call for the zero control on the local CPU. This is far below the 500 ms
  step budget, though the actual submitted 11-channel candidate remains the safer
  fallback and the final candidate must still be measured with `think_max_ms`.
- **Could the apparent mechanism be invented after the result?** Yes, if the pilot
  is interpreted causally without a pre-run prediction. This is therefore a
  hypothesis test only. The verdict must use paired Quick100 gates, then an
  independent adversarial audit and Full1000 measurement if the gate passes.

## Verdict

The pilot is sufficiently isolated and cheap to run. It is allowed to proceed, but
it is not evidence of improvement until both arms have completed and are compared at
epsilon zero. A score or kill improvement that is fragile, fails the paired CI rule,
or increases suicides by at least 0.03 is rejected.

