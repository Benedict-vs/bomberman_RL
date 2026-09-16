# Adversarial design audit: prioritized experience replay

## Isolated mechanism

`per_v1` samples replay transitions proportional to their current absolute TD
error (alpha 0.6), uses normalized importance weights (beta from 0.4 toward 1),
and updates the sampled priorities after every optimizer step. `per_control_v1`
uses the prior uniform sampler. Features, rewards, model, source checkpoint,
seed, opponents, target update and epsilon remain identical.

## Checks

Priorities are initialized to the current buffer maximum, so fresh experience
is not starved. Sampling is with replacement, as required by the probability
model. The bias-correction weights match the sampled probabilities. A unit test
checks priority updates and sampling; all 33 Task-4 tests pass.

## Gates

Quick100 requires non-negative score and kill direction and no suicide increase
over 0.03. Only then is a paired Full1000 warranted. No claim generalizes beyond
this PER parameterization and concrete source/seed/horizon.
