# Adversarial design audit: event-balanced replay pilot

## Claim to challenge

Oversampling rare event transitions will improve kill conversion without merely
changing the reward, features or inference policy.

## Isolated change

- Control: `event_replay_control_v1` uses the existing uniform replay sampler.
- Candidate: `event_replay_balanced_v1` samples approximately 20% transitions
  tagged with `KILLED_OPPONENT` and 15% tagged with `GOT_KILLED`; the remainder
  is uniform.
- Event tags are attached before n-step aggregation and unioned across each
  n-step window. They survive terminal conversion and symmetry augmentation.
- Both arms start from the same Mixed-Kill-2000 model, use the same Mixed field,
  seed 11, rewards, features, architecture and 1,000 training episodes.
- Official evaluation remains greedy External Trio Quick100 first, with live
  progress. Full1000 is allowed only if the Quick100 is directionally positive.

## Risks and gates

Oversampling deaths may teach avoidance at the expense of offense; a kill-only
improvement is insufficient if score falls or suicides rise. A positive result
requires higher score, no kill loss, no suicide regression above `0.03`, no
fragile analysis row, and Full1000 confirmation. If the candidate is safer but
not higher-scoring, it remains a research artifact rather than a replacement.

## Audit verdict

The pilot is causally interpretable and does not duplicate the rejected
kill-reward experiment. It is approved for a single control/candidate Quick100
screen. Codex must not start either training.
