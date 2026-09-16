# Adversarial design audit: kill-reward plateau pilot

## Audit question

Try to find a design or interpretation error that would make the planned
`kill_reward_control_*` versus `kill_reward75_*` pilot unable to answer whether
additional training improves the agent.

## Planned comparison

- Both arms start from the identical audited
  `ben_task4_mixed_kill_v1_2000ep_seed11.pt` checkpoint.
- Four sequential phases of 250 episodes are used, with the previous phase's
  end model as the next phase's input. The total is therefore 1,000 additional
  episodes, not a hidden fresh 1,000-episode run.
- Control uses the existing internal kill reward `+5`; candidate uses only the
  training reward `+7.5` for `KILLED_OPPONENT`. Official evaluation still scores
  every kill as `+5`.
- Both arms alternate the same fields by phase: Mixed on odd phases and the
  External Trio on even phases. The only intended reward difference is the kill
  reward.
- After every phase, the resulting model is greedily evaluated for 100 rounds
  against the fixed External Trio with visible progress. The plateau checker
  consumes those evaluation CSVs, never the training score or loss.

## Findings and safeguards

1. The control is necessary because a later checkpoint can improve merely from
   more updates, independent of the reward change.
2. Alternating fields are a deliberate compromise: they test transfer to the
   difficult External Trio while retaining the incumbent's Mixed training
   distribution. The result must not be generalized to all opponents or seeds.
3. Four 250-episode phases give the checker enough history for its default
   two-delta patience rule. Before phase 3 it reports insufficient history.
4. `BM_PLATEAU_STOP=1` may stop after a detected plateau; the default is
   monitor-only (`0`) so an accidental environment variable cannot silently
   shorten a run.
5. A positive plateau result is not a promotion. Promotion requires the
   candidate to beat its control on score and kills in the same held-out field,
   avoid a suicide increase above `0.03`, pass the non-fragility rule, and then
   receive a Full1000 confirmation. A plateau with safety regression is a
   rejection signal, not a reason to train longer.

## Audit verdict

The pilot is causally interpretable for this concrete checkpoint, schedule and
seed. It does not establish a general kill-reward optimum, and it cannot turn a
training-curve plateau into evidence of a tournament improvement. The scripts
are approved for Ben to start manually; Codex must not start them.
