# Adversarial audit: kill-reward plateau Full1000

## Claim to challenge

The `+7.5` internal kill-reward candidate should replace the `+5` control after
the promising Quick100 result.

## Full1000 comparison

The candidate minus control on the same 1,000 External-Trio arenas is:

- Score: `+0.103 [-0.018,+0.226]`, sign-flip `p=0.1050`
- Coins: `+0.118 [+0.039,+0.198]`, sign-flip `p=0.0041`
- Kills: `-0.003 [-0.020,+0.014]`, sign-flip `p=0.8247`
- Suicides: `-0.037 [-0.080,+0.005]`, sign-flip `p=0.1015`
- Survival: `+0.050 [+0.008,+0.093]`, sign-flip `p=0.0262`

The Full1000 files contain 1,000 paired rounds per agent and the corresponding
metadata. The earlier p4 model hashes remain distinct and verified against the
metadata.

## Attempted refutation

The Quick100 score gain (`+0.460`) shrinks substantially at Full1000 and its CI
includes zero. The candidate's safety and coin gains are real in this field, but
the official primary metric is total score and the intended hypothesis was
better kill conversion. Kills are slightly lower, not higher. Neither the
plateau output nor the Quick100 can override this held-out result.

## Verdict

No promotion. The `+7.5` reward improves defensive/coin behavior in this
External-Trio measurement but does not demonstrate a score or kill advantage.
The Mixed-Kill-2000 submission incumbent remains active. The kill-reward pilot
is closed unless a genuinely new hypothesis and independent control are
defined.
