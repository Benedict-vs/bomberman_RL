# `scratchpad/strategy/` — session of 2026-08-17, strategy review after E38

Analysis-only session: no training run, nothing under `agent_code/` written, nothing committed.
**Start a fresh session here:** read `NEXT_STEPS.md` first, then this index for provenance.

## The three documents

| file | what is in it |
|---|---|
| **`NEXT_STEPS.md`** | **the ranked plan — read this first.** Two corrections to project conventions found in `final_project.pdf`, the four-field robustness table, and the ranked to-do / do-not-do lists |
| `TASK_A_survey_vs_ours.md` | `scratchpad/survey/REPORT.md` mined against our feature map. Why the "191× the state space vs E's 335" premise is false, what E encodes that we do not, what the corpus tried and failed, contradictions with `experiments/benedict_task4.md` |
| `TASK_B_argument.md` | the strategy argument. §1–5 the phase model and the hunt supply; §6 Benedict's hunt hypothesis tested; §7 the hunt ceiling measured at n = 4000 |

## Headline results, each with the probe that produced it

| finding | number | probe |
|---|---|---|
| our *effective* state count is 290, not 64 000 — E's 335 is not 191× smaller | 1 407 rows visited, exp-entropy 289.5 | `state_visits.py` |
| the round's whole economy closes by step ~200; 94 % of our score exists then | 9 coins gone by ~220, 122 crates by ~250 | `round_economy.py` |
| every "survival" intervention bought **phase-2** survival, which is worth ~nothing | E36: P1 +0.043, P2 +0.245, score +0.007 | `phase_split.py` |
| `won` is a linear readout of score | +0.088 [+0.076, +0.101] per point | `survival_value.py`, `won_anatomy.py` |
| the crate pool is exactly zero-sum and fully consumed | ours 33.41 + theirs 89.27 = 122.68 of 122 | `denial.py` |
| digit 6 cannot say *what* it points at | opponent 39.1 % / crate 36.6 % / coin 21.4 % | `target_type.py` |
| "is BOMB safe here" would be a constant for us | an escape exists on 99.08 % of armed steps | `trap_opportunity.py` |
| splitting digit 7 crate/opponent is 98.3 % noise | in blast range 25.3 % of armed steps, *trapped* 0.42 % | `trap_opportunity.py` |
| crates **do** make opponents trappable (Benedict's mechanism, confirmed) | 27.2 % trappable at 30–59 crates vs 16.7 % at 0 | `hunt_window.py` |
| …but trap information has a ~3-step horizon | still lethal on arrival: 16 % at walk 1–2, 4.0 % at 5–8 | `trap_persistence.py` |
| **the hunt ceiling is real and tiny** | k=4: score **+0.116 [+0.002, +0.233]**, won +0.037, kills +0.032, crates −0.422 | `hunt_ceiling.py` |
| ~45 % of our bombs clear zero crates, but the table already grades by crate count | P(BOMB) 0.16/0.36/0.56/0.64 for 0/1/2/3 crates | `bomb_siting.py` |
| the between-seed MDE table for any pre-registration | score 0.254 at n = 15, ≈0.11 with a collapse screen | `mde.py` |
| kills scale inversely with the field's evasion, suicides with its aggression | kills 1.55→0.73→0.47→0.23; suicides 0.087→0.160→0.403→0.488 | `scratchpad/strategy/fields/` |

## Data written

- `ceil/` — hunt-ceiling evaluations. `huntceil_k*` are the n = 1000 first pass (k = −1, 0, 2, 4, 8);
  `huntceil4k_k*` are the n = 4000 rerun of k = −1, 0, 4 that carries the verdict. Same CSV schema as
  `tools/evaluate.py` minus the think-time columns, so `analyze.py --compare` needs an explicit
  `--metrics` list.
- `fields/` — the shipped table against `peaceful` / `coin_collector` / `mixed`, 300 rounds each at
  seed 990731. Written by `tools/evaluate.py` itself, so these are fully standard.
- `pilot/` — throwaway 100-round validation of the ceiling harness. Safe to delete.

Reproduce the verdict:

```bash
uv run python scratchpad/strategy/hunt_ceiling.py --k -1 --n-rounds 4000 --seed 990731 \
    --label huntceil4k_k-1__task4_rb_ship990731 --out-dir scratchpad/strategy/ceil
uv run python tools/analyze.py --compare \
    scratchpad/strategy/ceil/huntceil4k_k-1__task4_rb_ship990731.csv \
    scratchpad/strategy/ceil/huntceil4k_k4__task4_rb_ship990731.csv \
    --metrics score won kills coins crates suicides survived --markdown
```

## Not done, and deliberately

- Nothing was committed, staged, or added to `.gitignore`; `experiments/*.md`, `AGENTS.md`,
  `MEASUREMENT.md` and `tools/` are untouched. **Two of them are wrong** and the corrections are
  written up in `NEXT_STEPS.md` §0 for you to apply: `AGENTS.md`'s "`won` matters more than mean
  score" and the same claim as fact in `tools/evaluate.py:264-267`, against `final_project.pdf` §3
  ("a winner by total score"); and `benedict_task4.md` §6's `won ≈ 0.113 × score`, which is a ratio
  of means where the marginal is 0.088.
- No experiment ledger entry was written. The hunt ceiling is a complete, pre-registerable negative
  result and belongs in `experiments/benedict.md` as its own entry.
- Nobody has audited any of this. Every load-bearing claim above is one session's work, and this
  project's record is that seven of eight audits overturned something.
