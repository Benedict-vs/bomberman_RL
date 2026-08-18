# E40 (design, pre-agents) — the shipped table against a field that is not `rule_based_agent`

**Status: design only. Written 2026-08-17, before any external agent has been obtained**, so that
the decision rule cannot be chosen after seeing the result. Nothing here has been run. The ledger
entry goes into `experiments/benedict.md` when the line-up is fixed; this file is the part that must
be settled first.

---

## 1 · Why

All **412** committed rung-4 evaluation CSVs are against 3 × `rule_based_agent`
(`ls results/eval/task4_tournament/*.csv | grep -v _rb_` returns nothing — verified). Every
strategic conclusion this project holds is conditional on one opponent's behaviour:

| conclusion | the `rule_based` behaviour it rests on |
|---|---|
| the economy closes at step ~200 | it clears crates fast — opponents take **89.3 of 122** |
| traps decay to 4 % over a 5-step walk (E39) | it **flees actively**; a trap's half-life is ~1 move |
| the kill pool is not free | it suicides **1.50×/round** of 1.82 deaths |
| `won` = 0.088 × score | the margin distribution against *the best of three* `rule_based` |

A learned student agent plausibly breaks all four. The course explicitly sanctions testing against
other teams' agents (`final_project.pdf` p. 2, `#final-project-beat-my-agent`).

**Mechanically ready today.** `tools/evaluate.py --agents A B C D` with no `--opponents` accepts
arbitrary agent folder names; `agents.py:204` imports `agent_code.<name>.callbacks` and
`agents.py:305` chdirs into that folder per round. No change to `tools/` is required.

---

## 2 · The primary metric — and it is **not** our absolute score

Our absolute score is nearly meaningless across fields. Same table, no retraining, 300 rounds each
(`scratchpad/strategy/fields/`, reproduced 2026-08-17):

| field | our score | best opponent in that field |
|---|---|---|
| 3 × `peaceful` | 14.49 | 0.06 |
| 3 × `coin_collector` | 6.03 | 4.18 |
| mixed | 5.80 | **7.41** |
| 3 × `rule_based` | 3.95 | 5.32 |

A 3.7× swing driven entirely by who else is on the board. So "our score went up against team X"
proves nothing: the crate pool is zero-sum and fully consumed (ours 33.41 + theirs 89.27 = 122.68
of 122), and *any* field that clears crates worse than `rule_based` hands us score for free. **A
prediction that our score rises against weaker crate-clearers is close to unfalsifiable and must not
be the pre-registration.**

**Primary: the paired within-round margin `score_ours − score_theirs`,** computed per round on
identical arenas, bootstrapped. That is the quantity a total-score tournament actually sums, and it
is invariant to the field's overall difficulty in a way our absolute score is not.

Secondaries: absolute `score`, `kills`, `suicides`, `killed_by`, `crates`, `crates/bomb`,
`survived`, `think_max_ms`. `won`/`rank` reported but **never primary** (`AGENTS.md`; MDE 0.029
against +0.088/point).

---

## 3 · Line-ups, in priority order

1000 rounds each at the held-out ship seed **990731**. `--out-dir results/eval/task4_tournament`,
label `benedict_task4_shipped_e37__task4_ext<NAME>_ship990731`.

| # | line-up | what only this one answers |
|---|---|---|
| 1 | ours + 3 × external | our margin against that team, cleanly |
| 2 | 4 × external | **the symmetric bar** — prices the field itself, exactly as `benedict_task4.md` §1 does with 4 × `rule_based` (`won` 0.282). Without it we cannot tell a strong opponent from an easy arena |
| 3 | external + 3 × `rule_based` | **the calibration constant.** Puts *their* agent in *our* slot against *our* reference field. This is what converts source E's bare "5.04" into a number comparable to our 3.949 |
| 4 | ours + 1 external + 2 × `rule_based` | the mixed case closest to an actual tournament round |

Run 1–3 for the first agent obtained; 4 only if two or more agents arrive.

---

## 4 · Pre-registered decision rule — write nothing after seeing the numbers

1. **P1 — external validity of the shipped agent.** Paired margin `ours − theirs` is **> 0** with a
   CI excluding 0 in line-up 1. *Refutation:* a margin ≤ 0 against a student agent means the
   tournament standing is materially worse than 412 CSVs suggest, and the remaining weeks go to
   robustness rather than to any feature on the `NEXT_STEPS.md` list.
2. **P2 — is the agent `rule_based`-overfitted?** Our **crates/bomb** against the external field is
   within ±0.15 of the 1.16 measured against `rule_based`. *Refutation:* a drop means the bomb-siting
   policy E37 bought is tuned to one opponent's movement, which would be the most consequential
   negative available and would reopen E37's mechanism.
3. **P3 — the hunt question, reopened conditionally.** E39 closed hunting at an oracle ceiling of
   +0.116 [+0.002, +0.233] **against `rule_based`, which flees actively and is the hardest possible
   evader.** If the external field's `survived` under our bombs is materially higher than
   `rule_based`'s, rerun `hunt_ceiling.py` k = 4 against *that* field. **Reopen the hunting
   direction only if the oracle clears +0.25 there** — the same bar E39 pre-registered, unchanged.
4. **P4 — kills.** Our kills against the external field exceed the 0.226 against `rule_based`.
   Directional only; `scratchpad/strategy/fields/` predicts it from evasion quality alone.
5. **Guard — `think_max_ms` < 500** for *every* agent in the line-up, ours included.

**What changes as a result, stated in advance.** If P2 fails → the next sweep is a retrain against a
mixed field, not a feature. If P2 holds and P3 does not reopen → the next feature is **target type**
(`NEXT_STEPS.md` §3.1), unchanged. If P1 fails → stop feature work and diagnose.

---

## 5 · Validity threats that must be checked *before* any score is read

- **A slow external agent plays crippled, and that is not a neutral distortion.** `TIMEOUT = 0.5 s`
  → `WAIT`, with the overrun billed to its next step. An agent that times out is *artificially
  weak*, so our margin against it is inflated. `NEXT_STEPS.md` §1 says this "shows up rather than
  silently distorting" — that understates it: it shows up *and* distorts.
  **Exclusion, fixed now: if an external agent's `think_over_limit` rate exceeds 1 % of its steps,
  its field is reported as a compatibility result only and no strength claim is drawn from it.**
- **Their agent may not be at its trained operating point.** Some repos ship a training-mode ε or
  load a checkpoint by a path relative to a different working directory. `agents.py:305` chdirs into
  the agent folder, which usually saves them — but verify the loaded table/model is non-trivial
  (not all-zeros, not a fresh init) *before* running 1000 rounds, or we measure an untrained agent
  and call it a field.
- **Noise floor.** ±0.12 on `score` at n = 1000 from the opponents' unseeded stdlib RNG, and an
  external agent adds its own unseeded RNG on top. Do not read any margin under ~0.15 without a
  second run at a different seed.
- **`evaluate.py:258` undercounts `killed_by_opponent`** (~2×, `benedict_task4.md` §5.8). Against a
  new field the own-bomb/enemy-bomb split is exactly what we want, so quote `suicides` and `died`
  and treat `killed_by` as a lower bound.

---

## 6 · Handling their code

- Third-party agents go in `agent_code/` because the framework requires it — **but nothing of theirs
  enters `agent_code/benedict_task4/`, and the submission zip is built from that folder alone.**
- Check the licence before committing anything of theirs. Preferred: **do not commit their code**;
  record the repo URL and the commit hash actually run in the `.meta.json` / ledger entry instead.
- Using an agent as an *opponent* is measurement, not copying — it is what `rule_based_agent` is
  for. `AGENTS.md`'s no-copy-pasting rule is about our solution, and it is untouched.
- Their agents may be **stronger**. That is information, not failure, and finding it out five weeks
  before the deadline is the entire point.
