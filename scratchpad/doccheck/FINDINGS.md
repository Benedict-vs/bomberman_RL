# Doc fact-check — `agent_code/benedict_task4/README.md` and `experiments/benedict_task4.md`

Verification pass, 2026-08-21. Read-only outside this directory.

## READ THIS FIRST — both targets were rewritten while I audited them

| file | when I started (10:30) | when I finished (10:40) |
|---|---|---|
| `agent_code/benedict_task4/README.md` | 2 772 B, mtime 08-16 20:12 | **4 091 B, mtime 08-21 10:37** |
| `experiments/benedict_task4.md` | 15 872 B, mtime 08-17 20:07 | **25 056 B, mtime 08-21 10:39** |
| `agent_code/benedict_task4/train.py` | 32 121 B, mtime 08-17 16:21 | **23 647 B, mtime 08-21 10:36** |

Another session rewrote all three mid-audit, and the rewrite fixed most of what the pre-10:30
versions got wrong (external-agent results, the noise floor, E50's truncation mechanism, the
env-var-free training path). **Everything below is scored against the 10:40 state.** Snapshots of
both versions are in this directory (`README_snapshot.md`, `benedict_task4_snapshot.md` = old,
`benedict_task4_snapshot2.md` = new, `train_snapshot_*.py`).

**Unrelated to the docs, but check before you commit:** `git status` shows **12 committed files
deleted from the working tree**, all E37 seed-106 training records —

```
 D results/train/task4_tournament/benedict_task3__q_e37_{PLB2,ctl2,PAR,SHF}_s106.{csv,meta.json}
 D results/train/task4_tournament/e37_{PLB2,ctl2,PAR,SHF}_s106.out
```

`benedict_task3__q_e37_PLB2_s106.csv` is the training log of the run that produced the **shipped
table**, and `benedict_task4.md:376` cites `results/train/task4_tournament/` as committed evidence.
If that deletion is not deliberate, `git restore` it.

---

## 1 · Problem table

Severity: **H** wrong and consequential · **M** stale or misleading · **L** cosmetic.
Line numbers are against the 10:40 files.

### `experiments/benedict_task4.md`

| # | line | what it says | what is true | sev |
|---|---|---|---|---|
| A1 | 180, 314 | "**E45** — the bombs that clear nothing are the *safe* ones …"; "bombing discipline (E45/E46)" | **There is no E45 entry in `experiments/benedict.md`.** The headers run `## E46` (L576) → `## E44` (L770); E45 exists only as seven in-text references inside E46 and as `scratchpad/strategy/bomb_siting.py`. Every number in this paragraph (72.9 %, 18.8 %, the 0.40×/1.54×/7× lift tables) is real — I verified each against E46's Question section — but the citation resolves to nothing, and `AGENTS.md` makes the per-entry ledger the thing the report is built from. Either write the E45 entry or re-cite these as E46's pre-work. | **H** |
| A2 | 181 | "72.9 % of armed steps have zero crates in blast range and the policy bombs on 18.8 % of them, **so ~45 % of bombs clear nothing**. But … **0 crates is 32.2 % of bombs**" | The two figures for the same quantity sit in adjacent sentences and disagree. 45 % is a derived number (it implies a ~62 % bombing rate on crate-bearing armed steps); 32.2 % is the measured share from E46's killing-bomb table. One of them needs a qualifier saying which field/denominator it uses, or the 45 % should go. *Moderate confidence* — I could not check the derivation because E45 has no entry (A1). | **M** |
| A3 | 276 | "`tools/evaluate.py:**258**` undercounts `killed_by_opponent`" | The statement is correct; the line is **259**. `sed -n '259p' tools/evaluate.py` → `record["killed_by_opponent"] = max(0, died - suicides)`. Line 258 is `record["died"] = died`. | L |
| A4 | 226 | "**The mechanism is survival, not siting.**" | This is E41's reading and **E47 contradicts it in as many words** (`benedict.md:531`): *"Their suicides are 0.377 against our 0.505: halved, not eliminated, and E46 already priced that exchange at −0.283 score. **Their edge is not survival.**"* E46 then *tested* buying survival on that same external field and lost 0.283 score — which §3 of this document reports two pages earlier. The paragraph's own second half (pool +57 %, conversion 69.5 % → 91.3 %) is the E47 account and is right; the opening sentence is the one that needs softening to "survival collapses, and the kill deficit splits two ways". | **M** |
| A5 | 369–374 | "Reproducing **E24–E36** requires checking out the commit … The same now applies to **E19, E30 and E33**: `BM_SHAPE`, `BM_D4` and `BM_ESCAPE` were removed from `train.py` … along with the `ALPHA`/`EPS` mode switches" | Correct and newly accurate, but **`BM_ARM` is missing from the list** and it is the one that bites widest: `scratchpad/benedict/e30_arms.sh` … `e38_arms.sh` (nine scripts, **including `e37_arms.sh` and `e38_arms.sh`**) all set `BM_ARM`, and today's `train.py` dropped it. So the range should extend past E36 to **E38**, and `BM_ARM` should be named. Side effect worth a sentence: `RUN_NAME` is now `q_task4_s{RUN_INDEX}` with no arm component, so two arms run at the same `BM_RUN_INDEX` will append into one training CSV — the exact failure `train.py`'s own comment says cost half an hour in E05b. | **M** |
| A6 | 227 | "survival collapses **0.440 → 0.223**" | True but it is the worst of the three fields (`feature_is_everything`). E41's other two are 0.310 and 0.273. "0.440 → 0.22–0.31" is the honest range; the surrounding sentence pairs it with `binary_v6`'s crates/bomb 1.80 and bombs 19.6, which are from *different* fields (1.80 is `binary_v6`, 19.6 is `bindist_v2`). Three fields, three cherry-picked endpoints in one sentence. | **M** |
| A7 | 9–10 | "**Audited six times** … **All six overturned something**" | Consistent with `scratchpad/audit7/`…`audit12/` (six rung-4 audits, all present and tracked). Note `AGENTS.md` still says "**Eight** independent audits have run on this project" — that is now the stale one (twelve exist). Not this document's bug, but fix it in the same pass. | L |
| A8 | 22 | "score 3.694 → **3.949**, +0.255 [+0.039, +0.475]" | **Verified and not fragile.** I re-ran `analyze.py --compare` on the two committed CSVs: `+0.255 [+0.039, +0.475]`, t = +2.25, sign-flip p = 0.0246, Wilcoxon 0.024, **no `(fragile)` marker**, and every other row of §1's table reproduces to the digit. Recorded because the `(fragile)` rule postdates the original document and this row had never been scored under it. | ok |
| A9 | 128–152 (§4), 160–199 (§5), 204–241 (§6) | E38 horizon table; E44 parent/coverage; E43 death attribution; E41 calibration and margins | **All verified against `experiments/benedict.md`.** E38's five-column table matches L2085 exactly (3.976/3.810/3.848/3.637/3.206, crates/bomb 1.182→0.988, −0.770 [−1.355, −0.185] 0/5). E44's −1.661/−1.936/−2.019 and 3 595/8 168 → 44 % match L861–880. E43's 78.6/14.9/4.2/2.3 and 76.7 % match L975–1000. E41's 5.572/5.336/5.143/4.690 and −1.008/−0.797/−0.544/+0.912 match L1396–1420. | ok |
| A10 | 285–290 (§7.11) | "`analyze.py` seeded its percentile bootstrap with a hard-coded `default_rng(12345)`" | Verified: `tools/analyze.py:220`. | ok |
| A11 | 355–357, 376–378 | Cited paths: `checkpoints/benedict_task4/q_table_parent.npy`, `…q_table_e37_PLB2_s106__ep20000.npy`, `scratchpad/audit7…12/`, `scratchpad/benedict/`, `scratchpad/strategy/`, `scratchpad/external/`, `scratchpad/strategy/survival_value.py` | **All exist and are tracked.** md5 identity claim verified: `agent_code/benedict_task4/q_table.npy` and `checkpoints/benedict_task4/q_table_e37_PLB2_s106__ep20000.npy` are both `54d63bc79179fdf80d3b9bfb80f21461`. | ok |

### `agent_code/benedict_task4/README.md`

| # | line | what it says | what is true | sev |
|---|---|---|---|---|
| B1 | 32–33 | "Mean decision time **0.137 ms**, worst single step observed **53.4 ms** across **180 000 evaluated rounds**" | **The only claim in the rewritten README that is still stale.** These are `scratchpad/audit9/REPORT.md`'s figures over the **E37 sweep alone** — I reproduce that subset exactly (`benedict_q_e37_*` CSVs: 184 000 benedict agent-rounds, max **53.401 ms**). Across **all** committed eval CSVs the worst benedict step is **54.435 ms** (`results/eval/task4_tournament/benedict_q_e36_PLB_s103__ep20000__task4_rb_val550731.csv`, and that file predates the README) over **674 800** agent-rounds — 472 600 in `task4_tournament` alone. Sentence reads as global; it is one sweep, the max is understated, and the round count is 2.6–3.7× low. No safety consequence: `think_over_limit` is 0 in every CSV and 54 ms sits far under 500 ms. **I cannot reproduce 0.137 at all** — on that subset I get 0.130 step-weighted and 0.110 unweighted, and audit 9's own timing paragraph quotes a different statistic (mean `think_max_ms` ≈ 0.27 ms). *Low confidence on where 0.137 came from; high confidence it is not recomputable from the CSVs.* | **M** |
| B2 | 52–53 | "the mechanism is **survival under pressure** rather than bomb siting" | Same issue as A4 — E47 says "Their edge is not survival", E46 priced buying it at −0.283 score. The cross-reference to `benedict_task4.md` §6 now resolves correctly (§6 is the external-validity section). | **M** |
| B3 | 65–70 | "The run writes `q_table_trained.npy` beside the agent … `q_table.npy` is never written by training, so a stray `--train` cannot destroy the submitted model." | Verified true, and a real improvement. **But incomplete:** `checkpoint_file()` derives from `OUTPUT_FILE`, so the same env-var-free run also writes `q_table_trained__ep5000.npy`, `__ep10000.npy` and `__ep20000.npy` — **3 × 3 MB of checkpoints inside `agent_code/benedict_task4/`**, i.e. inside the submission zip. `.gitignore` (`agent_code/*/q_table_*.npy`) hides them from git but not from the zip, and `train.py:212–217`'s own comment says keeping checkpoints out of that folder is the point. Either say so in the README or route checkpoints to `CHECKPOINT_DIR` unconditionally. | **M** |
| B4 | 57, 65–66 | "Defaults in `train.py` are the shipped configuration … including the exploration seed (`BM_RUN_INDEX=106`)" | **Now true** — today's rewrite moved the default from 20 to 106 and dropped the stale comment that said the shipped seed was 5. It was **false at HEAD**. Confirm the `train.py` rewrite is intended before committing; it is a 239-line deletion touching the training code that produced the shipped table. | verify |
| B5 | 59–63 | Training command, no `BM_QUIET_LOGS=1` | `AGENTS.md` says to always set it for a sweep. Arguably a single run is not a sweep, and the old README did set it. Flagging, not asserting. | L |
| B6 | 17, 42–43, 45–46, 48–51, 72–74 | md5 identity; 3.949 / 0.406 / 0.226 / 0.488; 3.254 / 0.286 / 0.196 / 0.533; "re-evaluated gives 3.828"; 5.572 / 5.336 / 5.143 / 4.690; "~44 % of the warm table's learned rows (E44)" | **Every one verified.** Sources in §4 below. | ok |

---

## 2 · Claims overturned by E38–E50

**The 10:39 rewrite of `experiments/benedict_task4.md` already absorbed almost all of this.** What
follows is the list I built against the pre-rewrite version, kept because it is the checklist to
score the new document against — and because two items (marked ⚠) are **still live**.

| pre-rewrite claim (old §/line) | entry that changes it | status in the 10:40 document |
|---|---|---|
| "The `rule_based_agent` field is a *proxy* … **Nothing here measures play against other students' learned agents.**" (old §6, L214–215) | **E41** (`benedict.md:1246`), n = 1000, four third-party agents; also E42/E44/E46/E47/E48/E49 | **Fixed.** New §6 is a full external-validity section and §1 L40–41 forward-references it. |
| "`end_of_round` treats truncation at `MAX_STEPS` as termination (no bootstrap) on **~33 %** of rounds, worth ≈8.8 Q units. Known since rung 3, never fixed in isolation." (old §6, L188–189) | **E50** (`benedict.md:24`): it is a **duplicate un-bootstrapped update on the same (s,a)**, not a missing bootstrap — "a fix written from that description would have made it *worse*"; the 33/70 % figure is an **evaluation** number, training survival is 10–22 %, so ≈0.1 % of Q updates; recommendation is *do not fix* | **Fixed.** New §8 bullet 3 states mechanism, rate, bound and the correct fix. |
| "eight of nine interventions still failed — seven of them because they bought survival" (old §intro) | E38, E42, E44, E46, E48, E49 add six more failures; E46 is the 7th and E48 the 8th survival replication, and **E46 is the first to price it: +0.050 survived = −0.283 score** | **Fixed.** New §3 lists fifteen interventions and carries E46's exchange rate and E48's inert death penalty. |
| "survival … does not convert into points **on this board**" stated from `rule_based`-only evidence | **E41** (`benedict.md:1425`): *"Survival was never worthless; it was **non-binding** against the only opponent we ever measured. Six replications of a null measured one field."* | **Fixed.** New §6 L104–109 states the reversal explicitly. |
| "The answer to a kill deficit is a feature, not a price — which is exactly what E37 then was." (old §3) | **E43**: the opponent-danger digit is refuted at a **2.3 %-of-deaths** ceiling before being built; **E47**: the deficit is 65–70 % kills, splitting into pool (+57 %) and conversion (69.5 % → 91.3 %); **E44**: the training-distribution fix fails at both learning rates | **Fixed.** New §5 (E43) and §6 carry both. |
| "coverage has not bound since E30 — all-zero-row share is 0.00006 across every arm" (old §4) | **E44**: from-scratch tables reach **3 595–4 095 nonzero rows against 8 168–8 631 warm, i.e. 44 %** — coverage binds hard the moment the parent is removed | **Fixed.** New §4 makes the warm parent the largest effect on the rung (−1.661/−1.936/−2.019). |
| "Longer training … (E38, and the only rung-4 hyperparameter that was inherited rather than tested)" | E48/E49 subsequently test the whole reward table; E50 characterises the truncation defect | **Fixed.** New §3/§8 cover both. |
| "Why the four-way split beats the lattice bit alone … is untested. A 2-way information-free control would test it." | Still untested — **but E49's P4** is an independent echo of the same α mechanism: ×2 reward magnitude at fixed ratio cost **−0.202 [−0.324, −0.089]** held out and raised `invalid` +0.873/+1.028 | **Fixed.** New §8 bullet 4 cites E49. |
| ⚠ "**The mechanism is survival, not siting**" | **E47** (`benedict.md:531`): *"Their edge is not survival."* **E46**: buying survival on the external field cost −0.283 score with the mechanism working exactly as designed | **STILL LIVE** in new §6 L226 and in `README.md` L52. See A4 / B2. |
| ⚠ `tools/evaluate.py:258` | line is now **259** | **STILL LIVE**, new §7.9 L276. See A3. |

Two further things later entries add that **neither document mentions**, offered as suggestions
rather than defects:

- **E49's verdict that the reward table is closed** ("`COIN_COLLECTED = 5`, `CRATE_DESTROYED = 1.0`,
  `GOT_KILLED = −5`, `KILLED_OPPONENT = 0` is at or adjacent to a local optimum", with a **cliff**
  below the crate reward at −1.789 and flat ground above) is a per-hyperparameter result and
  `AGENTS.md` calls hyperparameter optimisation an explicit grading criterion. New §3 lists the
  arms; the four-direction summary table from E49's verdict is the compact form.
- **E49's cleanest statement of the survival null** — *"the agent's survival behaviour is controlled
  by what it is paid to do, not by what it is paid to avoid"* (crate reward ×2 raises suicides
  +0.132, while deleting the −5 death penalty moves them +0.010) — is a better one-line summary than
  §3 currently has.

---

## 3 · Reproduction commands

I audited flags, env-var names and paths statically against `main.py --help`, `tools/evaluate.py`,
`agent_code/benedict_task4/train.py` and `callbacks.py`. **I did not run training** — it writes
outside `scratchpad/doccheck/`.

### `benedict_task4.md` §9 L347–353 and `README.md` L59–63 — training

```bash
uv run python main.py play --agents benedict_task4 \
    rule_based_agent rule_based_agent rule_based_agent \
    --scenario classic --train 1 --n-rounds 20000 --no-gui --seed 810731
```

**Would work today.** Every flag exists (`--agents`, `--train {0..4}`, `--scenario classic`,
`--seed`, `--n-rounds`, `--no-gui`). With no `BM_MODEL_SUFFIX`:

- `OUTPUT_FILE` → `agent_code/benedict_task4/q_table_trained.npy` (does not exist → `save_table`'s
  pre-existence guard passes). ✓
- warm start → `CHECKPOINT_DIR/q_table_parent.npy` = `checkpoints/benedict_task4/q_table_parent.npy`,
  **which exists**. ✓

**This was broken until 10:36 today.** At HEAD, `warm_start` anchored to
`os.path.dirname(MODEL_FILE)`, which without `BM_MODEL_SUFFIX` is the *agent* folder, so the
env-var-free command would have died with `FileNotFoundError` on
`agent_code/benedict_task4/q_table_parent.npy`. The rewrite introduced `CHECKPOINT_DIR` and fixed
it. Nothing to do — noted so you know it was checked and when.

**One undocumented side effect:** see B3 — three `q_table_trained__ep*.npy` files land inside the
submitted agent folder.

### `benedict_task4.md` §9 L364–367 — evaluation

```bash
uv run python tools/evaluate.py --agents benedict_task4 --opponents rule_based \
    --n-rounds 1000 --seed 990731 --label benedict_task4_shipped_e37__task4_rb_ship990731 \
    --out-dir results/eval/task4_tournament
```

**Would run, and that is the problem.** All flags valid (`--opponents` accepts `rule_based` →
`["rule_based_agent"] * 3`). But `tools/evaluate.py:334` opens the CSV in `"w"`, and
`results/eval/task4_tournament/benedict_task4_shipped_e37__task4_rb_ship990731.csv` **is committed
evidence** — the very file the document's own §1 L35–38 cites for the 3.828 re-evaluation. Copying
the block and running it overwrites that measurement with a fresh one differing by up to the ±0.12
noise floor. Suggest printing a distinct label (e.g. `…_ship990731_rerun`) in the doc.

### Two conventions worth reconciling with `AGENTS.md`

1. **`--seed 990731` / `550731` / `810731` are undocumented project-wide.** `AGENTS.md` says
   *"**Never change `--seed`.** Default `20260731`."* and neither `AGENTS.md` nor `MEASUREMENT.md`
   mentions 990731 (held-out ship), 550731 (validation) or 810731 (training world) — I grepped both.
   Every rung-4 entry and both documents depend on that three-seed protocol. As written, the
   reproduction blocks look like they violate a hard project rule. Add the protocol to
   `MEASUREMENT.md`.
2. **`BM_ARM` removal** — see A5; extend §9's warning to E38 and name `BM_ARM`.

---

## 4 · Every number I traced, and to what

Verified from the committed CSVs with `scratchpad/doccheck/summ.py` (agent-slot means over 1000
rounds), and cross-checked with `tools/analyze.py --compare`.

| figure | source | result |
|---|---|---|
| shipped 3.949 / `won` 0.406 / kills 0.226 / suicides 0.488 / killed_by 0.048 / crates 33.55 | `benedict_q_e37_PLB2_s106__ep20000__task4_rb_ship990731.csv` | exact (33.5450) |
| previous ship 3.694 / 0.372 / 0.196 / 0.749 / 0.050 | `benedict_q_e33_ctl_s104__ep20000__task4_rb_ship990731.csv` | exact |
| paired diffs +0.255 / +0.034 / +0.030 / −0.261 / −0.002 / +0.032 with CIs | `analyze.py --compare --preset task4` | exact, **no row `(fragile)`** |
| reference in slot 3.254 / `won` 0.286 / kills 0.196 / suicides 0.533 | `ref_rule_based_agent__task4_rb_ship990731.csv`, slot 0 | exact |
| symmetric bar `won` 0.282 | same file, mean of the four slots = 0.28175 | exact |
| re-evaluation 3.828 (and 0.379 / 0.214 / 0.505 / 33.28 / survived 0.440) | `benedict_task4_shipped_e37__task4_rb_ship990731.csv` | exact; E47 and E49 both use these |
| 3.828 vs 3.694 as a paired comparison | `analyze.py --compare` | **+0.134 [−0.081, +0.350], p = 0.24 — not demonstrated.** The documents' framing (the sweep is the evidence, the held-out run only confirms) is exactly right, and this is why |
| E46's independent control on the same field | `benedict.md:686` | 3.944 at n = 4000 — corroborates 3.949 |
| E38 horizon table, all 15 cells + −0.770 [−1.355, −0.185] 0/5 | `benedict.md:2085` | exact |
| E44 warm parent −1.661/−1.936/−2.019; 3 595 vs 8 168 = 44 % | `benedict.md:861–880` | exact |
| E43 78.6/14.9/4.2/2.3, 76.7 %, 9.3 % → 23.3 % | `benedict.md:975–1000` | exact |
| E41 calibration 5.572/5.336/5.143/4.690; margins −1.008/−0.797/−0.544/+0.912; crates/bomb 1.249–1.960; survived 0.440 → 0.223–0.310 | `benedict.md:1396–1445` | exact |
| E47 gap decomposition: pool 0.308 → 0.482, conversion 69.5 % → 91.3 %, coins/crate 0.083 vs 0.134 | `benedict.md:494–520` | exact |
| E37 internals: 40 960/64 000, 43.9 %, 11 477 + 6 427 of 17 904, H = 0.192 of 0.942, −0.060 (14/15), 1.123 vs 0.361 ratio 3.12, BOMB 0.1810/0.1790/0.1789/0.1790, +0.130 [+0.061, +0.198], +0.224 + 5×0.011 = +0.279 of +0.280 | `benedict.md:2158–2355` | exact |
| md5 `q_table.npy` == `q_table_e37_PLB2_s106__ep20000.npy` | `md5` | `54d63bc79179fdf80d3b9bfb80f21461` both |
| `q_table.npy.layout.json` = `[4,4,4,4,5,5,2,5]`, product 64 000, file 3 072 128 B = 64 000×6×8 + 128 | filesystem | consistent |
| `analyze.py` bootstrap `default_rng(12345)` | `tools/analyze.py:220` | exact |
| `killed_by_opponent = max(0, died - suicides)` | `tools/evaluate.py:**259**` | doc says 258 |
| 53.4 ms / 180 000 rounds | `scratchpad/audit9/REPORT.md:176` — E37 sweep only | reproduces on that subset (53.401, 184 000 rounds); **global max is 54.435 ms over 674 800** |

---

## 5 · Could not verify

| item | why |
|---|---|
| README L32 "Mean decision time **0.137 ms**" | Not recomputable from any CSV subset I tried: E37 sweep gives 0.130 step-weighted / 0.110 unweighted; all of `task4_tournament` gives 0.110 / 0.093; all evaluations give 0.081 / 0.074. Audit 9's timing paragraph quotes mean `think_max_ms` ≈ 0.27 ms instead. Possibly a transcription of the unrelated `PAR` figure on `audit9/REPORT.md:191`. **Low confidence — treat as "recheck", not "wrong".** |
| §5 L180–190 E45's raw figures (72.9 %, 18.8 %, the lift tables) | No E45 ledger entry (A1). I confirmed every number appears in E46's Question section and is consistent there; I did not re-derive them from `scratchpad/strategy/bomb_siting.py`, and the ~45 % in A2 is the one that does not reconcile. |
| §3 E36 row (suicides 0.616 → 0.422, survived 0.333 → 0.525) and E35 row (reward 26.82 → 31.31, 17 % kill income, kills −0.002) | Inside E36/E35, below the E38 line in the ledger. Both predate the E38–E50 window I was asked to check and neither is contradicted by anything I read; not independently re-verified. |
| §7.3's E40 arithmetic ("passing P3 guaranteed failing P1"), §7.5's +0.280 → +0.198, §7.6's 96 % of the treatment effect | These come from audits 9–11. The audit reports exist and are tracked; I did not re-derive them. |
| "412 committed rung-4 evaluations before E41" | Correct as a historical statement (E41 asserts it). `results/eval/task4_tournament/` now holds **615 CSVs**, all tracked, 0 untracked. |
| Whether the `train.py` rewrite (−239 lines) preserves training behaviour | Out of scope and it is still in flux. It removed `BM_D4`, `BM_ESCAPE`, `BM_SHAPE`, `BM_ALPHA`, `BM_EPS`, `BM_ARM`, changed `BM_RUN_INDEX`'s default 20 → 106, and split `OUTPUT_FILE` from `MODEL_FILE`. **Worth a separate review before it is committed** — it is the code path that produced the shipped table. |

---

## 6 · Sections that are accurate

Said briefly rather than padded: **`benedict_task4.md` §1, §2, §4, §5 (E43/E47 parts), §6 and §8
are accurate** against the ledger and the CSVs, to the digit, apart from the items flagged above.
§7's method-failure list is accurate everywhere I could check it. The 10:39 rewrite is a large
improvement — the pre-rewrite version had two hard errors (external validity, the truncation
mechanism) that the new one has closed. The rewritten `README.md` is accurate apart from B1's
timing paragraph and B2's one sentence.

---

## 7 · Re-check at 10:45

`train.py` changed again (10:42) and `experiments/benedict.md` is now also modified. I re-verified
the three train.py-dependent findings against the 10:42 file — **all still hold**:

- `grep -c BM_ARM agent_code/benedict_task4/train.py` → **0** (A5 stands).
- `RUN_INDEX = int(os.environ.get("BM_RUN_INDEX", 106))` at L184 (B4 stands).
- `checkpoint_file()` still does `os.path.splitext(OUTPUT_FILE)`, and `OUTPUT_FILE` (L208) is the
  agent folder when `BM_MODEL_SUFFIX` is unset → `q_table_trained__ep*.npy` inside the submission
  folder (B3 stands).

Both target documents are byte-identical to my snapshots (`README.md` md5 `4fc33f5d…`,
`benedict_task4.md` md5 `d2ff318a…`), so §1–§6 above are scored against exactly what is on disk.
