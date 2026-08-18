# Installing four external agents as evaluation opponents

Status: COMPLETE. All four agents installed, model-verified and smoke-tested.
Started 2026-08-18.

Scope rules honoured: no git state changes, no `.gitignore` edit, no framework edits, no package
installs, no pre-existing `agent_code/` folder touched. New folders only, plus this directory.

## Summary table

(see "Model verification" below for the evidence behind the "model verified" column)

| agent | source commit | fixes applied | model verified | 30-rd score | invalid/rd | think_max_ms | over_limit | verdict |
|---|---|---|---|---|---|---|---|---|
| `ext_xiaoxiae_bindist_v2` | xiaoxiae/BombermanML `50b682f` | rename `train.py`; drop eps branch; unguard `setup` | yes - 1 078 278 params | **5.333** [4.07, 6.67] | 2.27 | 51.9 (mean-of-round-max 6.7) | **0** | **READY** |
| `ext_xiaoxiae_binary_v6` | xiaoxiae/BombermanML `50b682f` | same three | yes - 1 078 278 params | **5.967** [4.43, 7.60] | 2.23 | 36.9 (mean-of-round-max 6.0) | **0** | **READY** |
| `ext_aielka_ql_atom` | AI-ELka/BombermanRLAgents `8d85731` | absolute->relative imports; delete `train.py`+`add_own_events.py`; force `train=False` | yes - 329-state Q-table, 93 % nonzero | **5.033** [3.67, 6.53] | 18.93 (median 4) | 26.0 (mean-of-round-max 16.7) | **0** | **READY**, with the caveat in its section |
| `ext_lijesse_featureeverything` | Li-Jesse-Jiaze/MLE_project_bomberman `a7fe504` | pickle module alias; delete `train.py`; unguard `setup`; drop eps + periodic-reload branches | yes - 5 407 params | **5.800** [4.60, 7.17] | 7.10 | 61.5 (mean-of-round-max 37.9) | **0** | **READY**, slowest of the four - see timing |

Scores are the agent's own mean over 30 rounds vs 3x `rule_based_agent`, seed 990731,
with the 95 % bootstrap CI from `analyze.py`. **n = 30 is a liveness check, not a result** -
`AGENTS.md` puts the noise floor on `score` at +-0.12 even at 1000 rounds, and these CIs are
+-1.3 to +-1.6 wide. Do not quote them; re-measure at 300/1000 rounds before writing anything down.

For context in those same games the three `rule_based_agent`s averaged 2.4-2.8, and our shipped
rung-4 table measures 3.949 (`experiments/benedict_task4.md`). All four external agents landed
above that here, which is consistent with FINDINGS.md's n = 15 shortlist.

## Model verification (the check that matters most)

Script: `scratchpad/external/install/verify_models.py`. It imports each agent exactly as
`agents.py:204` does (`importlib.import_module('agent_code.<name>.callbacks')`), chdirs into the
agent folder as `agents.py:305` does, calls `setup(self)` with `self.train = False`, and then
dumps what actually landed in memory. Full output: `scratchpad/external/install/verify_models.txt`.

| agent | what loaded | evidence it is not a fresh init |
|---|---|---|
| `ext_xiaoxiae_bindist_v2` | `DQN(21 -> 1024 -> 1024 -> 6)`, 6 tensors, **1 078 278 params**, from `target-model.pt` (sha1 `bb9f8953c1051ada`) | mean abs difference vs a freshly constructed `DQN(...)` is 0.22-0.45 per tensor; output-layer bias mean +0.263 (a fresh init is ~0) |
| `ext_xiaoxiae_binary_v6` | same architecture, 1 078 278 params, `target-model.pt` sha1 `87cb349c029563ea` | 0.16-0.29 vs fresh init; **weights differ from bindist_v2** (all four `.pt` md5s distinct), so the two folders are genuinely two different agents |
| `ext_aielka_ql_atom` | `dict` Q-table, **329 states**, 20-tuple keys -> 6 Q-values | 1 974 Q-values, **1 836 nonzero (93.0 %)**, range [-11.37, +1.22], and **0 of 329 rows are all-zero**. 329 states matches FINDINGS.md's identification of this as lukevoss's "Atom" table |
| `ext_lijesse_featureeverything` | duelling `DQN(34 -> 64 -> 32 -> 16 -> value/advantage)`, 20 tensors, **5 407 params**, unpickled from `my-saved-model.pt` | mean abs difference vs fresh init 0.78 over all tensors; the LayerNorm gains have drifted from their init value of 1.0 to a mean of **3.13** (std 3.42) - that cannot happen without training |

The loader in every case fails loudly rather than falling back: xiaoxiae calls
`load_state_dict` on a path built from `__file__` (raises if absent), AI-ELka's
`q_learning.py:30` explicitly `raise FileNotFoundError`, Li-Jesse `open()`s the checkpoint
directly. There is no silent-untrained-fallback path in any of the four.

## Per-agent detail

Common to all four: copied with `cp -RL` (never `ln -s`, per the FINDINGS.md warning; `-L`
dereferences the xiaoxiae avatar/bomb symlinks that point at a sibling folder, which a plain
`cp -R` would have left dangling). `__pycache__`, `.git` and training logs were excluded.
`agents.py:226` recreates a `logs/` directory inside each agent folder on first run - that is the
framework, not the agent, and is harmless.

---

### 1. `ext_xiaoxiae_bindist_v2` and `ext_xiaoxiae_binary_v6`

Source: `scratchpad/external/xiaoxiae__BombermanML/agent_code/{binary_distance_agent_v2,binary_agent_v6}`,
commit `50b682f`, **GPL-3.0**. The two `callbacks.py` are byte-identical; only the weights differ.

`setup(self)` / `act(self, game_state)` present with our signatures. No absolute paths: the model
path is `f"{cwd}/target-model.pt"` with `cwd = os.path.abspath(os.path.dirname(__file__))`, which
is absolute but computed at import time from the file's own location, so it survives the
`agents.py:305` chdir and the folder rename. `torch.load(..., map_location=device)` with
`device = cpu`, so no CUDA-deserialisation problem. No import from outside the agent folder except
`events`/`settings`, which resolve against the project root. `matplotlib` is imported at module
scope; it is installed (3.11.1) so this works, but note it is deliberately kept out of
`requirements.txt` - irrelevant for an opponent, relevant if anyone ever bundles one of these.

Three changes, identical in both folders:

1. **`train.py` -> `_agent_defs.py`, and `from .train import *` -> `from ._agent_defs import *`.**
   Their `callbacks.py` gets *everything* from `train.py` by star-import - `DQN`,
   `state_to_features`, `TARGET_MODEL_PATH`, `FEATURE_VECTOR_SIZE`, `LAYER_SIZES`, `ACTIONS`,
   `device`, `MANUAL`, `torch` itself. Deleting `train.py` outright would break the agent at
   import. Renaming it achieves the stated goal - `agents.py:206` can no longer find a `train`
   module, so `setup_training`/`game_events_occurred`/`end_of_round` are unreachable - while
   keeping the definitions `act()` needs. The training functions are still *defined* inside
   `_agent_defs.py`; they are simply never called.
2. **Removed the epsilon-exploration branch** at the top of `act()` (`if self.train: ... return
   choice(ACTIONS)`). Inert in practice (`EPS_START = EPS_END = 0.00`, and `self.train` is False
   for an opponent), removed so a frozen opponent cannot take a random action under any invocation.
3. **`if not self.train and not MANUAL:` -> `if not MANUAL:`** in `setup()`, so the trained net is
   always loaded rather than left unset in train mode. `MANUAL` is `False` in both.

Behaviour: aggressive bombers, ~25 bombs and ~25 crates per round, ~0.4 kills, but they *die* -
survival 0.60 / 0.50, suicides 0.33 / 0.43. Rounds end early (254 / 269 steps).

---

### 2. `ext_aielka_ql_atom`

Source: `scratchpad/external/AI-ELka__BombermanRLAgents/agent_code/ql`, commit `8d85731`,
**no LICENSE**. This is FINDINGS.md's recovery of lukevoss's "Atom" (published 5.04).

`setup(self)` / `act(self, game_state)` present with our signatures. Model path is
`os.path.join('./models', 'q_table.pkl')` - relative, correct under the `agents.py:305` chdir -
and `q_learning.py:30` raises `FileNotFoundError` rather than falling back to an empty table.

Three changes:

1. **Absolute package imports rewritten to relative.** Every module did
   `from agent_code.ql.utils import *`, `from agent_code.ql.feature_extraction import ...`,
   `from agent_code.ql.q_learning import ...`, `import agent_code.ql.own_events as own_e`.
   FINDINGS.md records that this hard-wires the folder name `ql` and that renaming the folder
   breaks it. Rewritten to `from .utils import *`, `from .feature_extraction import ...`,
   `from .q_learning import ...`, `from . import own_events as own_e` in `callbacks.py`,
   `feature_extraction.py` and `q_learning.py`. **The FINDINGS.md "must keep the name `ql`"
   constraint no longer applies to this copy.** No `agent_code.` string remains in the folder.
2. **Deleted `train.py` and `add_own_events.py`.** `add_own_events.py` is imported only by
   `train.py`, so it is dead once `train.py` is gone. Nothing in `callbacks.py` touches either.
   `QLearningAgent.training_step`/`update_q_value`/`save` remain defined in `q_learning.py` but
   are now unreachable.
3. **`train = self.train` -> `train = False`** in the single `self.agent.act(...)` call. Already
   the effective value for an opponent; hardcoding it removes the epsilon and
   "explore-untried-actions-systematically" paths permanently.

**The 30.6 invalid actions/round from FINDINGS.md: partly reproduced, and not what it looks like.**
Through our harness the mean is 18.93/round, but the *median is 4* - the distribution is bimodal:

```
0 1 1 1 1 1 1 1 2 2 2 2 3 3 4 4 4 5 5 5 5 5 6 8 9 | 39 79 105 120 144
```

Five of thirty rounds carry 487 of the 568 invalid actions. I chased the mechanism
(`why_invalid_ql.py`, `why_invalid_ql2.py`, both in this directory) and my first hypothesis was
**wrong**, so both scripts are kept:

- *Hypothesis (from `q_learning.py:37`):* with epsilon forced to 0, the only random-action branch
  left is `if state not in self.q_table`, so a 329-state table must be missing states.
  **Refuted: 0 table misses in 3 384 steps.** The 329 states cover everything its feature
  extractor emits in play.
- *What is actually happening:* **state aliasing**. The 20-bit feature vector encodes only
  directional *recommendations* (coin / crate / opponent / safety) plus one "can I bomb safely"
  bit - it carries **no tile-legality information at all**. The row for the state
  `(0,0,...,0,1)` - no recommendation in any direction, bomb available - has a single
  untied greedy action, `LEFT`. Over 40 traced rounds that state was visited 847 times and
  produced an invalid action **757** of them. Because an invalid action leaves the agent in place
  and nothing else in the vector changes, the state recurs and the same `LEFT` fires again:
  the longest observed run is **227 consecutive invalid steps** in one round. The agent wedges
  itself against a wall and stays there until the 400-step cap.
- Ties are not the cause: 13 938 of 13 951 traced steps had a single unique argmax.

Consequence for measurement: this is Atom's own behaviour, not a porting artefact (FINDINGS.md saw
it at n = 15 through `main.py`, before any of my edits). It is a legitimate opponent - it is the
agent as its authors shipped it - but be aware its score is depressed in roughly one round in six,
that those rounds inflate `steps` (349/round) and `survived` (0.80) for the wrong reason, and that
`invalid` for this agent should be reported as a median, not a mean.

---

### 3. `ext_lijesse_featureeverything`

Source: `scratchpad/external/Li-Jesse-Jiaze__MLE_project_bomberman/agent_code/feature_is_everything`,
commit `a7fe504`, **no LICENSE**. Third place, SS2024.

`setup(self)` / `act(self, game_state)` present with our signatures. Model path is
`open("my-saved-model.pt", "rb")` - relative, correct under the chdir. Only intra-folder imports
(`from .features import Feature`), plus `settings`.

Four changes:

1. **A `sys.modules` alias for the pickle - this one is load-bearing and would otherwise have been
   a silent-ish failure.** `my-saved-model.pt` is not a state dict; it is a *pickled `nn.Module`
   instance*, and the pickle records its class as
   `agent_code.feature_is_everything.model.DQN` (verifiable with
   `strings my-saved-model.pt | grep agent_code`). Renaming the folder to
   `ext_lijesse_featureeverything` makes that module path unimportable and `pickle.load` raises
   `ModuleNotFoundError`. Fixed at the top of `callbacks.py`:

   ```python
   sys.modules.setdefault("agent_code.feature_is_everything", sys.modules[__package__])
   sys.modules.setdefault("agent_code.feature_is_everything.model", _model)
   ```

   `verify_models.py` confirms the alias is the path actually taken: the loaded object's class
   resolves to `agent_code.ext_lijesse_featureeverything.model.DQN`, i.e. *our* copy of `model.py`.
2. **Deleted `train.py`** (nothing in `callbacks.py` imports it).
3. **`if not self.train:` removed from `setup()`** so the net always loads, plus an added
   `self.policy_net.eval()`. The `eval()` is provably inert for this architecture - the net is
   `Linear`/`LayerNorm`/`ReLU` only, no dropout and no batchnorm - it is there so the model can
   never be left in train mode.
4. **Removed two training-only branches in `act()`**: the epsilon-greedy branch
   (`if self.train: ... random.randrange`) and `self.steps_done += 1`, and with them the
   `if game_state["round"] % 500 == 0 and game_state["step"] == 1: setup(self)` self-play reload.
   Checked before removing: `Feature` keeps no cross-round state that the reload was resetting -
   `main_enemy` is recomputed every `__call__`, and `acc_scores` is dead (the only line reading it
   is commented out upstream). So the removal reloads the same weights less often and changes
   nothing else.

Behaviour: the "farmer" archetype FINDINGS.md describes. 44.9 crates and 3.97 coins per round from
only 16.8 bombs, 0.37 kills, **survival 0.87 and 0.07 suicides** - by far the most survivable of
the four.

**Timing, the one thing to watch.** 14.24 ms/step mean, 61.5 ms worst single step, 37.9 ms
mean-of-round-max - about 10x the other three and ~25x `rule_based_agent`. `think_over_limit` is
0/10 504 steps here, but the reference tournament box (one thread of a Ryzen 5 2600) is ~4-6x
slower than this M5, which projects a mean of ~60-85 ms/step and a worst step of ~250-370 ms.
That is inside the 0.5 s limit but not comfortably. It is fine as an opponent (opponents are not
timed the way our submission is) and it costs real wall-clock: 158 s for 30 rounds versus 22 s for
the xiaoxiae pair, so a 1000-round evaluation against it is ~90 minutes.

---

### Things I did not do, and one thing I did not do that happened anyway

- **No package was installed**, no framework file was edited, no pre-existing `agent_code/` folder
  was touched, and no `git add`/`commit`/`push` was run.
- **I did not edit `.gitignore`.** For the record, `git status` shows it modified with a new
  `agent_code/ext_*/` rule that appeared at 09:21 while I was working - that edit is not mine
  (nor are the modification to `experiments/benedict.md` or the untracked `scratchpad/audit11/`).
  The rule is the right one to have: `agent_code/ext_*/` is now correctly ignored
  (`git check-ignore -v` -> `.gitignore:169`), which matters because two of the three source repos
  carry **no licence at all** and the submission zip is built out of `agent_code/`.
- The four folders total **17 MB**, almost all of it the two 4.3 MB xiaoxiae `.pt` pairs. The
  `policy-model.pt` files are unused at inference (only `target-model.pt` is loaded) but were kept
  so the copies stay faithful to the source.

## Working evaluate.py invocations

These four are exactly the commands that produced the numbers above. They ran clean - **no round
threw an exception in any of the four**, all 30 rounds and 120 agent-rows landed in every CSV.

```bash
cd /Users/benedictvonschubert/Projects/bomberman_RL

BM_QUIET_LOGS=1 uv run python tools/evaluate.py --agents ext_xiaoxiae_bindist_v2 \
  --opponents rule_based --n-rounds 30 --seed 990731 \
  --label smoke_ext_xiaoxiae_bindist_v2 --out-dir scratchpad/external/install

BM_QUIET_LOGS=1 uv run python tools/evaluate.py --agents ext_xiaoxiae_binary_v6 \
  --opponents rule_based --n-rounds 30 --seed 990731 \
  --label smoke_ext_xiaoxiae_binary_v6 --out-dir scratchpad/external/install

BM_QUIET_LOGS=1 uv run python tools/evaluate.py --agents ext_aielka_ql_atom \
  --opponents rule_based --n-rounds 30 --seed 990731 \
  --label smoke_ext_aielka_ql_atom --out-dir scratchpad/external/install

BM_QUIET_LOGS=1 uv run python tools/evaluate.py --agents ext_lijesse_featureeverything \
  --opponents rule_based --n-rounds 30 --seed 990731 \
  --label smoke_ext_lijesse_featureeverything --out-dir scratchpad/external/install
```

Read back with, e.g.:

```bash
uv run python tools/analyze.py scratchpad/external/install/smoke_ext_aielka_ql_atom.csv --preset task4
```

To use one as an *opponent* of our agent, pass it to `--agents` alongside ours (the smoke runs
above used `--opponents rule_based`, i.e. the preset field, which is what makes them comparable to
FINDINGS.md's numbers). For the mixed-field measurement FINDINGS.md argues for:

```bash
BM_QUIET_LOGS=1 uv run python tools/evaluate.py \
  --agents benedict_task4 ext_xiaoxiae_binary_v6 ext_lijesse_featureeverything ext_aielka_ql_atom \
  --n-rounds 300 --label ours_vs_ext_mixed --out-dir results/eval/task4_tournament
```

(untested by me - I only ran the four single-agent smokes above, and I did not touch
`results/`. Note the standard seed 20260731 applies there, not the 990731 used for the smokes.)

## Files written by this task

Everything is inside the two sanctioned locations.

| path | what |
|---|---|
| `agent_code/ext_xiaoxiae_bindist_v2/` | installed agent (git-ignored) |
| `agent_code/ext_xiaoxiae_binary_v6/` | installed agent (git-ignored) |
| `agent_code/ext_aielka_ql_atom/` | installed agent (git-ignored) |
| `agent_code/ext_lijesse_featureeverything/` | installed agent (git-ignored) |
| `scratchpad/external/install/INSTALL.md` | this file |
| `scratchpad/external/install/verify_models.py` + `.txt` | the model-is-real check and its output |
| `scratchpad/external/install/why_invalid_ql.py` | the refuted table-miss hypothesis for Atom |
| `scratchpad/external/install/why_invalid_ql2.py` | the state-aliasing trace that explains it |
| `scratchpad/external/install/smoke_ext_*.csv` + `.meta.json` | the four 30-round smoke runs |
