# External Bomberman agents — sourcing findings

Third-party opponents to evaluate our agent against, beyond the framework's `rule_based_agent`.
Sanctioned by `final_project.pdf`.

Written 2026-08-18. Everything lives in `scratchpad/external/`; nothing outside it was touched.
No `git add/commit/push` was run in the project repo. Nothing was installed.

---

## Headline

1. **We recovered lukevoss's "Atom" — the 5.04 agent from the survey — from a *different* repo.**
   `lukevoss/Bomberman_RL_2024` ships Atom's code but **not** its `q_table.pkl`, so it is dead on
   arrival. `AI-ELka/BombermanRLAgents` contains a byte-for-byte copy of Atom's code under
   `agent_code/ql/` **with** `models/q_table.pkl` (329 states — the survey's "335-state table"),
   and it runs. Details in §Atom recovery.
2. **Three external agents look clearly stronger than our shipped table** (~3.95–3.98 score vs
   3× rule_based, `experiments/benedict_task4.md`): `binary_distance_agent_v2` (6.00),
   `binary_agent_v6` (5.60), `ql`/Atom (5.60). n = 15 each, so treat the ranking as a shortlist,
   not a result — but the gap to 3.95 is well outside what n = 15 noise plausibly explains for
   the top three.
3. **`xiaoxiae/BombermanML` is the single best find**: 15 trained agent variants, the two
   strongest agents measured here, an ELO harness of their own — and it is **GPL-3.0 licensed**,
   the only strong candidate with a licence permitting anything beyond local use.
4. **Licence reality: 20 of the 24 repos cloned have no LICENSE file at all.** Only
   `xiaoxiae/BombermanML` (GPL-3.0), `KunkelAlexander/bomberman_rl` (MIT),
   `daryl336/CSXX46p` (MIT) and `stefanDeveloper/bomberman` (MIT) carry one. No licence means
   all rights reserved. Clone and run locally, yes; vendor into our repo, redistribute, or ship in
   the submission zip, no. Keep `scratchpad/external/` untracked.

---

## How the load tests were done — read this before trusting any verdict

`scratchpad/external/testbed/` is a **copy** of our framework files (`environment.py`,
`settings.py`, `items.py`, `agents.py`, `events.py`, `main.py`, `fallbacks.py`, `replay.py`,
`assets/`) plus our provided agents, with each candidate agent folder copied into
`testbed/agent_code/`. "RAN" below therefore means the agent ran **against our `environment.py`
and our `settings.py`** — not against its own repo's copy.

**A warning about my own first pass, since it is exactly the failure mode you asked me to avoid.**
I initially *symlinked* the framework into the testbed. Python resolves `sys.path[0]` through the
symlink, so every run silently loaded the *project's* `agent_code/` instead of the testbed's, and
`timeout` (which does not exist on macOS) swallowed the resulting errors. All thirteen agents
"passed". Every one of those passes was fake. The testbed was rebuilt with real file copies and
everything below was re-measured. If you re-run this, `cp`, do not `ln -s`.

Smoke evaluation command per agent:

```
main.py play --agents <X> rule_based_agent rule_based_agent rule_based_agent \
  --no-gui --n-rounds 15 --seed 20260731 --scenario classic --save-stats
```

**n = 15 with `main.py --seed`, which is *not* our per-round-reseeded `evaluate.py` protocol.**
These numbers are a liveness check and a rough ranking. They sit far below the ±0.12 noise floor
`AGENTS.md` records for `score` at 1000 rounds, so **do not quote them anywhere**. `rb_avg` — the
mean of the three rule-based opponents *in the same games* — is the only thing that makes the
comparison worth anything at this n.

---

## Summary table

| repo | commit | licence | their agent folder(s) | weights? | verdict |
|---|---|---|---|---|---|
| **xiaoxiae/BombermanML** | `50b682f` | **GPL-3.0** | `binary_distance_agent_v2`, `binary_agent_v6`, `binary_agent_v5`, `binary_agent_v4`, `q_agent_v1..v3`, `distance_agent_v1`, … (15 total) | yes — `policy-model.pt`/`target-model.pt` per variant, `model.pkl` for q-agents | **loads and plays as-is** (except `distance_agent_v1`, CUDA fix) — **strongest set found** |
| **AI-ELka/BombermanRLAgents** | `8d85731` | none | **`ql` (= lukevoss's Atom, with weights)**, `dql`, `dql_large`, `ppo`, `ppo_l` | yes — `models/q_table.pkl` (329 states), `.pth`/`.pt` for the rest | **`ql` loads and plays as-is**; `dql_large` weak; `dql`/`ppo`/`ppo_l` broken |
| **Li-Jesse-Jiaze/MLE_project_bomberman** | `a7fe504` | none | `feature_is_everything` (3rd place SS24), `dqn`, `dqn_5tile` | yes — `my-saved-model.pt` | **`feature_is_everything` loads and plays as-is**; `dqn`/`dqn_5tile` need the CUDA fix |
| **leonbegiristain/RL_Bomberman** | `2b4c799` | none | `hung_ry_agent` (tabular), `deep_agent` (MLP) | yes — `Q_table.pkl`+`features.pkl`; `mlp_save.pt` | `hung_ry_agent` **small fix**; `deep_agent` small fix **+** CUDA fix |
| **KunkelAlexander/bomberman_rl** | `38eccb6` | **MIT** | `tq_allstar`, `tq_cratehero`, `tq_coingrabber` (+4 TF CNN agents, no weights) | yes — `q_table.npz` | **small fix** (3 root modules); **trained on a 9×9 board**, see caveat |
| **nickstr15/bomberman** ("Maverick") | `fdfcd6e` | none | `maverick` | yes — `network_parameters/final_parameters.pt` | **loads and plays as-is**; only matches rule_based |
| **daryl336/CSXX46p** | `3408b84` | **MIT** | 24 folders: `ppo_final`, `dqn_torch`, `maverick*`, `llm_*`, … | yes, many | `ppo_final`/`dqn_torch` load but are weak/broken; `maverick_enhanced` small fix; `llm_*` need `requests`+an API |
| **madham97/Least_Miserables** | `0d5ffac` | none | `Ultimate_agent`, `Less_Ultimate_agent`, `Least_Ultimate_agent` | yes — `my-saved-model.pt` | loads, but weak (1.07 / 0.20) |
| **Felix-95/bomberman-NN** | `9946583` | none | `ffm_agent`, `ffm_agent_intervention`, `beginner_agent` | yes — `.pt` + `proven_models/` | `beginner_agent` loads (weak); `ffm_agent*` need **`wandb`** (not installed) |
| **Abhinand-p/Reinforcement-Learning-Bomberman** | `0caadd8` | none | `classic_1`, `novoice_agent` | yes — `Q_table-Table_20000.npy` | `classic_1` needs **`igraph`** (not installed); `novoice_agent` crashes at runtime |
| **smitha13798/bomberman_rl** | `0f86dde` | none | `dqn_age4` | yes — `latest_model.pth` | loads, **policy broken** (0.00 score, 107 invalid/round) |
| **itisacloud/GlasHoch_Rangers** | `947c12a` | none | `GlasHoch_Rangers` (DQN + imitation, 400k eps) | yes, several `.pth` | needs **`pyyaml`** (not installed) — untested, plausibly strong |
| **jepetolee/bomberman_rl** | `dd4e28d` | none | `selfplay_opponent`, `ppo_agent` | yes — `ppo_model.pt` | loads with a small fix but is **broken** (0.00, 124 invalid/round, 65 ms/step) |
| **ngmars/RL_Bomberman_game** | `e6318b4` | none | `my_agent` | yes | loads; **degenerate** (0.20 score, never bombs) |
| **JanMStraub/ML-bomberman-project** | `23f36a0` | none | `my_agent`, `the_gadget` | yes | `the_gadget` loads; **degenerate** (0.00, never bombs) |
| **stefanDeveloper/bomberman** | `ad1784f` | **MIT** | `hitchcock_full_game`, `blindfisch`, `rosa_diaz`, … | yes | loads; very weak (0.27) |
| **libaum/bomberman-rl** | `75e51ea` | none | `agent_quapsel`, `deep_quapsel` | yes | `agent_quapsel` **broken by sklearn version drift**; `deep_quapsel` needs TensorFlow |
| **thisishiu/Reinforcement-learning-Boomberman** | `19b3a46` | none | `Terminators`, `q_agent` | yes | **trained on a different game** — their `settings.py` is 13×13, `BOMB_TIMER=16`. Untested, expect garbage |
| **rahul2227/bomberman_rl_RBN** | `e62777d` | none | `Servus` | yes — `model_weights.h5` | needs **TensorFlow/Keras** (not installed); untested |
| **lukevoss/Bomberman_RL_2024** ("Echo"/"Atom") | `2c9fc2a` | none | `atom` (claimed 5.04), `echo` (claimed 5.21) | **NO — weights absent from the repo** | **unusable from this repo**; use AI-ELka's `ql` instead |
| **FreWill9/bomberman_rl** | `83cd12e` | none | `agent_fred`, `agent_fred2` | no | unusable |
| **antonH22/cnn-based-dql** | `06f245e` | none | `agent_a` | no | unusable |
| **aanhlongg/bomberman_rl** | `60c4d12` | none | — (only `fail_agent`) | no | **empty — current-semester (SS2026) peer, framework only** |
| **leanderwer/bomberman_rl** | `24ac7d2` | none | `q_learning_agent` | no | **empty shell — current-semester (SS2026) peer** |

Commit hashes are `git rev-parse --short=7 HEAD` on the clone in `scratchpad/external/`, taken
2026-08-18. Most repos were cloned `--depth 1`; that does not change the HEAD hash.

---

## Smoke-evaluation results (n = 15, liveness check only — do NOT quote)

Each row: the agent vs 3× `rule_based_agent`, `classic`, 15 rounds, `--seed 20260731`.
`rb_avg` is the mean of the three rule-based opponents *in those same games*.

| agent | repo | score | rb_avg | coins | kills | suicides | crates | invalid | ms/step (M5) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `binary_distance_agent_v2` | xiaoxiae | **6.00** | 2.64 | 3.33 | 0.53 | 0.20 | 26.6 | 2.9 | 2.7 |
| `binary_agent_v6` | xiaoxiae | **5.60** | 2.56 | 3.60 | 0.40 | 0.27 | 26.9 | 2.1 | 2.7 |
| `ql` (= Atom) | AI-ELka | **5.60** | 2.69 | 2.93 | 0.53 | 0.07 | 30.0 | 30.6 | 4.2 |
| `binary_agent_v5` | xiaoxiae | 5.13 | 2.84 | 3.13 | 0.40 | 0.40 | 21.4 | 2.9 | 2.8 |
| `feature_is_everything` | Li-Jesse-Jiaze | 4.53 | 2.71 | 3.87 | 0.13 | 0.07 | 39.7 | 6.7 | 17.2 |
| `binary_agent_v4` | xiaoxiae | 4.33 | 2.78 | 3.00 | 0.27 | 0.47 | 26.9 | 1.0 | 2.6 |
| `hung_ry_agent` | leonbegiristain | 4.07 | 2.38 | 2.73 | 0.27 | 0.67 | 33.7 | 1.7 | 34.8 |
| `tq_allstar` | KunkelAlexander | 3.80 | 2.93 | 3.13 | 0.13 | 0.80 | 31.3 | 1.9 | 0.6 |
| `maverick` | nickstr15 | 3.27 | 3.11 | 2.60 | 0.13 | 0.67 | 41.1 | 7.9 | 2.3 |
| `dql_large` | AI-ELka | 2.87 | 3.56 | 1.20 | 0.33 | 0.13 | 24.5 | 2.8 | 5.2 |
| `tq_cratehero` | KunkelAlexander | 2.27 | 3.67 | 1.93 | 0.07 | 0.73 | 20.5 | 0.3 | 0.7 |
| `daryl_ppo_final` | daryl336 | 1.60 | 4.24 | 1.27 | 0.07 | 0.60 | 21.1 | 0.9 | 10.1 |
| `beginner_agent` | Felix-95 | 1.53 | 3.16 | 1.53 | 0.00 | 0.60 | 22.3 | 13.5 | 0.8 |
| `q_agent_v3` | xiaoxiae | 1.20 | 3.36 | 0.87 | 0.07 | 0.93 | 15.3 | 9.8 | 3.0 |
| `Ultimate_agent` | madham97 | 1.07 | 3.87 | 1.07 | 0.00 | 0.67 | 17.9 | 3.0 | 0.7 |
| `q_agent_v2` | xiaoxiae | 0.33 | 4.31 | 0.33 | 0.00 | 0.60 | 9.9 | 16.7 | 2.1 |
| `hitchcock_full_game` | stefanDeveloper | 0.27 | 3.76 | 0.27 | 0.00 | 0.93 | 8.7 | 1.6 | 0.3 |
| `Least_Ultimate_agent` | madham97 | 0.20 | 3.82 | 0.20 | 0.00 | 1.00 | 6.5 | 0.1 | 0.4 |
| `ng_my_agent` | ngmars | 0.20 | 4.73 | 0.20 | 0.00 | 0.00 | 0.5 | 0.9 | 0.3 |
| `dql` | AI-ELka | 0.00 | 3.67 | 0.00 | 0.00 | 0.93 | 3.1 | 25.9 | 2.9 |
| `dqn_age4` | smitha13798 | 0.00 | 4.67 | 0.00 | 0.00 | 0.20 | 0.4 | 106.6 | 0.9 |
| `dqn_torch` | daryl336 | 0.00 | 3.27 | 0.00 | 0.00 | 1.00 | 2.8 | 16.5 | 0.1 |
| `ppo` | AI-ELka | 0.00 | 3.78 | 0.00 | 0.00 | 1.00 | 2.7 | 5.6 | 2.4 |
| `ppo_l` | AI-ELka | 0.00 | 3.89 | 0.00 | 0.00 | 0.93 | 2.5 | 45.1 | 4.8 |
| `selfplay_opponent` | jepetolee | 0.00 | 4.73 | 0.00 | 0.00 | 0.00 | 0.0 | 123.7 | 64.9 |
| `the_gadget` | JanMStraub | 0.00 | 5.42 | 0.00 | 0.00 | 0.00 | 0.0 | 17.5 | 0.2 |
| `tq_coingrabber` | KunkelAlexander | 0.00 | 3.49 | 0.00 | 0.00 | 1.00 | 2.9 | 0.0 | 0.6 |

**Timing.** `ms/step` is on the M5. The reference tournament box (one thread of a Ryzen 5 2600) is
roughly 4–6× slower, so `hung_ry_agent` at 34.8 ms/step lands near 150–200 ms and
`selfplay_opponent` at 64.9 ms/step could breach the 0.5 s limit. Everything in the top five is
comfortable. This only matters if you want a like-for-like `think_ms` comparison; as *opponents*
they run under `TRAIN_TIMEOUT`/no limit anyway.

**Reading the zero rows.** `the_gadget`, `ng_my_agent` and `selfplay_opponent` show
`suicides = 0.00` **and** `crates ≈ 0` — they never place a bomb, so they survive by doing
nothing. That is the always-`WAIT` degenerate policy `scratchpad/survey/REPORT.md` records for
source N, not a cautious agent. `dqn_torch`, `ppo`, `tq_coingrabber` show `suicides = 1.00`: they
bomb and die every round.

---

## Per-repo detail

### 1. xiaoxiae/BombermanML — best find

- URL `https://github.com/xiaoxiae/BombermanML`, commit `50b682f`, **GPL-3.0** (LICENSE present).
- Not a fork of `ukoethe/bomberman_rl`; found by code search on `"BombeRLeWorld"`.
- 15 agent folders in three families, documented in their README: `binary_agent*` (binary feature
  vectors), `binary_distance_agent*` (same but 1/distance), `distance_agent*` (all distances),
  `q_agent*` (tabular). They also ship `elo/stats.json` and `elo.py`, their own ELO harness.
- Weights: each NN variant ships `policy-model.pt` + `target-model.pt`; q-agents ship `model.pkl`.
- API: `setup(self)` / `act(self, game_state)`; relative model paths; imports `torch` and
  `matplotlib`, both installed. No `multiprocessing`. `settings.py` identical to ours on every
  constant that matters. No absolute paths anywhere.
- **Verdict: loads and plays as-is.** Verified for `binary_distance_agent_v2`, `binary_agent_v4/5/6`,
  `q_agent_v2/v3`. `distance_agent_v1` fails with the CUDA-deserialisation error (below).
- Caveat: `matplotlib` is imported at module scope in these agents, and `AGENTS.md` keeps
  matplotlib out of `requirements.txt`. Irrelevant for local evaluation; relevant if anyone ever
  tries to bundle one.

### 2. AI-ELka/BombermanRLAgents — the Atom recovery

- URL `https://github.com/AI-ELka/BombermanRLAgents`, commit `8d85731`, **no LICENSE**.
- `agent_code/ql/` is **lukevoss's Atom**. Evidence, all checkable:
  - `agent_code/ql/callbacks.py` still carries Atom's docstring verbatim, including
    *"Agent \"atom\" continiously achieves about 5.1 points per game, in the classic setting with
    3 rule based agents as opponents"*.
  - `diff lukevoss/agent_code/atom/q_learning.py AI-ELka/agent_code/ql/q_learning.py` differs on
    **two lines only**, both import-path renames (`agent_code.atom.*` → `agent_code.ql.*`).
  - `feature_extraction.py` differs only by the same rename plus a dropped `import torch`.
  - The shipped `models/q_table.pkl` is a `dict` of **329** 20-tuple states → 6 Q-values. The
    survey records source E as a 20-dim feature vector reduced to **335** states — 329 vs 335 is
    near-identical but not exact, so this is very likely the same table at a slightly different
    training checkpoint rather than the byte-identical one behind the 5.04 figure.
- Measured 5.60 here against the paper's claimed 5.04 — same ballpark, and n = 15 easily spans it.
- Their own `dql` / `ppo` / `ppo_l` are broken (score 0.00); `dql_large` scores 2.87, below
  rule_based.
- **Verdict for `ql`: loads and plays as-is.** Uses absolute package imports
  (`from agent_code.ql.utils import *`), so **the folder must keep the name `ql`** — I broke it
  once by renaming it to `aielka_ql`. Same constraint applies to `dql`, `dql_large`, `ppo`, `ppo_l`.
- 30.6 invalid moves per round is Atom's own behaviour, not a porting artefact.

### 3. Li-Jesse-Jiaze/MLE_project_bomberman — 3rd place SS2024

- URL `https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman`, commit `a7fe504`, **no LICENSE**.
- README: "🥉We won third place in 24ss." Their submission is `agent_code/feature_is_everything`
  (their per-agent README: 5-tile one-hot features + a bomb digit + best-target slot, Dueling DQN
  with a pure MLP). `dqn` and `dqn_5tile` are earlier iterations.
- Weights present in all three (`my-saved-model.pt`, 30–87 KB), loaded by **relative** path
  `open("my-saved-model.pt")` — correct for our `agents.py:305` chdir.
- `features.py` reads exactly `field`, `bombs`, `explosion_map`, `coins`, `self`, `others`, and
  `import settings as s`. No absolute paths, no `multiprocessing`. Imports `torch` (installed).
- Their `environment.py` differs from ours by **one added log line**; `game_state` is unchanged.
- Their `settings.py` has `MAX_STEPS = 200` (ours 400) and altered `loot-crate`/`coin-heaven`
  densities — but **`classic` is identical to ours** (`CRATE_DENSITY 0.75`, `COIN_COUNT 9`). The
  agent was trained under a 200-step cap and is being run here under 400; that is a fairness
  caveat in its favour, not against it.
- **Verdict: `feature_is_everything` loads and plays as-is.** `dqn` and `dqn_5tile` fail with:

  ```
  RuntimeError: Attempting to deserialize object on a CUDA device but torch.cuda.is_available() is False.
  ```

  **The fix (not applied):** their `callbacks.py` does
  `self.policy_net = pickle.load(file).to(self.device)`. Replace the `pickle.load` with
  `torch.load(file, map_location="cpu", weights_only=False)`. `feature_is_everything` does not
  need this because its checkpoint was saved from CPU.

### 4. KunkelAlexander/bomberman_rl — the survey's source K, MIT

- URL `https://github.com/KunkelAlexander/bomberman_rl`, commit `38eccb6`,
  **MIT** ("Copyright (c) 2025 Alexander Kunkel"). The only permissively licensed strong-ish repo.
- The blog posts in `scratchpad/survey/REPORT.md` (source K) belong to this repo.
- Tabular agents with weights: `tq_allstar` (`q_table.npz`, 1.4 MB), `tq_cratehero` (439 KB),
  `tq_coingrabber` (181 KB). `tq_representator` is the "representator" control from the survey —
  **no weights**, and by construction it does not need a table. The four TensorFlow CNN agents
  (`cnn_allstar*`, `dqn_allstar_duel`) ship **no weights** and TF is not installed.
- **Verdict: small fix.** `agent_code/tq_*/callbacks.py` does `from q_tabular_agent import ...`
  and `from q_helpers import ...` — those modules live at the **repo root**, not in the agent
  folder. Fix: copy `q_helpers.py`, `q_tabular_agent.py`, `q_agent_parent.py` from the repo root
  onto `sys.path` (I put them in the testbed root). With that they load and play.
- **Board-size caveat, important:** their `settings.py` sets `COLS = ROWS = 9`. Their tables were
  trained on a **9×9** board. Their features are local (18-bit neighbour/danger/direction code)
  and `q_helpers.py` imports only `BOMB_POWER`, `BOMB_TIMER`, `EXPLOSION_TIMER` from settings — all
  identical to ours — so nothing breaks on 17×17, and `tq_allstar` still scores 3.80. But this is
  an out-of-distribution transfer, so it is *not* a fair reproduction of their published 6.3.
- Their `environment.py` diff (79 lines) is entirely GUI/debug/video code plus a
  `name.startswith("tq_")` branch in the renderer. `game_state` is untouched.
- `tq_coingrabber` is a task-1 coin agent: in `classic` it suicides in **every** round. Useless
  as an opponent but a perfect illustration of the `AGENTS.md` "keep the action set in step with
  the features" rule.

### 5. leonbegiristain/RL_Bomberman

- URL `https://github.com/leonbegiristain/RL_Bomberman`, commit `2b4c799`, **no LICENSE**.
- `hung_ry_agent` (tabular; ships `Q_table.pkl` + `features.pkl`) and `deep_agent` (MLP;
  `mlp_save.pt`). There is also `agent_code/base_agent/`, a shared library folder with **no
  `callbacks.py`** — it is not an agent.
- **Verdict: small fix.** Both do `from agent_code.base_agent... import ...`, so
  `agent_code/base_agent/` must be copied alongside them. With that, `hung_ry_agent` runs and
  scores 4.07. `deep_agent` additionally hits the same **CUDA deserialisation** error and needs
  the `map_location="cpu"` fix.
- `hung_ry_agent` costs 34.8 ms/step — the slowest thing in the usable set.

### 6. nickstr15/bomberman — "Maverick", the survey's source M

- URL `https://github.com/nickstr15/bomberman`, commit `fdfcd6e`, **no LICENSE**. Ships
  `MaverickReport.pdf`. 327 MB clone (full history, gifs).
- `agent_code/maverick/` with `network_parameters/final_parameters.pt`.
- **Verdict: loads and plays as-is** in our framework, despite the repo being from **2021** with a
  heavily modified framework of its own (`environment.py` 429 diff lines, `main.py` 197,
  `agents.py` 69, `items.py` 37). The `game_state` contract evidently did not drift.
- Scores 3.27 vs `rb_avg` 3.11 — consistent with the survey's "3.2, below rule_based". A useful
  *calibration* opponent (a known-weak but non-degenerate learned agent), not a threat.

### 7. lukevoss/Bomberman_RL_2024 — the headline that does not run

- URL `https://github.com/lukevoss/Bomberman_RL_2024`, commit `2c9fc2a`, **no LICENSE**.
  Ships `Echo_Report_Müller_Tiedl_Voss.pdf` (the survey's source E).
- `agent_code/atom/callbacks.py` sets `pretrained_model="atom.pkl"` and
  `q_learning.py` loads `os.path.join('./models', model_name)` — a correct **relative** path.
  `agent_code/atom/models/` contains **only `.gitkeep`**. `agent_code/echo/` has no models
  directory at all and `ppo.py` raises if `./models/echo.pt` is missing.
- Confirmed by running it in the testbed: `FileNotFoundError: Pretrained model at
  ./models/atom.pkl not found.`, raised from `q_learning.py:30` during `setup()`. Credit where due
  — their loader raises rather than falling back to an empty table, so it fails loudly instead of
  producing a plausible-looking but meaningless opponent.
- **Verdict: unusable from this repo.** Use `AI-ELka/BombermanRLAgents` → `agent_code/ql` instead.
  Their README says the training *dataset* is too large for GitHub and to email the authors; the
  weights appear to have been left out by the same oversight.

### 8. Everything else, briefly

- **daryl336/CSXX46p** (`3408b84`, **MIT**, 355 MB): 24 agent folders including LLM-driven ones
  (`llm_*`, `pure_llm_agent`) that call out over `requests`/`google` — not installed, and an
  external API call would blow the 0.5 s budget anyway. `ppo_final` (1.60) and `dqn_torch` (0.00)
  load but are weak. `maverick_enhanced` needs a root-level `metrics.py` on `sys.path`
  (same class of fix as KunkelAlexander). Their `settings.py` has `TIMEOUT = 180`.
- **itisacloud/GlasHoch_Rangers** (`947c12a`, no licence, 136 MB): DQN with imitation pretraining,
  checkpoints at 400 000 episodes, config-driven. Fails on `ModuleNotFoundError: No module named
  'yaml'`. **This is the one untested candidate I would guess is strong** — big training budget,
  imitation scaffolding, and the survey notes that is exactly what makes deep agents work here.
  Needs `pyyaml` (I did not install it).
- **Felix-95/bomberman-NN** (`9946583`): `ffm_agent` and `ffm_agent_intervention` import `wandb`
  at module scope → `ModuleNotFoundError`. `beginner_agent` runs but scores 1.53.
- **Abhinand-p** (`0caadd8`): `classic_1` needs `igraph` (not installed; `networkx` is).
  `novoice_agent` loads but crashes in `act()` with
  `ValueError: operands could not be broadcast together with shapes (17,) (2,)` — a genuine bug
  in their feature code, not a porting issue.
- **libaum/bomberman-rl** (`75e51ea`): `agent_quapsel` unpickles a scikit-learn model and dies on
  `ModuleNotFoundError: No module named 'sklearn.ensemble._gb_losses'` — that private module was
  removed in sklearn 1.2. Would need an old sklearn pinned in a separate venv. Not worth it.
- **thisishiu/Reinforcement-learning-Boomberman** (`19b3a46`): their `settings.py` is
  `COLS = ROWS = 13`, `BOMB_TIMER = 16`, `TIMEOUT = 1.5`. Trained on a materially different game.
  Not tested; I would not trust any number from it.
- **rahul2227/bomberman_rl_RBN** (`e62777d`): `Servus` needs TensorFlow/Keras. Untested.
- **smitha13798** (`0f86dde`): `dqn_age4` loads, scores 0.00 with 106.6 invalid moves/round — the
  policy is broken.
- **jepetolee/bomberman_rl** (`dd4e28d`): `selfplay_opponent` needs `agent_code/ppo_agent/` copied
  alongside it; with that it runs, but scores 0.00 with 123.7 invalid moves/round at 65 ms/step.
- **ngmars/RL_Bomberman_game** (`e6318b4`, pushed 2026-08-16) and
  **JanMStraub/ML-bomberman-project** (`23f36a0`): both load, both never place a bomb.
- **FreWill9**, **antonH22**: code only, no weights.
- **aanhlongg/bomberman_rl** (2026-08-15) and **leanderwer/bomberman_rl** (2026-08-13): the two
  **current-semester** forks of upstream. Both are the bare framework —
  `leanderwer`'s single commit is "Add initial Q-learning agent template" and has no weights;
  `aanhlongg` has no agent folder at all. Nothing to test yet. **Worth re-checking in September**,
  since they are our actual tournament cohort.

---

## Compatibility issues seen, and the exact fixes

My scan (`analyze.py`, regex over every `.py` in every non-provided agent folder for
`/Users`, `/home`, `/mnt`, `/content`, `C:\`) found **no absolute path** in any of the 24 repos —
the classic crash `AGENTS.md` warns about did not occur once. The real failure modes were these five:

| # | symptom | affected | fix (NOT applied) |
|---|---|---|---|
| 1 | `RuntimeError: Attempting to deserialize object on a CUDA device…` | Li-Jesse `dqn`, `dqn_5tile`; leonbegiristain `deep_agent`; xiaoxiae `distance_agent_v1` | swap the `pickle.load(f)` / `torch.load(f)` in `setup()` for `torch.load(f, map_location="cpu", weights_only=False)` |
| 2 | `ModuleNotFoundError` for a helper that lives at **their repo root** | KunkelAlexander `tq_*` (`q_helpers`, `q_tabular_agent`, `q_agent_parent`); daryl336 `maverick_enhanced` (`metrics`) | copy those root-level `.py` files somewhere on `sys.path`, or into the agent folder and make the imports relative |
| 3 | `ModuleNotFoundError: No module named 'agent_code.<sibling>'` | leonbegiristain (`base_agent`); jepetolee (`ppo_agent`); Li-Jesse/xiaoxiae internally | copy the sibling folder too. **And keep the folder name** — several agents use absolute `agent_code.<name>.*` imports, so renaming the folder breaks them |
| 4 | missing third-party package | `wandb` (Felix-95), `igraph` (Abhinand-p), `pyyaml` (itisacloud), TensorFlow/Keras (KunkelAlexander CNNs, rahul2227, libaum `deep_quapsel`), `requests`/`google` (daryl336 `llm_*`) | would need installing — **I installed nothing**. Already present and usable: `numpy`, `scipy`, `sklearn`, `torch`, `networkx`, `matplotlib` |
| 5 | stale pickle against a newer library | libaum `agent_quapsel` (`sklearn.ensemble._gb_losses`, removed in sklearn ≥ 1.2) | separate venv with an old sklearn; not worth it |

Also worth knowing: **no agent folder in any of the 24 repos imports `multiprocessing`**, and
their `items.py` is byte-identical to ours in every case except nickstr15 (2021).

---

## Recommended order to try

The reasoning: measure the ones that already run first, spend fix-effort only where the payoff is
a *distinct* opponent archetype, and stop bothering with anything that scores 0.

**Tier 1 — measure these properly with `tools/evaluate.py` (all run as-is, no fixes):**

1. **`xiaoxiae/binary_distance_agent_v2`** (6.00) — strongest measured, 2.7 ms/step, GPL-3.0 so
   the licence is clean, and the repo ships a sibling family (`binary_agent_v4/5/6`) that gives
   you a *graded* difficulty ladder from one codebase. Start here.
2. **`AI-ELka/ql`** (5.60) — this is the survey's source E "Atom", the exact agent our prior-work
   survey benchmarked at 5.04. Measuring against it is the single most quotable comparison
   available, because there is a published number and a 47-page report to cite next to it.
   Keep the folder named `ql`.
3. **`xiaoxiae/binary_agent_v6`** (5.60) — a second, architecturally different top agent.
4. **`Li-Jesse-Jiaze/feature_is_everything`** (4.53) — the 3rd-place SS2024 submission; the
   closest thing to a "typical strong course submission" in the set, and it plays a visibly
   different game (3.87 coins, 39.7 crates, almost no kills — a farmer, not a hunter).

That is four opponents spanning 4.5–6.0, all zero-fix, all under 20 ms/step. It is enough for a
real opponent-diversity result and I would not go further before measuring these.

**Tier 2 — one small fix each, worth it for archetype diversity:**

5. **`KunkelAlexander/tq_allstar`** (3.80, fix #2) — MIT, 0.6 ms/step, and *tabular* like ours,
   which makes it the most directly comparable opponent in the set. Its 9×9 training makes it an
   out-of-distribution transfer; say so if you report it.
6. **`leonbegiristain/hung_ry_agent`** (4.07, fix #3) — a fourth-quartile-of-Tier-1 tabular agent.
7. **`itisacloud/GlasHoch_Rangers`** (untested, needs `pyyaml`) — **the highest-variance option**.
   400k-episode DQN with imitation pretraining. If anything unmeasured here beats Tier 1, it is
   this. Costs one `uv add pyyaml`, which I deliberately did not do.

**Tier 3 — calibration only:**

8. **`nickstr15/maverick`** (3.27, zero fix) — a *known-weak* learned agent, useful as a floor:
   an opponent our agent should beat comfortably, which is a different kind of evidence from
   beating a strong one.

**Do not bother:** everything scoring 0.00, the two 2026 peer forks (empty), lukevoss (no
weights), thisishiu (different game), libaum (sklearn drift), and anything needing TensorFlow.

---

## Would any of these beat us?

Our shipped rung-4 table measures **3.949** [paired +0.255 vs the previous version] against
3× `rule_based_agent` (`experiments/benedict_task4.md`), with the 1000-round noise floor at ±0.12.

Four external agents scored above that here: 6.00, 5.60, 5.60, 4.53. **At n = 15 those are not
results** — but the top three sit 1.6–2.0 points clear, which is more than an order of magnitude
above the noise floor, and two of them (Atom at a published 5.04, `feature_is_everything` as a
placed submission) have independent corroboration. The honest reading is: **yes, very probably
three of these beat us, and the gap is large.**

Two things blunt that, and both should be checked before anyone panics:

- **All of these were measured against 3× `rule_based_agent`, never against each other.** A field
  of one strong agent plus three rule-based ones is not the tournament. Running Tier 1 in a
  *mixed* field is the measurement that actually predicts standing.
- **The comparison is not perfectly matched.** These numbers come from `main.py --seed`, whereas
  our 3.949 comes from `evaluate.py`'s per-round reseeding over 1000 rounds. Re-measure the Tier-1
  four through `evaluate.py` at the standard seed before writing any of this down.

The constructive read: `xiaoxiae/binary_distance_agent_v2` and `AI-ELka/ql` are both **tabular or
small-feature** agents, not big networks. Whatever they are doing better is in the feature vector,
which is exactly the lever `scratchpad/survey/REPORT.md` already identified and exactly the kind
of thing that can be read out of a 329-entry Q-table by hand.

---

## Reproducing / cleaning up

- Clones: `scratchpad/external/<owner>__<repo>/` (24 repos, ~1.1 GB total).
- Testbed: `scratchpad/external/testbed/` — framework copy + `agent_code/` with every candidate.
  Re-run one agent with the command in the "How the load tests were done" section.
- Raw smoke stats: `scratchpad/external/testbed/stats/*.json` (plus `*.err` for the failures).
- Scanning scripts: `analyze.py` (per-repo static scan), `scan.py` (GitHub-wide sweep),
  `repos.txt` (158 candidate repos), `scan.tsv` (sweep results: pushed date, licence, weight files).
- **`scratchpad/external/` must stay untracked** — 20 of 24 repos have no licence.
