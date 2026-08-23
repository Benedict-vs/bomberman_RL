# Docker submission gate — `benedict_task4` with the E51 death filter ON by default

Re-run of the 2026-08-21 gate after E51 flipped `BM_DEATH_FILTER`'s default to on,
which is a change to the shipped policy and therefore requires the gate again.
Date 2026-08-23. Working tree = the E51 result commit plus the default flip.

Images built from `Dockerfile.pinned`. The repo `Dockerfile` still fails at the
TensorFlow line because `FROM continuumio/miniconda3` is untagged and now
resolves to Python 3.14 — the course's file and the course's dependency, not
ours; our agent needs only numpy, installed two lines earlier. Repo `Dockerfile`
untouched.

**`q_table.npy` md5 `54d63bc79179fdf80d3b9bfb80f21461` before and after every
test below. E51 trained nothing; the model is untouched.**

## 1 · Build — PASS

`bomberman` (dev repo, 9.31 GB) and `bomberman_clean` (pristine upstream +
the unzipped submission, 8.97 GB) both build from `Dockerfile.pinned`.
Rebuilt after the README edit so the images match what ships.

## 2 · The agent runs in the container — PASS

`python main.py play --agents benedict_task4 rule_based_agent ×3 --scenario classic
--n-rounds 20 --no-gui --save-stats`, dev-repo image, no `tools/`, no `uv`:

| agent | score | coins | kills | suicides | invalid |
|---|---|---|---|---|---|
| **benedict_task4** | **91** | 56 | 7 | 9 | 106 |
| rule_based_agent_0 | 56 | 36 | 4 | 9 | 122 |
| rule_based_agent_1 | 74 | 59 | 3 | 6 | 164 |
| rule_based_agent_2 | 49 | 29 | 4 | 12 | 161 |

## 3 · The zip in isolation — PASS (the test that matters)

Zip unpacked into a `git archive d7eed90` checkout — copied, never symlinked —
with no `tools/`, no `checkpoints/`, no `scratchpad/`, no `results/`:

| agent | score | coins | kills | suicides | invalid |
|---|---|---|---|---|---|
| **benedict_task4** | **81** | 61 | 4 | 9 | 89 |
| rule_based_agent_0 | 65 | 35 | 6 | 12 | 113 |
| rule_based_agent_1 | 75 | 50 | 5 | 7 | 134 |
| rule_based_agent_2 | 64 | 34 | 6 | 8 | 157 |

20 rounds is far too few to rank agents — the point is that it plays. **And the
table was demonstrably loaded**, read from the agent's own log inside the
container rather than inferred from the score:

```
[benedict_task4_code] INFO: Loading Q-table from disk.
```

**The filter is live in the tournament configuration.** With no environment
variables set, in the clean image:

```
DEATH_FILTER = 2   (2 = full filter)
numpy 2.2.5
```

## 4 · Timing against the 0.5 s limit — PASS

Clean image, `--cpus=1` (the tournament reference is one thread of a Ryzen 5
2600), 40 rounds, 11 206 decisions, measured with the framework's own
`note_stat` instrumentation:

| | ms |
|---|---|
| mean | 0.222 |
| median | 0.096 |
| p99 | 0.749 |
| p99.9 | 0.919 |
| **max** | **2.588** |
| limit | 500 |
| **over limit** | **0** |
| headroom | **193×** |

The filter costs roughly 0.02 ms of mean and runs at all only on the 33.6 % of
steps with a bomb or fire on the board. The previous gate's worst step was
9.33 ms without the filter, so the tail here is not worse for having it.

## 5 · Training in the container, no environment variables — PASS

**5a, clean image (no warm parent, as the zip ships).** Fails loudly and writes
nothing:

```
FileNotFoundError: [Errno 2] No such file or directory:
  '/home/bomberman/agent_code/benedict_task4/../../checkpoints/benedict_task4/q_table_parent.npy'
```

Directory afterwards is unchanged and `q_table.npy` is still `54d63bc7…`.
This is correct: training is not part of the submission.

**5b, dev image with the parent mounted.** Exit 0, writes
`q_table_trained.npy` (+ its layout sidecar), and **`q_table.npy` is still
`54d63bc7…`** — a stray `--train` cannot destroy the submitted model.

## Zip contents — exactly five files

```
benedict_task4/callbacks.py
benedict_task4/train.py
benedict_task4/q_table.npy
benedict_task4/q_table.npy.layout.json
benedict_task4/README.md
```

No `__pycache__`, no `logs/`, no `q_table_trained.npy`, no checkpoints, and no
`agent_code/ext_*` (checked explicitly — 20 of their 24 source repos carry no
licence). `final_project.pdf` §8: the grader searches for the first directory
containing `callbacks.py`, which this matches.

## Verdict: **GO**

The agent runs from the zip alone in the provided image, loads the trained
table, plays a full `classic` game against three `rule_based_agent`s, clears the
timing limit by 193× on a single CPU, and cannot overwrite its own model.

Build the submission with:

```bash
bash scratchpad/benedict/docker/rebuild.sh
# -> scratchpad/benedict/docker/final-project-agent-code.zip
```
