# Fresh-session prompt — Docker submission test for `benedict_task4`

Copy everything below the line into a new session.

---

You are working on a Bomberman RL university project at
`/Users/benedictvonschubert/Projects/bomberman_RL`. Read `AGENTS.md` first — it is the project
spec and it binds you. `CLAUDE.md` adds tool-specific notes.

**Your one job this session: verify that `agent_code/benedict_task4/` runs correctly inside the
provided `Dockerfile`, exactly as the tournament will run it.** This has never been done. The code
deadline is **21.09.2026** and the agent is otherwise finished, so this is the last gate before
submission.

## The workflow — follow this exactly

**Benedict runs everything and commits everything. You never touch git.**

1. **Never** run `git add` / `git commit` / `git push`, and never edit `.gitignore` — propose, don't
   apply.
2. Never edit the framework files (`environment.py`, `agents.py`, `items.py`, `settings.py`,
   `main.py`); they are reset to upstream for the tournament.
3. `agent_code/benedict_task4/` was cleaned and verified on 2026-08-21 and is **frozen**. If the
   Docker test shows it must change, say so and propose the diff — do not edit it unless Benedict
   asks in this session.
4. You own `scratchpad/`, `tools/`, `experiments/*.md`, `AGENTS.md`, `MEASUREMENT.md`.
5. Write your findings to `scratchpad/benedict/docker/` as you go, not at the end.

## What ships, and what must not

- The submission is a zip of **`agent_code/benedict_task4/` alone**, named
  `final-project-agent-code.zip`.
- It must contain exactly: `callbacks.py`, `train.py`, `q_table.npy`, `q_table.npy.layout.json`,
  `README.md`. **Nothing else** — no `__pycache__`, no `logs/`, no `q_table_trained.npy`, no
  checkpoints.
- `q_table.npy` is the submitted model, md5 **`54d63bc79179fdf80d3b9bfb80f21461`**, byte-identical
  to `checkpoints/benedict_task4/q_table_e37_PLB2_s106__ep20000.npy`. **Verify this before and
  after anything you do.** If it ever differs, stop and say so.
- `agent_code/ext_*/` are third-party agents used for measurement. **They must never enter the zip**
  — 20 of their 24 source repos carry no licence. They are gitignored; check the zip anyway.

## The tests, in order

**1 · Build.** `docker build -t bomberman .` from the repo root. It is a large image
(conda + pytorch + tensorflow); expect a long first build and cache it. Report the build time and
any warnings that matter.

**2 · The agent runs at all, in the container, with no `tools/` and no `uv`.** The tournament has
neither. Run the *provided* entry point, not ours:

```
python main.py play --agents benedict_task4 rule_based_agent rule_based_agent rule_based_agent \
    --scenario classic --n-rounds 10 --no-gui
```

Confirm no traceback, and confirm from the output that the agent actually scores rather than
standing still. `callbacks.py` imports only `os`, `numpy` and the framework's `settings`; if
anything else is required, that is a submission bug.

**3 · The zip, in isolation — this is the test that matters.** Build the zip, unpack it into a
*clean* checkout of the upstream framework inside the container (no `tools/`, no `checkpoints/`,
no `scratchpad/`, no `results/`), and run the same command. This catches the classic failure that
the project has hit before: a path that resolves only because a development directory happened to
exist. `agents.py:305` chdirs into the agent's own folder each round, so relative paths work and
absolute ones break.

**4 · Timing against the real limit.** `settings.py` sets `TIMEOUT = 0.5 s` per step, and
`AGENTS.md` records the reference hardware as **one thread of an AMD Ryzen 5 2600, ≤8 GB RAM** —
far slower than this machine. Measure `think_max_ms` inside the container, ideally with the
container CPU-limited (`--cpus=1`), and compare against the 0.5 s budget. The agent's cost is
dominated by two BFS traversals over a 17×17 board; `README.md` claims mean 0.137 ms and a worst
observed 53.4 ms across 180 000 rounds on the dev machine. **Verify or correct that claim** — it is
in a document that goes to the graders.

**5 · Training inside the container.** `--train 1` must work with **no environment variables**:

```
python main.py play --agents benedict_task4 rule_based_agent rule_based_agent rule_based_agent \
    --scenario classic --train 1 --n-rounds 5 --no-gui --seed 810731
```

It should write `q_table_trained.npy` beside the agent and **leave `q_table.npy` untouched**.
Note that it warm-starts from `checkpoints/benedict_task4/q_table_parent.npy`, which is *outside*
the zip — so inside the isolated zip test this will fail on a missing parent. **That is expected and
correct**: training is not part of the submission. Confirm it fails *loudly* rather than silently
training from zero, and record the behaviour.

## Things this project has learned the hard way — do not repeat them

- **A symlinked framework is not a test.** An earlier session symlinked the framework into a
  testbed; Python resolved `sys.path[0]` through the symlink back to the project root and every
  agent silently loaded the *project's* `agent_code/`. Thirteen agents "passed" and every pass was
  fake. **Copy, never symlink.**
- **Missing files must fail loudly.** `callbacks.py` deliberately raises `FileNotFoundError` when
  the table is absent, because ten evaluations once silently measured an all-zero table's
  random policy. If you see the agent playing badly, check that it loaded a table before concluding
  anything.
- **Verify before you report.** Six audits have run on this project and all six overturned a claim
  the author was confident about. If a check passes, say what you actually ran.

## Deliverable

`scratchpad/benedict/docker/REPORT.md`, created as your **first** action with headings stubbed and
appended to as you go — agents on this project have previously reported paths they never created.

Cover: build result and time; each of the five tests with the exact command and its output;
`think_max_ms` measured, and whether `README.md`'s timing claim survives; the exact file list of
the zip; and a final **GO / NO-GO for submission** with the specific blocker if it is NO-GO.

Then give Benedict the `git` commands as a copy-paste block, and the exact command to build the
final `final-project-agent-code.zip`.
