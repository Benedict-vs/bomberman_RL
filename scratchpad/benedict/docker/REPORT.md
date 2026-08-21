# Docker submission gate — `agent_code/benedict_task4/`

Session date: 2026-08-21. Purpose: verify the agent runs in the provided `Dockerfile`
exactly as the tournament will run it. Written incrementally as the session runs.

## 0 · Pre-flight (integrity of the shipped table)

Run before touching anything:

```
$ md5 -q agent_code/benedict_task4/q_table.npy
54d63bc79179fdf80d3b9bfb80f21461
$ md5 -q checkpoints/benedict_task4/q_table_e37_PLB2_s106__ep20000.npy
54d63bc79179fdf80d3b9bfb80f21461
```

Both match the expected `54d63bc79179fdf80d3b9bfb80f21461`. **PASS.**
Re-checked after every stage below; see §6.

### Static checks on the frozen agent

| Check | Result |
|---|---|
| `callbacks.py` imports | `os`, `numpy`, `settings as s` — nothing else. **PASS** |
| `settings` constants used | `s.BOMB_POWER`, `s.BOMB_TIMER` only; both upstream. **PASS** |
| Absolute paths | none. `_AGENT_DIR = os.path.dirname(__file__)` (`callbacks.py:57`), `MODEL_FILE` joined onto it. **PASS** |
| `train.py` importing `tools/` | guarded (`try: from tools.trainlog import TrainLogger / except ImportError`, `train.py:113-115`). **PASS** |
| Framework modified? | only `settings.py`, and only the `BM_QUIET_LOGS` log-level switch whose *unset* default is byte-equal to upstream. `main.py`, `environment.py`, `agents.py`, `items.py`, `events.py`, `fallbacks.py` untouched since the import commit `d7eed90`. **PASS** |

### `logs/` is not in the zip — verified safe

`agents.py:226-227` builds the path and creates it itself:

```python
log_dir = f'agent_code/{self.code_name}/logs/'
if not os.path.exists(log_dir): os.makedirs(log_dir)
```

So shipping without `logs/` cannot crash the wrapper. (Relevant because the
course's pre-run explicitly returns "the logs printed to your agent's logs
folder".)

### What `final_project.pdf` actually specifies (§8, p. 11)

Read verbatim rather than from memory, because it changes two of the tests:

> 1. Unzip the submitted file.
> 2. Install libraries specified in `requirements.txt`.
> 3. Search for the **first directory in the unzipped files that contains a
>    `callbacks.py`**.
> 4. Copy this directory to our `agent_code`.
> 5. Run a single game with `self.train = False` **against three
>    `random_agent`s**.

Consequences:

- The zip must contain a **directory** holding `callbacks.py`, not the five
  files at the archive root. The build command in §7 does that
  (`benedict_task4/` at top level).
- The pre-run opposes **three `random_agent`s**, not `rule_based_agent`. That
  exact configuration is run as test 2b.
- `requirements.txt` is expected *inside* the zip. Ours lives at the repo root
  and the zip is specified as exactly five files. This is **not a blocker** —
  the agent needs only `numpy`, which the base image already has, so an absent
  `requirements.txt` installs nothing that was needed. Flagged for Benedict's
  decision in §7.

## 1 · Build

### 1a · Build context

The repo root is **19 GB** (`logs/` 10 G, `checkpoints/` 3.7 G, `scratchpad/`
2.0 G, `results/` 1.8 G) and there is **no `.dockerignore`**, so a literal
`docker build .` from the repo root uploads all 19 GB into the daemon and bakes
it into the image. That tests nothing about the agent — the graders' context is
the upstream framework plus our unzipped folder.

So the image was built from a **178 MB context** holding the framework, docs and
our own agents, with the **unmodified `Dockerfile`**:

```
rsync -a --exclude .git --exclude .venv --exclude logs --exclude checkpoints \
  --exclude scratchpad --exclude results --exclude replays --exclude screenshots \
  --exclude __pycache__ --exclude '*.pyc' --exclude 'agent_code/ext_*' \
  --exclude 'agent_code/*/logs' --exclude final_project.pdf \
  <repo>/ scratchpad/benedict/docker/ctx_repo/
```

`agent_code/ext_*` were excluded from the context as well, so the unlicensed
third-party agents cannot reach the image. `q_table.npy` in the context still
md5s to `54d63bc79179fdf80d3b9bfb80f21461`.

A `.dockerignore` is **proposed, not applied** — see §7.

### 1b · The provided `Dockerfile` does not build today — FAIL

```
$ docker build --progress=plain -t bomberman .
...
 > [ 7/11] RUN pip install scikit-learn tqdm tensorflow keras tensorboardX xgboost lightgbm:
0.818 ERROR: Could not find a version that satisfies the requirement tensorflow (from versions: none)
0.818 ERROR: No matching distribution found for tensorflow
ERROR: failed to build: failed to solve: process "/bin/sh -c pip install scikit-learn tqdm
tensorflow keras tensorboardX xgboost lightgbm" did not complete successfully: exit code: 1
```

Failed after **53 s**, at Dockerfile line 7.

Cause — the base image is unpinned and has moved:

```
$ docker run --rm continuumio/miniconda3 python -VV
Python 3.14.6 | packaged by Anaconda, Inc. | (main, Jun 18 2026, 21:31:22) [GCC 14.3.0]
```

`FROM continuumio/miniconda3` (no tag) now resolves to **Python 3.14**, and
TensorFlow publishes no cp314 wheel — hence `from versions: none`. The same base
also prints a deprecation notice: *"Updates to this image will be discontinued
after Miniconda version 26.7.x. The latest Miniconda Docker images are available
as `anaconda/miniconda`."*

**This is the course's `Dockerfile`, not our agent.** Nothing our agent imports
is involved: it needs `numpy` only, and `numpy` installs two lines earlier via
`conda`. The failing line is the TensorFlow/Keras line that exists for teams
doing deep learning. It is a genuine finding to pass on, but it is **not a
blocker for this submission** — and per §8 of the task PDF the graders maintain
the file (*"please git pull to get the newest version of the Dockerfile"*), so
the fix is theirs to make.

### 1c · Build with the base pinned — the image used for all tests below

One-line change, made **only** in `scratchpad/benedict/docker/ctx_repo/Dockerfile.pinned`;
the repo's `Dockerfile` is untouched:

```diff
-FROM continuumio/miniconda3
+FROM continuumio/miniconda3:25.1.1-2
```

`25.1.1-2` is the newest tag still on Python 3.12 (3.12.9), which TensorFlow has
wheels for. Result: **build succeeded in 151 s**, image **2.92 GB**. TensorFlow
resolved `cp312` aarch64 wheels and installed cleanly. The only Docker warning is
cosmetic: `JSONArgsRecommended: JSON arguments recommended for CMD` (line 12).

**All tests below run in this image.** Two tags exist:

| tag | `/home/bomberman` contains |
|---|---|
| `bomberman` | framework + all our agents (the dev-repo context) |
| `bomberman_clean` | upstream framework at `d7eed90` + **only** the unzipped submission |

**Architecture caveat.** This machine is Apple silicon, so the image is
`linux/arm64`; the tournament reference machine is an AMD Ryzen 5 2600
(`linux/amd64`). For *correctness* this is irrelevant — the agent is pure
NumPy + stdlib. For *timing* it matters, and §4 handles it explicitly.

## 2 · Agent runs in the container (provided entry point, no `tools/`, no `uv`)

### First attempt crashed — my error, not the agent's

```
FileNotFoundError: [Errno 2] No such file or directory: '/home/bomberman/logs/game.log'
  File "/home/bomberman/environment.py", line 61, in setup_logging
    handler = logging.FileHandler(f'{self.args.log_dir}/game.log', mode="w")
```

`environment.py:61` opens the *game* log without creating `logs/` first (unlike
`agents.py:227`, which does create the *agent* log dir). Upstream ships
`logs/.gitkeep` so the directory always exists in a real checkout; my rsync had
excluded `logs/`. Fixed by restoring `logs/.gitkeep` in the context and
rebuilding. **Recorded because it is a genuine framework property** — a
tournament checkout without `logs/` cannot start any game, for any agent. It is
not a property of our submission.

### 2a · `benedict_task4` vs 3 × `rule_based_agent`, `classic`, 20 rounds

```
docker run --rm bomberman \
  python main.py play --agents benedict_task4 rule_based_agent rule_based_agent rule_based_agent \
    --scenario classic --n-rounds 20 --no-gui --save-stats /tmp/s.json
```

Exit **0**, no traceback.

| agent | score | coins | crates | kills | suicides | invalid |
|---|---|---|---|---|---|---|
| **benedict_task4** | **84** | 59 | 718 | 5 | 10 | 100 |
| rule_based_agent_0 | 44 | 39 | 613 | 1 | 8 | 139 |
| rule_based_agent_1 | 53 | 43 | 576 | 2 | 12 | 113 |
| rule_based_agent_2 | 59 | 39 | 542 | 4 | 14 | 146 |

4.2 score/round, against the README's held-out 3.949 — consistent, and well
clear of the ±0.12 noise floor discussion at this sample size. The agent is
plainly *playing*: 4 497 moves, 598 bombs, 718 crates, 5 kills.

### 2b · The course's actual pre-run configuration — 3 × `random_agent`

`final_project.pdf` §8 step 5 says the pre-run is *"a single game with
`self.train = False` against three `random_agent`s"*, so that case is run
explicitly rather than assumed:

| agent | score | coins | crates | suicides | invalid |
|---|---|---|---|---|---|
| **benedict_task4** | **154** | 154 | 1990 | **0** | 8 |
| random_agent_0 | 0 | 0 | 47 | 20 | 205 |
| random_agent_1 | 1 | 1 | 57 | 20 | 195 |
| random_agent_2 | 0 | 0 | 68 | 20 | 216 |

7.7 score/round and **zero suicides across 20 rounds**. Exit 0, no traceback.

**Test 2: PASS.** Confirmed `callbacks.py` needs nothing beyond `os`, `numpy`
and the framework's `settings` — no `tools/`, no `uv`, no extra library.

## 3 · The zip in isolation (clean upstream checkout)

### How the isolated tree was built — copied, never symlinked

```
git -C <repo> archive d7eed90 | tar -x -C clean/     # upstream framework, verbatim
rm -f clean/final_project.pdf
unzip -q final-project-agent-code.zip -d clean/agent_code/
docker build -f Dockerfile.pinned -t bomberman_clean .   # context = clean/
```

`git archive` writes real files, so the earlier project failure — a symlinked
framework letting `sys.path[0]` resolve back to the project root and silently
loading the *project's* `agent_code/* — cannot occur. Verified:

```
$ find clean -type l          # (no output: zero symlinks)
$ diff <(git show d7eed90:settings.py) clean/settings.py && echo identical
identical
```

Verified *inside* the image rather than on the host:

```
$ docker run --rm bomberman_clean bash -c "ls tools checkpoints scratchpad results"
ls: cannot access 'tools': No such file or directory
ls: cannot access 'checkpoints': No such file or directory
ls: cannot access 'scratchpad': No such file or directory
ls: cannot access 'results': No such file or directory
$ docker run --rm bomberman_clean md5sum agent_code/benedict_task4/q_table.npy
54d63bc79179fdf80d3b9bfb80f21461  agent_code/benedict_task4/q_table.npy
```

The md5 therefore survives host → zip → unzip → `docker build` → image.
`agent_code/` in the isolated image holds only the seven provided agents plus
`benedict_task4`; **no `ext_*`**.

### 3a · 3 × `rule_based_agent`, `classic`, 20 rounds — PASS

| agent | score | coins | crates | kills | suicides | invalid |
|---|---|---|---|---|---|---|
| **benedict_task4** | **91** | 51 | 692 | 8 | 9 | 89 |
| rule_based_agent_0 | 70 | 45 | 631 | 5 | 11 | 133 |
| rule_based_agent_1 | 67 | 42 | 574 | 5 | 13 | 155 |
| rule_based_agent_2 | 52 | 42 | 577 | 2 | 8 | 156 |

Exit 0, no traceback. **And the table was demonstrably loaded** — not inferred
from the score, but read out of the agent's own log inside the container:

```
2026-08-21 18:29:39,923 [benedict_task4_code] INFO: Loading Q-table from disk.
```

`logs/` was absent from the zip and `agents.py:227` created it, as predicted.

### 3b · 3 × `random_agent` (the pre-run config), isolated — PASS

`benedict_task4` score **167**, 167 coins, **0 suicides**; all three
`random_agent`s score 0–0 and suicide in all 20 rounds.

### 3c · Missing table must fail loudly — PASS

The E18 guard, exercised by moving the table aside inside the container:

```
  File "/home/bomberman/agent_code/benedict_task4/callbacks.py", line 429, in setup
    raise FileNotFoundError(f"No Q-table at {MODEL_FILE} and not training.")
FileNotFoundError: No Q-table at /home/bomberman/agent_code/benedict_task4/q_table.npy
                   and not training.
```

It crashes rather than playing the all-zero random policy. Note the path is
`/home/bomberman/agent_code/benedict_task4/q_table.npy` — absolute at *runtime*
because it is derived from `__file__`, which is exactly right; nothing in the
source is an absolute path, so `agents.py:305`'s per-round `chdir` is harmless.

**Test 3: PASS — this is the test that mattered, and the zip stands alone.**

## 4 · Timing against the 0.5 s limit

### 4a · Does `README.md`'s claim survive? — YES, exactly

The README (which goes to the graders) says:

> Worst single step observed **54.4 ms** across **472 600** evaluated
> agent-rounds, against the 0.5 s per-step limit — 0 breaches, and 0 recorded
> `think_over_limit` in any evaluation.

Recomputed from scratch over **all 1366 CSVs** in `results/eval/**`, taking
every row whose `agent` is `benedict_task4`:

| quantity | recomputed | README |
|---|---|---|
| agent-rounds | **472 600** | 472 600 ✔ |
| global `max(think_max_ms)` | **54.4353 ms** | 54.4 ms ✔ |
| total `think_over_limit` | **0** | 0 ✔ |

The maximum is attained in
`results/eval/task4_tournament/benedict_q_e36_PLB_s103__ep20000__task4_rb_val550731.csv`,
round 772. **The claim is correct as written and is confirmed.**

Two clarifications worth having on record, neither of which contradicts it:

- That 54.4 ms row belongs to table `e36_PLB_s103`, **not** the shipped
  `e37_PLB2_s106`. The README's number is the worst step ever produced by *this
  agent code* over every table it evaluated, which is the conservative reading
  and the right one to publish — the cost is in the feature computation, which
  is identical across tables.
- Restricted to the **shipped** table (4 CSVs, 4 000 agent-rounds): max
  **21.25 ms**, mean **0.1029 ms**, `think_over_limit` 0.

The figure quoted in this session's brief — "mean 0.137 ms and worst 53.4 ms
across 180 000 rounds" — is the **superseded** number, and the README already
says so itself: *"(An earlier figure of 53.4 ms / 180 000 rounds was the E37
sweep alone; the global maximum is the number above.)"* No correction is needed.

### 4b · Measured inside the container

Harness: `scratchpad/benedict/docker/timing.py`, which patches `Agent.note_stat`
in memory exactly as `tools/evaluate.py` does, but imports nothing outside the
framework — so it runs in the **isolated** image where `tools/` does not exist.
The framework only sums think time per round; the per-step tail is what the
0.5 s limit acts on.

Relevant detail checked in the source rather than assumed:
`environment.py:112` instantiates `SequentialAgentBackend` (the process backend
on line 114 is commented out), so all four agents run **in one process,
sequentially**. `--cpus=1` is therefore a clean single-thread simulation with no
inter-agent contention inflating our numbers.

300 rounds, 3 × `rule_based_agent`, `classic`:

| | steps | mean | median | p99 | p99.9 | **max** | over 500 ms |
|---|---|---|---|---|---|---|---|
| container, `--cpus=1 --memory=8g` | 80 238 | 0.194 ms | 0.053 ms | 0.745 ms | 0.894 ms | **9.33 ms** | **0** |
| native host, same tree/harness | 78 831 | 0.119 ms | 0.037 ms | 0.454 ms | 0.518 ms | **2.81 ms** | **0** |

Containerised single-CPU costs ~**1.6×** on the mean over native.

### 4c · Does it fit on an AMD Ryzen 5 2600?

The binding number is not the 9.33 ms measured here but the **54.4 ms**
historical maximum, which comes from ~470 k agent-rounds — a far longer tail
than 300 rounds can sample. Against the 500 ms limit that is **9.2× headroom**.

So the falsifiable statement is: *the tournament machine would have to be more
than 9× slower per step than this development machine for a single breach.* A
2018 Zen+ desktop core is roughly 3–4× slower than an M5 core on this kind of
scalar/NumPy work, which leaves ~2–3× margin even at the extreme tail. I have
no Ryzen to measure on, so I am flagging that factor as an estimate rather than
a measurement — but the gap is large enough that the conclusion does not depend
on its precision.

Two structural reasons the tail is safe, both verifiable in the code: the cost
is two BFS traversals over a 17×17 board (bounded work, no data-dependent
blowup), and `act()` is a table lookup after the features — there is no search
whose depth could spike.

**Test 4: PASS.** `README.md`'s timing claim is accurate and needs no edit.

## 5 · Training inside the container (`--train 1`, no env vars)

### 5a · With the checkpoint parent available — PASS

Run in the **isolated** image (so `tools/` genuinely does not exist), with only
`checkpoints/benedict_task4/` mounted in, and **no environment variables**:

```
docker run --rm -v "$PWD/ckpt_mount:/home/bomberman/checkpoints" bomberman_clean \
  python main.py play --agents benedict_task4 rule_based_agent rule_based_agent rule_based_agent \
    --scenario classic --train 1 --n-rounds 5 --no-gui --seed 810731
```

Exit **0**, no traceback.

| check | result |
|---|---|
| `tools/` present? | no — `TrainLogger` import fell through to `None` as designed |
| `q_table.npy` md5 before | `54d63bc79179fdf80d3b9bfb80f21461` |
| `q_table.npy` md5 **after** | `54d63bc79179fdf80d3b9bfb80f21461` — **untouched** |
| written | `q_table_trained.npy` **+** `q_table_trained.npy.layout.json`, beside the agent |
| checkpoints dir | only the two parent files; no per-round checkpoint at 5 rounds |

So `--train` with no env vars works, and the guard against overwriting the
tournament's own input holds.

Side note worth having on record: the run created `__pycache__/` and `logs/` in
the agent directory. That is why the zip must be built from a **clean copy of
five named files** (§7) rather than by zipping the live directory.

### 5b · Inside the isolated zip, no `checkpoints/` — fails loudly, as intended

```
  File "/home/bomberman/agent_code/benedict_task4/train.py", line 432, in warm_start
    coarse = np.load(coarse_file)
FileNotFoundError: [Errno 2] No such file or directory:
  '/home/bomberman/agent_code/benedict_task4/../../checkpoints/benedict_task4/q_table_parent.npy'
```

And critically:

```
--- did it silently write a table anyway? ---
no q_table_trained.npy written -- good
--- q_table.npy md5 ---
54d63bc79179fdf80d3b9bfb80f21461
```

**This is expected and correct.** Training is not part of the submission; the
warm-start parent lives outside the zip by design. It fails loudly instead of
silently training from zero, and it does not touch the shipped table. The
tournament never passes `--train`, so this path is never taken there.

**Test 5: PASS.**

## 6 · Zip file list

```
$ unzip -l final-project-agent-code.zip
  Length      Date    Time    Name
---------  ---------- -----   ----
        0  08-21-2026 11:25   benedict_task4/
  3072128  08-21-2026 11:25   benedict_task4/q_table.npy
     4402  08-21-2026 11:25   benedict_task4/README.md
    20046  08-21-2026 11:25   benedict_task4/callbacks.py
    23902  08-21-2026 11:25   benedict_task4/train.py
       43  08-21-2026 11:25   benedict_task4/q_table.npy.layout.json
---------                     -------
  3120521                     6 files
```

146 KB compressed. Exactly the five required files under a single top-level
directory (which is what the PDF's *"first directory that contains a
`callbacks.py`"* rule needs).

| check | result |
|---|---|
| stray entries beyond the five | **none** (asserted by an explicit regex over `unzip -Z1`) |
| `__pycache__`, `logs/`, `q_table_trained.npy`, checkpoints | **absent** |
| `agent_code/ext_*` | **absent** — 0 matches |
| `.DS_Store` / `__MACOSX` | **absent** (`zip -X`, and the stage dir holds only the five copied files) |
| md5 of `q_table.npy` inside the zip | `54d63bc79179fdf80d3b9bfb80f21461` ✔ |

### Integrity, re-verified after all five tests

```
$ md5 -q agent_code/benedict_task4/q_table.npy
54d63bc79179fdf80d3b9bfb80f21461
$ md5 -q checkpoints/benedict_task4/q_table_e37_PLB2_s106__ep20000.npy
54d63bc79179fdf80d3b9bfb80f21461
$ git status --porcelain -- agent_code/benedict_task4/
(empty)
$ git status --porcelain
?? scratchpad/benedict/docker/
```

The frozen agent was not modified. Nothing outside `scratchpad/benedict/docker/`
changed.

## 7 · GO / NO-GO

# **GO.**

All five tests pass. The decisive one is test 3: the zip, unpacked into a
pristine upstream checkout with no `tools/`, no `checkpoints/`, no `scratchpad/`
and no `results/`, plays a full 20-round `classic` game in the container, beats
three `rule_based_agent`s 91–70–67–52, and its own log confirms it loaded the
trained table rather than falling back to anything. No blocker.

Summary:

| # | test | verdict |
|---|---|---|
| 0 | `q_table.npy` md5, before and after | **PASS** — `54d63bc7…` unchanged |
| 1 | build from the provided `Dockerfile` | **FAIL — course's file, not ours** (see below) |
| 1c | build with the base image pinned | **PASS** — 151 s, 2.92 GB |
| 2 | agent runs in the container | **PASS** — score 84 vs 44/53/59 |
| 2b | PDF pre-run config (3 × `random_agent`) | **PASS** — score 154, 0 suicides |
| 3 | **the zip in isolation** | **PASS** — score 91, table load confirmed in-log |
| 3c | missing table fails loudly | **PASS** |
| 4 | timing vs the 0.5 s limit | **PASS** — max 9.33 ms in-container; 9.2× headroom on the historical worst |
| 5 | `--train 1`, no env vars | **PASS** — writes `q_table_trained.npy`, leaves `q_table.npy` alone |
| 5b | training without the parent | **PASS** — fails loudly, writes nothing |
| 6 | zip contents | **PASS** — exactly 5 files, no `ext_*`, no `__pycache__` |

### The one thing that is not green, and why it does not block

`docker build .` on the provided `Dockerfile` **fails today**, at the TensorFlow
line, because `FROM continuumio/miniconda3` is untagged and now resolves to
Python 3.14, for which no TensorFlow wheel exists. This is the course's file and
the course's dependency. Our agent imports `os`, `numpy` and `settings` — all
present two `conda` lines earlier — so nothing about our submission depends on
the failing line.

It is still worth raising, because if the graders' image is built fresh from this
file it will fail for *everyone*. The PDF itself says *"please git pull to get
the newest version of the Dockerfile"*, so it is theirs to fix. If you want to
mention a fix, the minimal one is a base pin:

```diff
-FROM continuumio/miniconda3
+FROM continuumio/miniconda3:25.1.1-2
```

(newest tag still on Python 3.12; also dodges the *"updates discontinued after
26.7.x, move to `anaconda/miniconda`"* deprecation the base image now prints.)
**Not applied** — the repo `Dockerfile` is untouched; the pin lives only in
`scratchpad/benedict/docker/Dockerfile.pinned`.

### Two decisions for you

**1. `requirements.txt` in the zip.** The PDF's pre-run does *"unzip → install
libraries specified in `requirements.txt`"*. Ours sits at the repo root, not in
the zip, and the zip spec is exactly five files. **My recommendation: leave it
out.** The agent needs only `numpy`, which the base image already provides, so
an absent `requirements.txt` installs nothing that was needed — verified by test
3 running in an image built with no `pip install` of ours at all. Adding it
would be harmless but would break the "exactly five files" rule for no gain.

**2. `scratchpad/` is not gitignored, and this session wrote 331 MB into it.**
`ctx_repo/` (178 MB) and `clean/` (146 MB) are Docker build contexts. Do **not**
`git add scratchpad/` without excluding them. `rebuild.sh` regenerates every one
of them from scratch, so they are safe to delete:

```
rm -rf scratchpad/benedict/docker/{ctx_repo,clean,zipstage,ckpt_mount}
```

That leaves 3 MB: `REPORT.md`, `rebuild.sh`, `timing.py`, `Dockerfile.pinned`,
the build logs and the per-test logs (`t2a`, `t2b`, `t3a`, `t3b`, `t3c`,
`t4_cpus1`, `t4_native`, `t5a`, `t5b`).

Proposed `.gitignore` addition — **proposed, not applied**:

```
scratchpad/benedict/docker/ctx_repo/
scratchpad/benedict/docker/clean/
scratchpad/benedict/docker/zipstage/
scratchpad/benedict/docker/ckpt_mount/
scratchpad/benedict/docker/final-project-agent-code.zip
```

## 8 · Commands for Benedict

### Build the final submission zip

Run from the repo root. It copies the five named files into a fresh staging
directory rather than zipping the live folder — the live folder accumulates
`__pycache__/` and `logs/` the moment anything runs (observed in test 5a).

```bash
cd /Users/benedictvonschubert/Projects/bomberman_RL

rm -rf /tmp/bm_zip final-project-agent-code.zip
mkdir -p /tmp/bm_zip/benedict_task4
for f in callbacks.py train.py q_table.npy q_table.npy.layout.json README.md; do
  cp "agent_code/benedict_task4/$f" "/tmp/bm_zip/benedict_task4/$f"
done
( cd /tmp/bm_zip && zip -r -X -q final-project-agent-code.zip benedict_task4 )
mv /tmp/bm_zip/final-project-agent-code.zip .

# verify -- must print exactly 5 files and the expected md5
unzip -l final-project-agent-code.zip
unzip -p final-project-agent-code.zip benedict_task4/q_table.npy | md5 -q
#   expected: 54d63bc79179fdf80d3b9bfb80f21461
unzip -Z1 final-project-agent-code.zip | grep -Ev \
  '^benedict_task4/(callbacks\.py|train\.py|q_table\.npy|q_table\.npy\.layout\.json|README\.md)$|^benedict_task4/$' \
  && echo 'STRAY FILE -- do not submit' || echo 'clean'
```

`final-project-agent-code.zip` lands at the repo root — **do not commit it**;
add it to `.gitignore` or build it outside the tree.

### Git

I ran no git commands. Everything new is under `scratchpad/benedict/docker/`.

```bash
cd /Users/benedictvonschubert/Projects/bomberman_RL

# drop the 325 MB of regenerable Docker build contexts first
rm -rf scratchpad/benedict/docker/{ctx_repo,clean,zipstage,ckpt_mount}

git status --porcelain
git add scratchpad/benedict/docker/
git commit -m "Docker submission gate for benedict_task4: GO.

Five tests in the provided image, none previously run. The decisive one is the
zip in isolation - unpacked into a pristine upstream checkout (git archive of
d7eed90, copied never symlinked, no tools/ checkpoints/ scratchpad/ results/)
it plays 20 rounds of classic and beats three rule_based_agents 91-70-67-52,
with 'Loading Q-table from disk.' read out of the container's own agent log
rather than inferred from the score. q_table.npy md5 54d63bc7 is unchanged and
survives host -> zip -> unzip -> docker build -> image. The zip is exactly the
five required files, no ext_* and no __pycache__.

The provided Dockerfile does not build: FROM continuumio/miniconda3 is untagged
and now resolves to Python 3.14, which has no TensorFlow wheel. That is the
course's file and the course's dependency - our agent imports os, numpy and
settings only - so all tests ran on a base pinned to 25.1.1-2 (Python 3.12).
The pin lives in scratchpad only; the repo Dockerfile is untouched.

README.md's timing claim audited and confirmed rather than assumed: recomputed
over all 1366 eval CSVs, the max is 54.4353 ms across exactly 472600
benedict_task4 agent-rounds with 0 think_over_limit, so 54.4 ms / 472600 is
right as written. Restricted to the shipped table it is 21.25 ms over 4000
rounds. In-container at --cpus=1, 300 rounds / 80238 steps give mean 0.194 ms
and max 9.33 ms, 1.6x the native mean. Against 500 ms the historical worst
leaves 9.2x headroom.

Training with no environment variables works in the isolated image without
tools/: it writes q_table_trained.npy plus its layout sidecar and leaves
q_table.npy byte-identical. Without checkpoints/ it raises FileNotFoundError on
the warm-start parent and writes nothing, which is correct - training is not
part of the submission.

Two findings that are the framework's, not ours: environment.py:61 opens
logs/game.log without creating logs/, so a checkout missing logs/.gitkeep cannot
start any game; and the repo root is 19 GB with no .dockerignore, so a literal
docker build . there ships all of it into the image.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01BBjh4A5KXBT1VHEZ1G1EhU"
```

Two `.gitignore` edits are **proposed, not applied** — the Docker context
exclusions above, and `final-project-agent-code.zip`.

## 9 · Reproducing this

```bash
./scratchpad/benedict/docker/rebuild.sh     # rebuilds both images + the zip + the clean tree
docker run --rm bomberman_clean python main.py play \
  --agents benedict_task4 rule_based_agent rule_based_agent rule_based_agent \
  --scenario classic --n-rounds 20 --no-gui
docker run --rm --cpus=1 --memory=8g \
  -v "$PWD/scratchpad/benedict/docker/timing.py:/tmp/timing.py:ro" \
  -w /home/bomberman bomberman_clean python /tmp/timing.py 300
```

### Caveats on what was and was not tested

- The image is **`linux/arm64`** (Apple silicon); the tournament reference is
  `linux/amd64`. Irrelevant to correctness — the agent is pure NumPy + stdlib —
  and handled explicitly for timing in §4c. An emulated amd64 build was not run:
  under QEMU the timing numbers would be meaningless, and the correctness result
  would not change.
- Test 1 was not run as a literal `docker build .` at the repo root; §1a says
  why (19 GB context) and what was built instead.
- Round counts here are 20–300, chosen to exercise the code paths in the
  container, not to re-measure the agent's strength. The strength numbers stand
  on the 1000-round held-out evaluation already in `README.md`.
