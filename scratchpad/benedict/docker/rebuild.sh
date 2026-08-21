#!/usr/bin/env bash
# Rebuilds everything this session's Docker gate used, from scratch.
# Run from the repo root. Nothing here touches the repo working tree.
set -euo pipefail
R="$(cd "$(dirname "$0")/../../.." && pwd)"
D="$R/scratchpad/benedict/docker"
UPSTREAM=d7eed90          # the commit that imported the framework

# ---- 1. dev-repo build context (framework + our agents, minus the 19 GB of data)
rm -rf "$D/ctx_repo"; mkdir -p "$D/ctx_repo"
rsync -a --exclude .git --exclude .venv --exclude logs --exclude checkpoints \
  --exclude scratchpad --exclude results --exclude replays --exclude screenshots \
  --exclude __pycache__ --exclude '*.pyc' --exclude 'agent_code/ext_*' \
  --exclude 'agent_code/*/logs' --exclude final_project.pdf \
  "$R"/ "$D/ctx_repo"/
# environment.py:61 opens logs/game.log without creating logs/ -- upstream ships logs/.gitkeep
mkdir -p "$D/ctx_repo/logs" && touch "$D/ctx_repo/logs/.gitkeep"
cp "$D/Dockerfile.pinned" "$D/ctx_repo/"

# ---- 2. the zip, built from a clean copy of exactly five files
rm -rf "$D/zipstage" "$D/final-project-agent-code.zip"
mkdir -p "$D/zipstage/benedict_task4"
for f in callbacks.py train.py q_table.npy q_table.npy.layout.json README.md; do
  cp "$R/agent_code/benedict_task4/$f" "$D/zipstage/benedict_task4/$f"
done
( cd "$D/zipstage" && zip -r -X -q ../final-project-agent-code.zip benedict_task4 )

# ---- 3. pristine upstream checkout + the unzipped submission (COPIED, never symlinked)
rm -rf "$D/clean"; mkdir -p "$D/clean"
git -C "$R" archive "$UPSTREAM" | tar -x -C "$D/clean"
rm -f "$D/clean/final_project.pdf"
unzip -q "$D/final-project-agent-code.zip" -d "$D/clean/agent_code/"
cp "$D/Dockerfile.pinned" "$D/clean/"

# ---- 4. images
( cd "$D/ctx_repo" && docker build -q -f Dockerfile.pinned -t bomberman . )
( cd "$D/clean"    && docker build -q -f Dockerfile.pinned -t bomberman_clean . )

# ---- 5. checkpoint parent for the training test (outside the zip by design)
rm -rf "$D/ckpt_mount"; mkdir -p "$D/ckpt_mount/benedict_task4"
cp "$R/checkpoints/benedict_task4/q_table_parent.npy"{,.layout.json} \
   "$D/ckpt_mount/benedict_task4/"
echo "OK -- images 'bomberman' and 'bomberman_clean' rebuilt."
