#!/usr/bin/env bash
# Install the four third-party evaluation opponents into agent_code/ under the names this
# project measures them by. Self-contained: clone, patch and verify, from this file alone.
#
#     bash tools/install_ext_agents.sh            # install what is missing
#     bash tools/install_ext_agents.sh --force    # reinstall from scratch
#
# The agents are NOT in this repo -- `.gitignore` excludes `agent_code/ext_*/`, because two of
# the three source repos ship no licence and the submission zip is built out of agent_code/.
# None of the four loads in our harness as shipped, so this script also applies the source fixes
# recorded in scratchpad/external/install/INSTALL.md (relative imports, a pickle module alias,
# the train.py rename, and the removal of the epsilon / self-play branches so a frozen opponent
# cannot act randomly). Everything is pinned by commit SHA and checked against embedded SHA-256
# hashes, so every team member ends up with byte-identical opponents.
#
# Touches no git state in this repo. Needs: git, patch, shasum (or sha256sum), ~17 MB.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
DEST="$ROOT/agent_code"
FORCE="${1:-}"

[[ -f "$ROOT/main.py" && -d "$DEST" ]] || { echo "not a bomberman_RL checkout: $ROOT" >&2; exit 1; }
command -v git   >/dev/null || { echo "git not found" >&2; exit 1; }
command -v patch >/dev/null || { echo "patch not found" >&2; exit 1; }
if command -v shasum >/dev/null; then SHA="shasum -a 256"; else SHA="sha256sum"; fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# ---------------------------------------------------------------- embedded payloads
# The patches are diffs of the upstream files against the copies these evaluations were run
# against; regenerate them with `diff -u` if a source repo is ever re-pinned.
cat > "$TMP/xiaoxiae.patch" <<'PAYLOAD_xiaoxiae_patch'
--- a/callbacks.py
+++ b/callbacks.py
@@ -1,13 +1,15 @@
 import math
 from random import choice
 
-from .train import *
+# Renamed from `train.py` so the framework cannot load training callbacks;
+# callbacks.py needs the module-level definitions (DQN, state_to_features, paths).
+from ._agent_defs import *
 
 cwd = os.path.abspath(os.path.dirname(__file__))
 
 
 def setup(self):
-    if not self.train and not MANUAL:
+    if not MANUAL:
         self.model = DQN(FEATURE_VECTOR_SIZE, len(ACTIONS), LAYER_SIZES).to(device)
         self.model.load_state_dict(torch.load(TARGET_MODEL_PATH, map_location=device))
         self.model.eval()
@@ -22,14 +24,6 @@
         print(state.tolist()[0])
         return game_state['user_input']
 
-    if self.train:
-        threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * game_state["step"] / EPS_DECAY)
-
-        if random.random() <= threshold:
-            action = choice(ACTIONS)
-            self.logger.info(f"Picking random action of {action}.")
-            return action
-
     with torch.no_grad():
         state = state_to_features(game_state)
         model_result = self.model(state)
PAYLOAD_xiaoxiae_patch

cat > "$TMP/aielka_ql.patch" <<'PAYLOAD_aielka_ql_patch'
--- a/callbacks.py
+++ b/callbacks.py
@@ -8,8 +8,8 @@
 """
 from collections import deque
 
-from agent_code.ql.feature_extraction import state_to_small_features
-from agent_code.ql.q_learning import QLearningAgent
+from .feature_extraction import state_to_small_features
+from .q_learning import QLearningAgent
 
 
 
@@ -74,8 +74,9 @@
     feature_vector = state_to_small_features(
         game_state, num_coins_already_discovered)
     
-    return self.agent.act(feature_vector, 
-                        n_round = game_state["round"], 
-                        train = self.train)
+    # Frozen opponent: always greedy, never the training epsilon/systematic-exploration path.
+    return self.agent.act(feature_vector,
+                        n_round = game_state["round"],
+                        train = False)
     
     
--- a/feature_extraction.py
+++ b/feature_extraction.py
@@ -1,4 +1,4 @@
-from agent_code.ql.utils import *
+from .utils import *
 
 FEATURE_VECTOR_SIZE=20
 
--- a/q_learning.py
+++ b/q_learning.py
@@ -3,8 +3,8 @@
 import pickle
 
 import events as e
-import agent_code.ql.own_events as own_e
-from agent_code.ql.utils import *
+from . import own_events as own_e
+from .utils import *
 
 
 class QLearningAgent:
PAYLOAD_aielka_ql_patch

cat > "$TMP/lijesse.patch" <<'PAYLOAD_lijesse_patch'
--- a/callbacks.py
+++ b/callbacks.py
@@ -1,9 +1,17 @@
 import pickle
 import random
+import sys
 import torch
 
+from . import model as _model
 from .features import Feature
 
+# `my-saved-model.pt` is a pickled nn.Module whose class is recorded under the module path of
+# the ORIGINAL folder name (`agent_code.feature_is_everything.model`). Renaming the folder makes
+# that path unimportable, so alias it to this copy's `model` module before unpickling.
+sys.modules.setdefault("agent_code.feature_is_everything", sys.modules[__package__])
+sys.modules.setdefault("agent_code.feature_is_everything.model", _model)
+
 ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
 
 def encode_feature(features):
@@ -48,10 +56,11 @@
     """
     self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     self.feature = Feature()
-    if not self.train:
-        with open("my-saved-model.pt", "rb") as file:
-            self.policy_net = pickle.load(file).to(self.device)
-        self.logger.info("Loading model from saved state.")
+    # Frozen opponent: always load the trained net (originally guarded on `not self.train`).
+    with open("my-saved-model.pt", "rb") as file:
+        self.policy_net = pickle.load(file).to(self.device)
+    self.policy_net.eval()
+    self.logger.info("Loading model from saved state.")
 
 
 def act(self, game_state: dict) -> str:
@@ -66,23 +75,9 @@
     features = self.feature(game_state)
 
     features_tensor = torch.tensor(encode_feature(features), dtype=torch.float32).unsqueeze(0).to(self.device)
-    if self.train:
-        if random.random() > self.epsilon:
-            with torch.no_grad():
-                q_values = self.policy_net(features_tensor)
-                action_index = q_values.max(1)[1].item()
-        else:
-            action_index = random.randrange(len(ACTIONS))
-    else:
-        with torch.no_grad():
-            q_values = self.policy_net(features_tensor)
-            self.logger.debug(f"q_values: {q_values}")
-            action_index = q_values.max(1)[1].item()
+    with torch.no_grad():
+        q_values = self.policy_net(features_tensor)
+        self.logger.debug(f"q_values: {q_values}")
+        action_index = q_values.max(1)[1].item()
 
-    # If train with itself
-    if game_state["round"] % 500 == 0 and game_state["step"] == 1:
-        setup(self)
-    if self.train:
-        self.steps_done += 1
-
     return ACTIONS[action_index]
PAYLOAD_lijesse_patch

cat > "$TMP/MANIFEST.sha256" <<'PAYLOAD_MANIFEST_sha256'
7d0d7f429adae1c64eb56a64f38f2401287ce48e0a90200c97bc69aa03cf1288  ext_aielka_ql_atom/avatar.png
20d2d2667bcdcd6c9186991a6fee9946b49f6dc3fd6a413e181467b9d8e974e1  ext_aielka_ql_atom/bomb.png
a4e83d4ca4504ad6dc5c87d093e58ae0de33546001dd0350cb489bb15ac79697  ext_aielka_ql_atom/callbacks.py
19bc312c5a57ba99cef47632c2510765fd1f41a4c6d6be4243d6129c46958374  ext_aielka_ql_atom/feature_extraction.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  ext_aielka_ql_atom/models/.gitkeep
917724e510943658d68ce2e3fb7044c58ebf02dec8f42b4a7f37206050dc144f  ext_aielka_ql_atom/models/q_table.pkl
d79ae7d2672783c22bf25818f1777b7d00f11e5648e0a08d41174f5e483e96e1  ext_aielka_ql_atom/own_events.py
258f656f2a6fadb285763e944e933bd2247043d3dbe5388c62ea960a2a503512  ext_aielka_ql_atom/q_learning.py
b61e8fdc7433bcd242ead6ad153e793c3bab9bf229deb69ad4436898cc99a142  ext_aielka_ql_atom/utils.py
37ce872a8a2751eabb906164acfaa32f7f0f41fd9938dc2c36c35e8727bac76c  ext_lijesse_featureeverything/README.md
f2f4ca7f23e892d7236abd5919168a8e2669e4b9d0af8adab710b06eec8e37b2  ext_lijesse_featureeverything/avatar.png
2d258066ea0ca1d8991f09e001d2e15598a316f74b0863715efc65c48aad8232  ext_lijesse_featureeverything/bomb.png
08325acea9944ecf76c38005de359fcc1cfa4d75ec8560f36e7c20d8b7bdea88  ext_lijesse_featureeverything/callbacks.py
8baf96ae76f685a5dbfb74bae644e2cbb816952103859081801888ebe6b9f620  ext_lijesse_featureeverything/features.py
27736c5afa4ccf50582141b6aea57740699365e35879ad6d5cb4842b3fc7829c  ext_lijesse_featureeverything/model.py
0fb1df0b3c3f255db121e8ba7c6061d1293f55d3e7ae12772259643ee2060577  ext_lijesse_featureeverything/my-saved-model.pt
cac8ad3d8b197d1cca6fa67ca35d3db00d45c4c81afa549d797bb8b45657f82e  ext_lijesse_featureeverything/symmetry.py
4907a9151f4aca00012341208cf4d9434b18d79febb3d7573499e39157e45019  ext_xiaoxiae_binary_v6/_agent_defs.py
f8c2be2ec9e85392611e5f81e448da8907013ba1574b6ed1f0a19fb00e73f3f4  ext_xiaoxiae_binary_v6/avatar.png
1f3fa1afdd2ed845a4e20d9a02dc3597fd43144e62b62961321df868e291771d  ext_xiaoxiae_binary_v6/avatar.xcf
e3be6922ba219c4f67886deda1a5a95a90955642a6d560b9e01bfe92b463741f  ext_xiaoxiae_binary_v6/bomb.png
1733b334a6e02ef63d2803130698f4583e48da42df80fc955cd46728a15532f8  ext_xiaoxiae_binary_v6/bomb.xcf
fbce85557b7905eb78869eb673fcd7c9918fe26c01055c52c67dd5c48f823915  ext_xiaoxiae_binary_v6/callbacks.py
065212d88c4f8a2c21b979c5a09aa6a86ddc12bfcb6ed061fe7272430892b1d5  ext_xiaoxiae_binary_v6/policy-model.pt
adabec551f6f429eed479e363febf0b6929e36df8ec305a9fd3de1036a7a0bb6  ext_xiaoxiae_binary_v6/target-model.pt
cb9491f33e3fb896a50b352db1f4de91c70be6649492628c552edf2a91abbe44  ext_xiaoxiae_bindist_v2/_agent_defs.py
f8c2be2ec9e85392611e5f81e448da8907013ba1574b6ed1f0a19fb00e73f3f4  ext_xiaoxiae_bindist_v2/avatar.png
1f3fa1afdd2ed845a4e20d9a02dc3597fd43144e62b62961321df868e291771d  ext_xiaoxiae_bindist_v2/avatar.xcf
e3be6922ba219c4f67886deda1a5a95a90955642a6d560b9e01bfe92b463741f  ext_xiaoxiae_bindist_v2/bomb.png
1733b334a6e02ef63d2803130698f4583e48da42df80fc955cd46728a15532f8  ext_xiaoxiae_bindist_v2/bomb.xcf
fbce85557b7905eb78869eb673fcd7c9918fe26c01055c52c67dd5c48f823915  ext_xiaoxiae_bindist_v2/callbacks.py
8c6a937b682f001026c1f1ebde0d7e14a830783c36f8630f08d1a0dfea905cbe  ext_xiaoxiae_bindist_v2/policy-model.pt
4fcccb5807753e54369737ac939497503c5f71728eb3bdccf2603e5028675221  ext_xiaoxiae_bindist_v2/target-model.pt
PAYLOAD_MANIFEST_sha256

# ---------------------------------------------------------------- install
# name | repo url | commit | path within the repo
AGENTS=(
  "ext_xiaoxiae_bindist_v2|https://github.com/xiaoxiae/BombermanML.git|50b682f50c9bcffb3aa2b0a1f2f96d43121b8b09|agent_code/binary_distance_agent_v2"
  "ext_xiaoxiae_binary_v6|https://github.com/xiaoxiae/BombermanML.git|50b682f50c9bcffb3aa2b0a1f2f96d43121b8b09|agent_code/binary_agent_v6"
  "ext_aielka_ql_atom|https://github.com/AI-ELka/BombermanRLAgents.git|8d857315ecdff93abfa5c052f5e48f57e64e10c5|agent_code/ql"
  "ext_lijesse_featureeverything|https://github.com/Li-Jesse-Jiaze/MLE_project_bomberman.git|a7fe5041b02548ce4438e502ca3adb11576bae75|agent_code/feature_is_everything"
)

clone_at() {  # $1 url, $2 sha -> echoes the worktree path; one clone per (url, sha)
  local url="$1" sha="$2" dir="$TMP/repo_$(printf '%s' "$url$sha" | $SHA | cut -c1-12)"
  if [[ ! -d "$dir" ]]; then
    git init -q "$dir"
    git -C "$dir" remote add origin "$url"
    # Fetching a bare SHA keeps this to one commit; GitHub allows it. Fall back for hosts that don't.
    if ! git -C "$dir" fetch -q --depth 1 origin "$sha" 2>/dev/null; then
      git -C "$dir" fetch -q origin
    fi
    git -C "$dir" checkout -q "$sha" 2>/dev/null || git -C "$dir" checkout -q FETCH_HEAD
  fi
  printf '%s' "$dir"
}

for spec in "${AGENTS[@]}"; do
  IFS='|' read -r name url sha sub <<< "$spec"
  target="$DEST/$name"
  if [[ -d "$target" ]]; then
    if [[ "$FORCE" == "--force" ]]; then
      rm -rf "$target"
    else
      echo "skip  $name (exists; re-run with --force to reinstall)"
      continue
    fi
  fi

  echo "fetch $name  <-  ${url##*/} @ ${sha:0:7}  $sub"
  src="$(clone_at "$url" "$sha")/$sub"
  [[ -d "$src" ]] || { echo "  missing $sub in that commit" >&2; exit 1; }

  # -L dereferences the xiaoxiae avatar/bomb symlinks, which point at a sibling agent folder
  # that we do not copy; a plain -R would leave them dangling.
  cp -RL "$src" "$target"
  rm -rf "$target/__pycache__" "$target/logs" "$target/plot.txt"

  case "$name" in
    ext_xiaoxiae_*)
      # callbacks.py star-imports its definitions (DQN, state_to_features, paths) from train.py,
      # so the file has to stay -- renamed, so agents.py can no longer load it as training code.
      mv "$target/train.py" "$target/_agent_defs.py"
      patch -s -p1 -d "$target" < "$TMP/xiaoxiae.patch"
      ;;
    ext_aielka_ql_atom)
      rm -f "$target/train.py" "$target/add_own_events.py"   # add_own_events is imported only by train
      patch -s -p1 -d "$target" < "$TMP/aielka_ql.patch"
      ;;
    ext_lijesse_featureeverything)
      rm -f "$target/train.py"
      patch -s -p1 -d "$target" < "$TMP/lijesse.patch"
      ;;
  esac
  echo "  installed agent_code/$name"
done

# ---------------------------------------------------------------- verify
echo
echo "verifying 33 files against the embedded hashes"
( cd "$DEST" && $SHA -c "$TMP/MANIFEST.sha256" ) | grep -v ': OK$' || true
if ( cd "$DEST" && $SHA -c --status "$TMP/MANIFEST.sha256" ); then
  echo "all files match -- opponents are byte-identical to the ones our results were measured against."
else
  echo "MISMATCH: do not measure against these until it is explained." >&2
  exit 1
fi

cat <<'EOF'

Smoke test (30 rounds is a liveness check, not a result):
  BM_QUIET_LOGS=1 uv run python tools/evaluate.py --agents ext_xiaoxiae_binary_v6 \
    --opponents rule_based --n-rounds 30 --seed 990731 --out-dir results/eval/task4_tournament
EOF
