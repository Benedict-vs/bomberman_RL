#!/bin/zsh
while read r; do
  meta=$(gh api "repos/$r" --jq '[.pushed_at, (.license.spdx_id // "none")] | @tsv' </dev/null 2>/dev/null) || { print -r -- "$r\tGONE\t\t\t"; continue; }
  tree=$(gh api "repos/$r/git/trees/HEAD?recursive=1" --jq '.tree[].path' </dev/null 2>/dev/null)
  w=$(print -r -- "$tree" | grep -iE '^agent_code/.*\.(npy|npz|pt|pth|pkl|pickle|h5|joblib|keras|onnx|sav)$' | grep -viE '^agent_code/(rule_based|random|peaceful|coin_collector|tpl|user|fail)_agent/' | head -5 | tr '\n' ';')
  nw=$(print -r -- "$tree" | grep -icE '^agent_code/.*\.(npy|npz|pt|pth|pkl|pickle|h5|joblib|keras|onnx|sav)$')
  print -r -- "$r\t$meta\t$nw\t$w"
done < repos.txt
