import subprocess, json, re, sys
from concurrent.futures import ThreadPoolExecutor
repos=[l.strip() for l in open("repos.txt") if l.strip()]
done=set()
try:
    for l in open("scan.tsv"):
        done.add(l.split("\t")[0])
except FileNotFoundError: pass
todo=[r for r in repos if r not in done]
PROV=re.compile(r'^agent_code/(rule_based|random|peaceful|coin_collector|tpl|user|fail)_agent/')
WEXT=re.compile(r'^agent_code/.*\.(npy|npz|pt|pth|pkl|pickle|h5|joblib|keras|onnx|sav)$', re.I)
def gh(path):
    p=subprocess.run(["gh","api",path],capture_output=True,text=True,stdin=subprocess.DEVNULL)
    if p.returncode: return None
    return json.loads(p.stdout)
def work(r):
    m=gh(f"repos/{r}")
    if not m: return f"{r}\tGONE\t\t0\t"
    t=gh(f"repos/{r}/git/trees/HEAD?recursive=1")
    paths=[x["path"] for x in (t or {}).get("tree",[])]
    w=[p for p in paths if WEXT.match(p) and not PROV.match(p)]
    lic=(m.get("license") or {}).get("spdx_id") or "none"
    return f"{r}\t{m['pushed_at']}\t{lic}\t{len(w)}\t{';'.join(w[:5])}"
with ThreadPoolExecutor(8) as ex, open("scan.tsv","a") as f:
    for i,res in enumerate(ex.map(work, todo)):
        f.write(res+"\n"); f.flush()
print("done", len(todo))
