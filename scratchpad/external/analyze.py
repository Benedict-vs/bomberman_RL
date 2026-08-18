import os, re, sys, subprocess, json
from pathlib import Path

ROOT = Path("/Users/benedictvonschubert/Projects/bomberman_RL")
EXT = ROOT / "scratchpad/external"
PROVIDED = {"rule_based_agent","random_agent","peaceful_agent","user_agent","tpl_agent",
            "coin_collector_agent","__pycache__"}
WEIGHT_EXT = {".npy",".pt",".pth",".pkl",".pickle",".h5",".joblib",".npz",".json",".model",".sav",".keras",".onnx",".txt",".csv"}
STRONG_W = {".npy",".pt",".pth",".pkl",".pickle",".h5",".joblib",".npz",".keras",".onnx"}

OURS = {}
for line in (ROOT/"settings.py").read_text().splitlines():
    m = re.match(r'^([A-Z_]+)\s*=\s*(.+?)(\s*#.*)?$', line)
    if m: OURS[m.group(1)] = m.group(2).strip()

def analyze(repo: Path):
    out = {"repo": repo.name}
    try:
        out["commit"] = subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
        out["date"] = subprocess.check_output(["git","-C",str(repo),"log","-1","--format=%ad","--date=short"],text=True).strip()
    except Exception as e: out["commit"]=f"ERR {e}"
    lic = [p.name for p in repo.iterdir() if p.name.lower().startswith(("license","licence","copying"))]
    out["license"] = lic or None
    ac = repo/"agent_code"
    out["agents"]=[]
    if not ac.is_dir():
        out["error"]="no agent_code/"; return out
    for d in sorted(ac.iterdir()):
        if not d.is_dir() or d.name in PROVIDED: continue
        info={"name":d.name}
        files=[p for p in d.rglob("*") if p.is_file() and "__pycache__" not in str(p)]
        info["has_callbacks"]=(d/"callbacks.py").exists()
        info["has_train"]=(d/"train.py").exists()
        w=[(str(p.relative_to(d)), p.stat().st_size) for p in files if p.suffix.lower() in STRONG_W]
        w.sort(key=lambda x:-x[1])
        info["weights"]=w[:8]
        info["n_weights"]=len(w)
        # imports and abs paths across py files
        imps=set(); abspaths=[]; mp=False; setconst=set()
        for p in files:
            if p.suffix!=".py": continue
            try: t=p.read_text(errors="ignore")
            except: continue
            for m in re.finditer(r'^\s*(?:from|import)\s+([A-Za-z_][\w.]*)', t, re.M):
                imps.add(m.group(1).split(".")[0])
            for m in re.finditer(r'''["'](/(?:Users|home|mnt|content|c:|C:)[^"'\n]*)["']''', t):
                abspaths.append((p.name, m.group(1)))
            for m in re.finditer(r'''["']([A-Za-z]:[\\/][^"'\n]*)["']''', t):
                abspaths.append((p.name, m.group(1)))
            if re.search(r'\bmultiprocessing\b|\btorch\.multiprocessing\b', t): mp=True
            for m in re.finditer(r'\b(BOMB_POWER|BOMB_TIMER|EXPLOSION_TIMER|COLS|ROWS|MAX_STEPS|MAX_AGENTS|CRATE_DENSITY|COIN_COUNT|REWARD_KILL|REWARD_COIN|TIMEOUT|SCENARIOS)\b', t):
                setconst.add(m.group(1))
        std={'os','sys','pickle','random','numpy','np','collections','typing','math','copy',
             'itertools','json','time','settings','events','callbacks','train','logging',
             'dataclasses','functools','pathlib','abc','heapq','enum','re','glob','warnings',
             'operator','__future__','string','datetime','argparse','csv','subprocess','shutil','queue','threading','traceback','contextlib','struct','hashlib','textwrap','statistics','bisect'}
        info["ext_imports"]=sorted(i for i in imps if i not in std and not (d/f"{i}.py").exists() and not (d/i).is_dir())
        info["abs_paths"]=abspaths[:6]
        info["multiprocessing"]=mp
        info["settings_consts"]=sorted(setconst)
        out["agents"].append(info)
    # settings.py diff
    diffs=[]
    sp=repo/"settings.py"
    if sp.exists():
        theirs={}
        for line in sp.read_text().splitlines():
            m=re.match(r'^([A-Z_]+)\s*=\s*(.+?)(\s*#.*)?$', line)
            if m: theirs[m.group(1)]=m.group(2).strip()
        for k in ["COLS","ROWS","MAX_AGENTS","MAX_STEPS","BOMB_POWER","BOMB_TIMER","EXPLOSION_TIMER","TIMEOUT","REWARD_KILL","REWARD_COIN"]:
            if k in OURS and theirs.get(k)!=OURS[k]:
                diffs.append(f"{k}: ours={OURS[k]} theirs={theirs.get(k)}")
    out["settings_diff"]=diffs
    return out

if __name__=="__main__":
    targets=sys.argv[1:] or [d.name for d in sorted(EXT.iterdir()) if (d/".git").exists()]
    for t in targets:
        print(json.dumps(analyze(EXT/t), indent=1))
        print("="*70)
