"""Create the Q-preserving BombOutcomeDQN source from the incumbent."""
from pathlib import Path
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from agent_code.ben_task4.model import bomb_outcome_state_dict_from_classic

source = Path("agent_code/ben_task4/ben_task4_mixed_kill_v1_2000ep_seed11.pt")
destination = Path("agent_code/ben_task4/ben_task4_bomb_outcome_mixed_kill_source.pt")
if destination.exists(): raise SystemExit(f"Refusing to overwrite: {destination}")
torch.save(bomb_outcome_state_dict_from_classic(torch.load(source, map_location="cpu", weights_only=True)), destination)
print(f"Wrote {destination}")
