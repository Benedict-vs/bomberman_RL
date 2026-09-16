#!/usr/bin/env python3
"""Create a Q-preserving auxiliary-opponent-prediction source model."""
from pathlib import Path
import sys
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from agent_code.ben_task4.model import opponent_prediction_state_dict_from_classic

SOURCE = REPO_ROOT / "agent_code/ben_task4/ben_task4_mixed_kill_v1_2000ep_seed11.pt"
TARGET = REPO_ROOT / "agent_code/ben_task4/ben_task4_opponent_prediction_mixed_kill_source.pt"

if __name__ == "__main__":
    if TARGET.exists():
        raise FileExistsError(f"Refusing to overwrite {TARGET}")
    state = torch.load(SOURCE, map_location="cpu", weights_only=True)
    torch.save(opponent_prediction_state_dict_from_classic(state), TARGET)
    print(f"Wrote {TARGET}")
