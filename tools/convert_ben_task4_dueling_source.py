#!/usr/bin/env python3
"""Create an exactly Q-preserving dueling source from the frozen incumbent."""

from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_code.ben_task4.model import dueling_state_dict_from_classic


SOURCE = REPO_ROOT / "agent_code/ben_task4/ben_task4_mixed_kill_v1_2000ep_seed11.pt"
TARGET = REPO_ROOT / "agent_code/ben_task4/ben_task4_dueling_mixed_kill_source.pt"


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(f"Refusing to overwrite {TARGET}")
    state_dict = torch.load(SOURCE, map_location="cpu", weights_only=True)
    torch.save(dueling_state_dict_from_classic(state_dict), TARGET)
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
