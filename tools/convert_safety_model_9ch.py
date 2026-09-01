#!/usr/bin/env python3
"""Expand the historical 8-channel safety model with one zero-weight channel."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent_code.ben_task2.model import CoinCollectorDQN


FIRST_CONVOLUTION_WEIGHT = "convolutional.0.weight"


def convert_state_dict(old_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a 9-channel state dict with an inert final input channel."""
    converted = CoinCollectorDQN().state_dict()

    for name, target in converted.items():
        source = old_state[name]
        if name == FIRST_CONVOLUTION_WEIGHT:
            if source.shape[1] != 8 or target.shape[1] != 9:
                raise ValueError(
                    f"Expected first convolution 8→9 channels, got "
                    f"{tuple(source.shape)}→{tuple(target.shape)}."
                )
            expanded = torch.zeros_like(target)
            expanded[:, :8] = source
            converted[name] = expanded
        else:
            if source.shape != target.shape:
                raise ValueError(
                    f"Unexpected shape for {name}: "
                    f"{tuple(source.shape)} != {tuple(target.shape)}."
                )
            converted[name] = source.clone()

    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    if not args.source.is_file():
        raise SystemExit(f"Source model does not exist: {args.source}")
    if args.destination.exists():
        raise SystemExit(f"Refusing to overwrite: {args.destination}")

    old_state = torch.load(args.source, map_location="cpu", weights_only=True)
    converted = convert_state_dict(old_state)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, args.destination)
    print(f"Wrote {args.destination}")


if __name__ == "__main__":
    main()
