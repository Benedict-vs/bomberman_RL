#!/usr/bin/env python3
"""Append one initially inert input channel to a saved Bomberman DQN."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


FIRST_CONVOLUTION_WEIGHT = "convolutional.0.weight"


def convert_state_dict(
    source: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Copy a state dict and append a zero-weight convolution channel."""
    converted = {name: tensor.clone() for name, tensor in source.items()}
    weight = source[FIRST_CONVOLUTION_WEIGHT]
    expanded = torch.zeros(
        (weight.shape[0], weight.shape[1] + 1, *weight.shape[2:]),
        dtype=weight.dtype,
        device=weight.device,
    )
    expanded[:, : weight.shape[1]] = weight
    converted[FIRST_CONVOLUTION_WEIGHT] = expanded
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

    source = torch.load(args.source, map_location="cpu", weights_only=True)
    converted = convert_state_dict(source)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, args.destination)
    print(f"Wrote {args.destination}")


if __name__ == "__main__":
    main()
