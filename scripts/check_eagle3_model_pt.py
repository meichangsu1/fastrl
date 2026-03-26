#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _load_config(config_path: Path | None) -> dict:
    if config_path is None:
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _shape(x):
    if isinstance(x, torch.Tensor):
        return tuple(x.shape)
    return type(x).__name__


def main() -> int:
    parser = argparse.ArgumentParser(description="Check exported EAGLE3 model.pt structure.")
    parser.add_argument("model_pt", type=Path, help="Path to exported model.pt")
    parser.add_argument("--config", type=Path, default=None, help="Optional config.json for shape checks")
    args = parser.parse_args()

    state = torch.load(args.model_pt, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise SystemExit(f"Expected a flat state_dict dict, got {type(state).__name__}")

    config = _load_config(args.config)
    hidden_size = int(config["hidden_size"]) if "hidden_size" in config else None
    vocab_size = int(config["vocab_size"]) if "vocab_size" in config else None
    target_hidden_size = int(config.get("target_hidden_size", hidden_size or 0)) or None

    keys = sorted(state.keys())
    print(f"file: {args.model_pt}")
    print(f"num_keys: {len(keys)}")
    print("first_keys:")
    for key in keys[:20]:
        value = state[key]
        dtype = getattr(value, "dtype", type(value).__name__)
        print(f"  {key}: shape={_shape(value)} dtype={dtype}")

    errors: list[str] = []
    warnings: list[str] = []

    required_exact = ["d2t", "t2d", "fc.weight"]
    for key in required_exact:
        if key not in state:
            errors.append(f"missing required key: {key}")

    if not any(key.startswith("midlayer.") for key in keys):
        errors.append("missing any midlayer.* weights")

    frozen_keys = {"embed_tokens.weight", "lm_head.weight", "model.embed_tokens.weight", "model.lm_head.weight"}
    present_frozen = sorted(frozen_keys.intersection(keys))
    if present_frozen:
        warnings.append(f"found frozen/shared weights that are usually excluded: {present_frozen}")

    if any(key in state for key in ["model", "optimizer", "step"]):
        errors.append("looks like a training checkpoint wrapper, expected a flat model state_dict")

    fc_weight = state.get("fc.weight")
    if isinstance(fc_weight, torch.Tensor):
        if fc_weight.ndim != 2:
            errors.append(f"fc.weight should be 2D, got shape={tuple(fc_weight.shape)}")
        else:
            out_dim, in_dim = fc_weight.shape
            if hidden_size is not None and out_dim != hidden_size:
                errors.append(f"fc.weight out_dim mismatch: got {out_dim}, expected hidden_size={hidden_size}")
            if target_hidden_size is not None and in_dim != target_hidden_size * 3:
                warnings.append(
                    f"fc.weight in_dim={in_dim}, expected target_hidden_size*3={target_hidden_size * 3}"
                )

    d2t = state.get("d2t")
    t2d = state.get("t2d")
    if isinstance(d2t, torch.Tensor):
        if d2t.dtype not in (torch.int64, torch.int32, torch.long):
            warnings.append(f"d2t dtype is {d2t.dtype}, expected integer type")
    if isinstance(t2d, torch.Tensor):
        if t2d.dtype not in (torch.bool,):
            warnings.append(f"t2d dtype is {t2d.dtype}, expected bool")
        if vocab_size is not None and t2d.ndim == 1 and t2d.shape[0] != vocab_size:
            warnings.append(f"t2d length={t2d.shape[0]}, expected vocab_size={vocab_size}")

    print()
    if errors:
        print("errors:")
        for msg in errors:
            print(f"  - {msg}")
    if warnings:
        print("warnings:")
        for msg in warnings:
            print(f"  - {msg}")

    if not errors and not warnings:
        print("checkpoint structure looks consistent.")
    elif not errors:
        print("checkpoint structure is loadable, but review warnings above.")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
