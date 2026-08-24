#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from sd15_residual_delta.config import load_config, save_resolved_config
from sd15_residual_delta.dataset import ResidualPairDataset, identity_collate
from sd15_residual_delta.factory import build_surrogate_bank
from sd15_residual_delta.train import fit, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(int(cfg["train"]["seed"]))
    output = Path(cfg["train"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, output)
    target_steps = (
        {int(step) for step in cfg["train"].get("target_steps", [])} or None
    )
    train_set = ResidualPairDataset(
        cfg["capture"]["output_dir"],
        "train",
        int(cfg["train"]["seed"]),
        target_steps,
    )
    val_set = ResidualPairDataset(
        cfg["capture"]["output_dir"],
        "val",
        int(cfg["train"]["seed"]),
        target_steps,
    )
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, collate_fn=identity_collate)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=identity_collate)
    model = build_surrogate_bank(cfg)
    if args.checkpoint:
        import torch

        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    fit(model, train_loader, val_loader, cfg["train"], output)


if __name__ == "__main__":
    main()
