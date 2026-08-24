#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from flux_residual_delta.config import load_config
from flux_residual_delta.dataset import ResidualPairDataset
from flux_residual_delta.metrics import cosine_similarity, relative_mse
from train_surrogate import make_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    dataset = ResidualPairDataset(cfg["capture"]["output_dir"], "test", cfg["train"]["seed"])
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda")
    model = make_model(cfg).to(device=device, dtype=torch.bfloat16)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    model_mse, zero_mse, cosine = [], [], []
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            pred = model(
                batch["delta_h"], batch["anchor_residual"], batch["timestep"], batch["horizon"]
            )
            target = batch["target_delta_residual"]
            model_mse.append(relative_mse(pred, target).cpu())
            zero_mse.append(relative_mse(torch.zeros_like(target), target).cpu())
            cosine.append(cosine_similarity(pred, target).cpu())
    learned = torch.cat(model_mse).mean().item()
    baseline = torch.cat(zero_mse).mean().item()
    report = {
        "learned_relative_mse": learned,
        "fixed_reuse_relative_mse": baseline,
        "relative_improvement": 1 - learned / baseline,
        "cosine_similarity": torch.cat(cosine).mean().item(),
        "gate": cfg["evaluation"]["gate_relative_mse_improvement"],
        "gate_passed": (1 - learned / baseline)
        >= cfg["evaluation"]["gate_relative_mse_improvement"],
        "examples": len(dataset),
    }
    print(json.dumps(report, indent=2))
    output = Path(args.checkpoint).parent / "offline_test_metrics.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
