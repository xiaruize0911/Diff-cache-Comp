#!/usr/bin/env python
"""Residual-level predictor accuracy: decoupled from image pipeline / wall-clock noise.

Reports, on the held-out residual test split:
  - relative_mse       ||scale*pred - target||^2 / ||target||^2   (1.0 = fixed-cache baseline)
  - cosine_similarity  direction agreement between pred and target (scale-invariant)
  - magnitude_ratio    ||scale*pred|| / ||target||                (1.0 = correctly scaled)
"""
from __future__ import annotations

import argparse
import json

import torch
from torch.utils.data import DataLoader

from sd15_residual_delta.config import load_config
from sd15_residual_delta.dataset import ResidualPairDataset, identity_collate
from sd15_residual_delta.factory import build_surrogate_bank
from sd15_residual_delta.metrics import cosine_similarity, relative_mse
from sd15_residual_delta.train import prepare


@torch.no_grad()
def raw_predictions(model, loader, device):
    """One forward pass; cache raw (unscaled) prediction/target norms and cosine.

    cosine is invariant to any positive scalar rescaling of the prediction, so it
    only needs to be computed once regardless of how many scales are evaluated.
    """
    raw_norm, target_norm, cos, dot = [], [], [], []
    for item in loader:
        batch = prepare(item, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            raw_prediction = model(
                batch["delta_h"],
                batch["anchor_residual"],
                batch["timestep"],
                batch["horizon"],
                batch["module_id"],
            )
        target = batch["target"]
        p = raw_prediction.float().flatten(1)
        t = target.float().flatten(1)
        raw_norm.append(p.norm(dim=1).cpu())
        target_norm.append(t.norm(dim=1).cpu())
        cos.append(cosine_similarity(raw_prediction, target).float().cpu())
        dot.append((p * t).sum(dim=1).cpu())
    return {
        "raw_norm": torch.cat(raw_norm),
        "target_norm": torch.cat(target_norm),
        "cosine": torch.cat(cos),
        "dot": torch.cat(dot),
    }


def evaluate_at_scale(raw: dict, scale: float) -> dict:
    # relative_mse(scale*raw, target) expanded in closed form from cached norms/dot,
    # avoids a second forward pass per scale.
    raw_norm, target_norm, dot = raw["raw_norm"], raw["target_norm"], raw["dot"]
    error_sq = (scale**2) * raw_norm**2 - 2 * scale * dot + target_norm**2
    rel_mse = error_sq / target_norm.clamp_min(1e-8) ** 2
    mag_ratio = (scale * raw_norm) / target_norm.clamp_min(1e-8)
    return {
        "scale": scale,
        "examples": int(rel_mse.numel()),
        "relative_mse_mean": rel_mse.mean().item(),
        "fixed_reuse_relative_mse": 1.0,
        "relative_improvement_vs_fixed": 1.0 - rel_mse.mean().item(),
        "cosine_similarity_mean": raw["cosine"].mean().item(),
        "magnitude_ratio_mean": mag_ratio.mean().item(),
        "magnitude_ratio_median": mag_ratio.median().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="capture/dataset config (module14_s14)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-step", type=int, default=None)
    parser.add_argument("--scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    target_steps = {args.target_step} if args.target_step is not None else None
    dataset = ResidualPairDataset(
        cfg["capture"]["output_dir"],
        "test",
        int(cfg["train"]["seed"]),
        target_steps,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=identity_collate)

    device = torch.device("cuda")
    model = build_surrogate_bank(cfg)
    state = torch.load(args.checkpoint, map_location="cpu")
    state_dict = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(state_dict)
    model.to(device=device, dtype=torch.bfloat16).eval()

    raw = raw_predictions(model, loader, device)
    # least-squares-optimal scalar scale under the fitted direction: s* = <raw,target>/||raw||^2
    optimal_scale = (raw["dot"].sum() / (raw["raw_norm"] ** 2).sum().clamp_min(1e-8)).item()

    results = {f"scale_{scale}": evaluate_at_scale(raw, scale) for scale in args.scales}
    results["scale_optimal"] = evaluate_at_scale(raw, optimal_scale)

    payload = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "target_step": args.target_step,
        "dataset_examples": len(dataset.records),
        "optimal_scale": optimal_scale,
        "results": results,
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
