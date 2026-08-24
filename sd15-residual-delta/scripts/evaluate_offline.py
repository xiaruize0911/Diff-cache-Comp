#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sd15_residual_delta.config import load_config
from sd15_residual_delta.dataset import ResidualPairDataset, identity_collate
from sd15_residual_delta.factory import build_surrogate_bank
from sd15_residual_delta.metrics import cosine_similarity, relative_mse
from sd15_residual_delta.train import prepare


def summarize(records: list[dict[str, float]]) -> dict[str, float | int]:
    count = len(records)
    learned = sum(record["relative_mse"] for record in records) / count
    fixed = sum(record["fixed_relative_mse"] for record in records) / count
    return {
        "examples": count,
        "learned_relative_mse": learned,
        "fixed_reuse_relative_mse": fixed,
        "relative_improvement": 1.0 - learned / fixed,
        "cosine_similarity": sum(record["cosine"] for record in records) / count,
        "absolute_mse": sum(record["absolute_mse"] for record in records) / count,
        "target_mean_square": sum(record["target_mean_square"] for record in records)
        / count,
    }


def summarize_groups(groups: dict[str, list[dict[str, float]]]) -> dict[str, dict]:
    return {key: summarize(groups[key]) for key in sorted(groups)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-id", action="append")
    parser.add_argument("--output")
    args = parser.parse_args()
    cfg = load_config(args.config)
    target_steps = (
        {int(step) for step in cfg["train"].get("target_steps", [])} or None
    )
    dataset = ResidualPairDataset(
        cfg["capture"]["output_dir"],
        "test",
        int(cfg["train"]["seed"]),
        target_steps,
    )
    if args.prompt_id:
        selected_prompts = set(args.prompt_id)
        dataset.records = [
            record for record in dataset.records if record["prompt_id"] in selected_prompts
        ]
        if not dataset.records:
            raise ValueError(f"No test records found for prompt IDs: {sorted(selected_prompts)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=identity_collate)
    device = torch.device("cuda")
    model = build_surrogate_bank(cfg).to(device=device, dtype=torch.bfloat16).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    records: list[dict[str, float]] = []
    grouped: dict[str, dict[str, list[dict[str, float]]]] = {
        "horizon": defaultdict(list),
        "target_step": defaultdict(list),
        "channels": defaultdict(list),
        "resolution": defaultdict(list),
        "module": defaultdict(list),
    }
    with torch.inference_mode():
        for item in loader:
            batch = prepare(item, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(
                    batch["delta_h"],
                    batch["anchor_residual"],
                    batch["timestep"],
                    batch["horizon"],
                    batch["module_id"],
                )
            error = relative_mse(prediction, batch["target"]).mean().item()
            fixed_error = relative_mse(torch.zeros_like(prediction), batch["target"]).mean().item()
            cosine = cosine_similarity(prediction, batch["target"]).mean().item()
            absolute_mse = (
                (prediction.float() - batch["target"].float()).square().mean().item()
            )
            target_mean_square = batch["target"].float().square().mean().item()
            record = {
                "relative_mse": error,
                "fixed_relative_mse": fixed_error,
                "cosine": cosine,
                "absolute_mse": absolute_mse,
                "target_mean_square": target_mean_square,
            }
            records.append(record)
            height, width = item["delta_h"].shape[-2:]
            keys = {
                "horizon": str(int(item["horizon"])),
                "target_step": str(int(item["target_step"])),
                "channels": str(int(item["channels"])),
                "resolution": f"{height}x{width}",
                "module": f'{int(item["module_id"]):02d}:{item["module_name"]}',
            }
            for group_name, key in keys.items():
                grouped[group_name][key].append(record)

    overall = summarize(records)
    improvement = float(overall["relative_improvement"])
    report = {
        **overall,
        "by_horizon": summarize_groups(grouped["horizon"]),
        "by_target_step": summarize_groups(grouped["target_step"]),
        "by_channels": summarize_groups(grouped["channels"]),
        "by_resolution": summarize_groups(grouped["resolution"]),
        "by_module": summarize_groups(grouped["module"]),
        "gate": cfg["evaluation"]["gate_relative_mse_improvement"],
        "gate_passed": improvement >= cfg["evaluation"]["gate_relative_mse_improvement"],
        "examples": len(dataset),
        "prompt_ids": sorted({record["prompt_id"] for record in dataset.records}),
    }
    print(json.dumps(report, indent=2))
    output = (
        Path(args.output)
        if args.output
        else Path(args.checkpoint).parent / "offline_test_metrics.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
