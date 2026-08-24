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
from sd15_residual_delta.metrics import relative_mse
from sd15_residual_delta.train import prepare


def collect(model, dataset, device: torch.device) -> list[dict]:
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=identity_collate)
    records = []
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
            learned_error = relative_mse(prediction, batch["target"]).mean().item()
            fixed_error = relative_mse(
                torch.zeros_like(prediction), batch["target"]
            ).mean().item()
            records.append(
                {
                    "prompt_id": item["prompt_id"],
                    "module_id": int(item["module_id"]),
                    "module_name": item["module_name"],
                    "channels": int(item["channels"]),
                    "horizon": int(item["horizon"]),
                    "learned_error": learned_error,
                    "fixed_error": fixed_error,
                }
            )
    return records


def summarize(records: list[dict], selected: set[tuple[int, int]] | None) -> dict:
    fixed = sum(record["fixed_error"] for record in records) / len(records)
    learned = sum(record["learned_error"] for record in records) / len(records)
    if selected is None:
        gated = learned
        coverage = 1.0
    else:
        use_learned = [
            (record["module_id"], record["horizon"]) in selected for record in records
        ]
        gated = sum(
            record["learned_error"] if use else record["fixed_error"]
            for record, use in zip(records, use_learned, strict=True)
        ) / len(records)
        coverage = sum(use_learned) / len(use_learned)
    return {
        "examples": len(records),
        "fixed_relative_mse": fixed,
        "all_learned_relative_mse": learned,
        "gated_relative_mse": gated,
        "all_learned_improvement": 1.0 - learned / fixed,
        "gated_improvement": 1.0 - gated / fixed,
        "surrogate_call_fraction": coverage,
    }


def group_stats(records: list[dict]) -> dict[str, dict]:
    groups = defaultdict(list)
    for record in records:
        groups[(record["module_id"], record["horizon"])].append(record)
    report = {}
    for pair, items in sorted(groups.items()):
        fixed = sum(item["fixed_error"] for item in items) / len(items)
        learned = sum(item["learned_error"] for item in items) / len(items)
        first = items[0]
        report[f"{pair[0]:02d}:h{pair[1]}"] = {
            "module_id": pair[0],
            "horizon": pair[1],
            "module_name": first["module_name"],
            "channels": first["channels"],
            "examples": len(items),
            "fixed_relative_mse": fixed,
            "learned_relative_mse": learned,
            "relative_improvement": 1.0 - learned / fixed,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=[0.0, 0.05, 0.10, 0.20]
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda")
    model = build_surrogate_bank(cfg).to(device=device, dtype=torch.bfloat16).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])

    datasets = {
        split: ResidualPairDataset(
            cfg["capture"]["output_dir"], split, int(cfg["train"]["seed"])
        )
        for split in ("val", "test")
    }
    records = {split: collect(model, dataset, device) for split, dataset in datasets.items()}
    validation_groups = group_stats(records["val"])
    test_groups = group_stats(records["test"])

    policies = {}
    for threshold in args.thresholds:
        selected = {
            (stats["module_id"], stats["horizon"])
            for stats in validation_groups.values()
            if stats["relative_improvement"] >= threshold
        }
        policies[str(threshold)] = {
            "threshold": threshold,
            "selected_pairs": [list(pair) for pair in sorted(selected)],
            "validation": summarize(records["val"], selected),
            "test": summarize(records["test"], selected),
        }

    chosen_key = max(
        policies,
        key=lambda key: policies[key]["validation"]["gated_improvement"],
    )
    report = {
        "selection_split": "val",
        "evaluation_split": "test",
        "validation_prompt_ids": sorted(
            {record["prompt_id"] for record in records["val"]}
        ),
        "test_prompt_ids": sorted(
            {record["prompt_id"] for record in records["test"]}
        ),
        "checkpoint_step": state.get("step"),
        "validation_groups": validation_groups,
        "test_groups": test_groups,
        "policies": policies,
        "chosen_threshold": float(chosen_key),
        "chosen_policy": policies[chosen_key],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "chosen_threshold": report["chosen_threshold"],
        "chosen_policy": report["chosen_policy"],
    }, indent=2))


if __name__ == "__main__":
    main()
