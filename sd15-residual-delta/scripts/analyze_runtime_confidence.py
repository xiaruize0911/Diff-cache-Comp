#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from sd15_residual_delta.config import load_config
from sd15_residual_delta.factory import build_surrogate_bank
from sd15_residual_delta.pipeline import load_sd15_pipeline
from sd15_residual_delta.runtime import SD15AttentionRuntime


def vector_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def correlations(features: list[dict], target: list[float]) -> dict[str, dict]:
    target_array = np.asarray(target, dtype=np.float64)
    report = {}
    for key in features[0]:
        values = np.asarray([feature[key] for feature in features], dtype=np.float64)
        if values.std() == 0 or target_array.std() == 0:
            pearson = spearman = float("nan")
        else:
            pearson = float(np.corrcoef(values, target_array)[0, 1])
            value_ranks = np.argsort(np.argsort(values))
            target_ranks = np.argsort(np.argsort(target_array))
            spearman = float(np.corrcoef(value_ranks, target_ranks)[0, 1])
        report[key] = {"pearson": pearson, "spearman": spearman}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--runtime-grid-metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    source = json.loads(Path(args.runtime_grid_metrics).read_text(encoding="utf-8"))
    pipeline = load_sd15_pipeline(cfg["model"]["model_id"]).to("cuda")
    surrogate = build_surrogate_bank(cfg).to(device="cuda", dtype=torch.float16).eval()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    surrogate.load_state_dict(state["model"])
    common_base = {
        "height": cfg["model"]["height"],
        "width": cfg["model"]["width"],
        "num_inference_steps": cfg["model"]["num_inference_steps"],
        "guidance_scale": cfg["model"]["guidance_scale"],
    }

    # One unmeasured warmup avoids mixing allocator initialization into the first
    # diagnostic trajectory. Timing is deliberately not reported for this tool.
    with torch.inference_mode():
        pipeline(
            **common_base,
            prompt=source["cases"][0]["prompt"],
            generator=torch.Generator(device="cuda").manual_seed(999_999),
        )

    output_cases = []
    flat_features = []
    targets = []
    for source_case in source["cases"]:
        scalar_records: list[dict[str, torch.Tensor | int]] = []

        def observe(**item):
            scalar_records.append(
                {
                    "module_id": item["module_id"],
                    "horizon": item["horizon"],
                    "channels": item["delta_h"].shape[1],
                    "delta_h_ms": item["delta_h"].float().square().mean(),
                    "anchor_hidden_ms": item["anchor_hidden"].float().square().mean(),
                    "anchor_residual_ms": item["anchor_residual"].float().square().mean(),
                    "prediction_ms": item["predicted_delta_residual"]
                    .float()
                    .square()
                    .mean(),
                }
            )

        generator = torch.Generator(device="cuda").manual_seed(source_case["seed"])
        with torch.inference_mode(), SD15AttentionRuntime(
            pipeline.unet,
            cache_interval=int(cfg["reuse"]["cache_interval"]),
            surrogate_bank=surrogate,
            surrogate_observer=observe,
        ):
            pipeline(
                **common_base,
                prompt=source_case["prompt"],
                generator=generator,
            )

        eps = 1e-12
        delta_h_rms = torch.stack(
            [record["delta_h_ms"] for record in scalar_records]
        ).sqrt()
        anchor_hidden_rms = torch.stack(
            [record["anchor_hidden_ms"] for record in scalar_records]
        ).sqrt()
        anchor_residual_rms = torch.stack(
            [record["anchor_residual_ms"] for record in scalar_records]
        ).sqrt()
        prediction_rms = torch.stack(
            [record["prediction_ms"] for record in scalar_records]
        ).sqrt()
        vectors = {
            "input_drift_to_hidden": (
                delta_h_rms / anchor_hidden_rms.clamp_min(eps)
            ).cpu().numpy(),
            "prediction_to_anchor_residual": (
                prediction_rms / anchor_residual_rms.clamp_min(eps)
            ).cpu().numpy(),
            "prediction_to_input_drift": (
                prediction_rms / delta_h_rms.clamp_min(eps)
            ).cpu().numpy(),
            "prediction_rms": prediction_rms.cpu().numpy(),
        }
        summaries = {key: vector_summary(values) for key, values in vectors.items()}
        by_horizon = {}
        for horizon in (1, 2):
            indices = np.asarray(
                [record["horizon"] == horizon for record in scalar_records], dtype=bool
            )
            by_horizon[str(horizon)] = {
                key: vector_summary(values[indices]) for key, values in vectors.items()
            }
        by_channels = {}
        for channels in sorted({int(record["channels"]) for record in scalar_records}):
            indices = np.asarray(
                [record["channels"] == channels for record in scalar_records], dtype=bool
            )
            by_channels[str(channels)] = {
                key: vector_summary(values[indices]) for key, values in vectors.items()
            }

        feature = {
            f"{vector_name}_{stat_name}": stat_value
            for vector_name, summary in summaries.items()
            for stat_name, stat_value in summary.items()
        }
        improvement = source_case["mse_improvement_over_fixed"]["learned"]
        flat_features.append(feature)
        targets.append(improvement)
        output_cases.append(
            {
                "case": source_case["case"],
                "prompt_index": source_case["prompt_index"],
                "seed": source_case["seed"],
                "mse_improvement_over_fixed": improvement,
                "surrogate_calls": len(scalar_records),
                "overall": summaries,
                "by_horizon": by_horizon,
                "by_channels": by_channels,
            }
        )
        print(
            json.dumps(
                {
                    "completed": len(output_cases),
                    "total": len(source["cases"]),
                    "case": source_case["case"],
                    "mse_improvement_over_fixed": improvement,
                }
            ),
            flush=True,
        )

    correlation_report = correlations(flat_features, targets)
    ranked = sorted(
        correlation_report.items(),
        key=lambda item: abs(item[1]["spearman"]),
        reverse=True,
    )
    report = {
        "num_cases": len(output_cases),
        "source_runtime_grid_metrics": args.runtime_grid_metrics,
        "checkpoint_step": state.get("step"),
        "correlations_with_mse_improvement": correlation_report,
        "top_correlations_by_absolute_spearman": [
            {"feature": key, **value} for key, value in ranked[:10]
        ],
        "cases": output_cases,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["top_correlations_by_absolute_spearman"], indent=2))


if __name__ == "__main__":
    main()
