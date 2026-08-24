#!/usr/bin/env python3
"""Summarize runtime grids and apply predeclared quality/speed gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grid",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Checkpoint label and runtime_grid_metrics.json path",
    )
    parser.add_argument("--min-speedup", type=float, default=1.2)
    parser.add_argument("--min-aggregate-mse-improvement", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_grids(values: list[str]) -> list[tuple[str, dict]]:
    grids = []
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise ValueError(f"invalid --grid value: {value!r}")
        with Path(path).open() as handle:
            grids.append((label, json.load(handle)))
    case_ids = [[case["case"] for case in grid["cases"]] for _, grid in grids]
    if any(ids != case_ids[0] for ids in case_ids[1:]):
        raise ValueError("runtime grids do not contain the same ordered cases")
    return grids


def summarize(grid: dict, variant: str) -> dict:
    cases = grid["cases"]
    improvements = [case["mse_improvement_over_fixed"][variant] for case in cases]
    variant_mses = [case["image_metrics_vs_exact"][variant]["mse"] for case in cases]
    fixed_mses = [case["image_metrics_vs_exact"]["fixed"]["mse"] for case in cases]
    variant_ssims = [case["image_metrics_vs_exact"][variant]["ssim"] for case in cases]
    fixed_ssims = [case["image_metrics_vs_exact"]["fixed"]["ssim"] for case in cases]
    speeds = [case["speedup_vs_exact"][variant] for case in cases]
    return {
        "speedup_mean": mean(speeds),
        "per_case_mse_improvement_mean": mean(improvements),
        "per_case_mse_improvement_std": pstdev(improvements),
        "per_case_mse_improvement_min": min(improvements),
        "per_case_mse_improvement_max": max(improvements),
        "mse_win_count": sum(value > 0 for value in improvements),
        "ssim_mean": mean(variant_ssims),
        "fixed_ssim_mean": mean(fixed_ssims),
        "ssim_win_count": sum(
            value > fixed for value, fixed in zip(variant_ssims, fixed_ssims)
        ),
        "aggregate_mse_improvement": 1.0 - mean(variant_mses) / mean(fixed_mses),
    }


def main() -> None:
    args = parse_args()
    grids = load_grids(args.grid)
    report = {
        "num_cases": len(grids[0][1]["cases"]),
        "gates": {
            "min_speedup": args.min_speedup,
            "min_aggregate_mse_improvement": args.min_aggregate_mse_improvement,
            "require_ssim_nondecreasing": True,
        },
        "checkpoints": {},
        "eligible_operating_points": [],
        "selected_operating_point": None,
    }
    variants = sorted(grids[0][1]["cases"][0]["mse_improvement_over_fixed"])
    for label, grid in grids:
        report["checkpoints"][label] = {}
        for variant in variants:
            summary = summarize(grid, variant)
            summary["passes_speed_gate"] = summary["speedup_mean"] >= args.min_speedup
            summary["passes_mse_gate"] = (
                summary["aggregate_mse_improvement"]
                >= args.min_aggregate_mse_improvement
            )
            summary["passes_ssim_gate"] = summary["ssim_mean"] >= summary["fixed_ssim_mean"]
            summary["eligible"] = (
                summary["passes_speed_gate"]
                and summary["passes_mse_gate"]
                and summary["passes_ssim_gate"]
            )
            report["checkpoints"][label][variant] = summary
            if summary["eligible"]:
                report["eligible_operating_points"].append(
                    {"checkpoint": label, "variant": variant, **summary}
                )

    if report["eligible_operating_points"]:
        report["selected_operating_point"] = max(
            report["eligible_operating_points"],
            key=lambda item: item["aggregate_mse_improvement"],
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
