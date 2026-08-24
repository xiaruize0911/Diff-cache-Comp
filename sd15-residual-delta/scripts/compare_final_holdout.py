#!/usr/bin/env python3
"""Compare two runtime-grid checkpoints on the same final holdout cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, pstdev


VARIANTS = ("learned", "learned_320_640")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-label", default="on_policy_step0")
    parser.add_argument("--candidate-label", default="rollout_step50")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def summarize(grid: dict, variant: str) -> dict:
    improvements = [case["mse_improvement_over_fixed"][variant] for case in grid["cases"]]
    mses = [case["image_metrics_vs_exact"][variant]["mse"] for case in grid["cases"]]
    fixed_mses = [case["image_metrics_vs_exact"]["fixed"]["mse"] for case in grid["cases"]]
    ssims = [case["image_metrics_vs_exact"][variant]["ssim"] for case in grid["cases"]]
    fixed_ssims = [case["image_metrics_vs_exact"]["fixed"]["ssim"] for case in grid["cases"]]
    return {
        "per_case_mse_improvement_mean": mean(improvements),
        "per_case_mse_improvement_std": pstdev(improvements),
        "per_case_mse_improvement_min": min(improvements),
        "per_case_mse_improvement_max": max(improvements),
        "mse_win_count": sum(value > 0 for value in improvements),
        "mse_win_rate": sum(value > 0 for value in improvements) / len(improvements),
        "ssim_win_count": sum(value > fixed for value, fixed in zip(ssims, fixed_ssims)),
        "ssim_win_rate": sum(value > fixed for value, fixed in zip(ssims, fixed_ssims)) / len(ssims),
        "aggregate_mean_mse": mean(mses),
        "aggregate_fixed_mean_mse": mean(fixed_mses),
        "aggregate_mean_mse_improvement": 1.0 - mean(mses) / mean(fixed_mses),
        "aggregate_mean_ssim": mean(ssims),
        "aggregate_fixed_mean_ssim": mean(fixed_ssims),
    }


def main() -> None:
    args = parse_args()
    baseline = load(args.baseline)
    candidate = load(args.candidate)
    baseline_cases = [case["case"] for case in baseline["cases"]]
    candidate_cases = [case["case"] for case in candidate["cases"]]
    if baseline_cases != candidate_cases:
        raise ValueError("runtime grids do not contain the same ordered cases")

    output = {
        "num_cases": len(baseline_cases),
        "case_ids": baseline_cases,
        "baseline_label": args.baseline_label,
        "candidate_label": args.candidate_label,
        "variants": {},
    }
    for variant in VARIANTS:
        base_summary = summarize(baseline, variant)
        candidate_summary = summarize(candidate, variant)
        output["variants"][variant] = {
            args.baseline_label: base_summary,
            args.candidate_label: candidate_summary,
            "candidate_minus_baseline": {
                key: candidate_summary[key] - base_summary[key]
                for key in (
                    "per_case_mse_improvement_mean",
                    "per_case_mse_improvement_std",
                    "per_case_mse_improvement_min",
                    "mse_win_rate",
                    "ssim_win_rate",
                    "aggregate_mean_mse_improvement",
                )
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
