#!/usr/bin/env python3
"""Compute paired case-wise gains and bootstrap confidence intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator, samples: int) -> list[float]:
    draws = values[rng.integers(0, len(values), size=(samples, len(values)))]
    return np.quantile(draws.mean(axis=1), [0.025, 0.975]).tolist()


def summarize(values: np.ndarray, rng: np.random.Generator, samples: int) -> dict:
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "positive_cases": int((values > 0).sum()),
        "total_cases": int(len(values)),
        "bootstrap_95": bootstrap_interval(values, rng, samples),
        "per_case": values.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_821)
    args = parser.parse_args()

    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    cases = report["cases"]
    fixed_name = "fixed"
    candidate_name = args.candidate

    def relative_reduction(metric: str) -> np.ndarray:
        return np.asarray(
            [
                100.0
                * (
                    1.0
                    - case["variants"][candidate_name][metric]
                    / case["variants"][fixed_name][metric]
                )
                for case in cases
            ],
            dtype=np.float64,
        )

    ssim_delta = np.asarray(
        [
            case["variants"][candidate_name]["ssim_skimage_gaussian_vs_exact"]
            - case["variants"][fixed_name]["ssim_skimage_gaussian_vs_exact"]
            for case in cases
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(args.seed)
    result = {
        "input": str(Path(args.input)),
        "fixed_variant": fixed_name,
        "candidate_variant": candidate_name,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.seed,
        "mse_reduction_pct": summarize(
            relative_reduction("mse_vs_exact"), rng, args.bootstrap_samples
        ),
        "lpips_reduction_pct": summarize(
            relative_reduction("lpips_alex_vs_exact"), rng, args.bootstrap_samples
        ),
        "gaussian_ssim_absolute_delta": summarize(
            ssim_delta, rng, args.bootstrap_samples
        ),
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
