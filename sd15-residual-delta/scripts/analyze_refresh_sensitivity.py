#!/usr/bin/env python3
"""Compare oracle-refresh sensitivity rankings across prompt/seed cases."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def pearson(left: list[float], right: list[float]) -> float:
    left_mean, right_mean = mean(left), mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return numerator / denominator if denominator else 0.0


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        for position in range(start, end):
            result[ordered[position][0]] = average_rank
        start = end
    return result


def aligned_pair_metrics(left: dict, right: dict) -> dict:
    keys = sorted(left)
    if keys != sorted(right):
        raise ValueError("cases do not contain aligned sensitivity keys")
    left_values = [left[key] for key in keys]
    right_values = [right[key] for key in keys]
    return {
        "num_aligned": len(keys),
        "pearson": pearson(left_values, right_values),
        "spearman": pearson(ranks(left_values), ranks(right_values)),
        "sign_agreement": mean(
            [(left[key] > 0) == (right[key] > 0) for key in keys]
        ),
        "both_positive": sum(left[key] > 0 and right[key] > 0 for key in keys),
        "both_nonpositive": sum(left[key] <= 0 and right[key] <= 0 for key in keys),
    }


def main() -> None:
    args = parse_args()
    cases = []
    for value in args.case:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise ValueError(f"invalid --case value: {value!r}")
        with Path(path).open() as handle:
            cases.append((label, json.load(handle)))
    if len(cases) < 2:
        raise ValueError("at least two cases are required")

    report = {"cases": {}, "pairwise": [], "stable_positive_groups": []}
    intervention_maps = {}
    group_maps = {}
    group_names = {}
    for label, case in cases:
        values = [item["mse_improvement_over_fixed"] for item in case["interventions"]]
        report["cases"][label] = {
            "prompt": case["prompt"],
            "seed": case["seed"],
            "fixed_mse_vs_exact": case["fixed"]["mse_vs_exact"],
            "num_interventions": len(values),
            "mse_improvement_over_fixed": {
                "mean": mean(values),
                "std": pstdev(values),
                "min": min(values),
                "max": max(values),
            },
            "positive_rate": sum(value > 0 for value in values) / len(values),
        }
        intervention_maps[label] = {
            (item["step"], item["module_id"]): item["mse_improvement_over_fixed"]
            for item in case["interventions"]
        }
        group_maps[label] = {
            (item["module_id"], item["horizon"]): item[
                "mse_improvement_over_fixed"
            ]["mean"]
            for item in case["by_module_horizon"]
        }
        group_names.update(
            {
                (item["module_id"], item["horizon"]): item["module_name"]
                for item in case["by_module_horizon"]
            }
        )

    for left_index, (left_label, _) in enumerate(cases):
        for right_label, _ in cases[left_index + 1 :]:
            report["pairwise"].append(
                {
                    "left": left_label,
                    "right": right_label,
                    "intervention": aligned_pair_metrics(
                        intervention_maps[left_label], intervention_maps[right_label]
                    ),
                    "module_horizon_group": aligned_pair_metrics(
                        group_maps[left_label], group_maps[right_label]
                    ),
                }
            )

    group_keys = sorted(next(iter(group_maps.values())))
    for key in group_keys:
        values = {label: group_maps[label][key] for label, _ in cases}
        if all(value > 0 for value in values.values()):
            report["stable_positive_groups"].append(
                {
                    "module_id": key[0],
                    "horizon": key[1],
                    "module_name": group_names[key],
                    "case_values": values,
                    "mean_across_cases": mean(values.values()),
                }
            )
    report["stable_positive_groups"].sort(
        key=lambda item: item["mean_across_cases"], reverse=True
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
