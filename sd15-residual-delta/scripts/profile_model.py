#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from sd15_residual_delta.pipeline import load_sd15_pipeline
from sd15_residual_delta.runtime import find_transformer2d_modules


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/model_profile.json")
    args = parser.parse_args()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    pipeline = load_sd15_pipeline().to("cuda")
    targets = find_transformer2d_modules(pipeline.unet)
    channels = Counter(int(module.config.in_channels) for _, module in targets)
    pipeline_parameters = sum(
        parameter.numel()
        for component in (pipeline.unet, pipeline.vae, pipeline.text_encoder)
        for parameter in component.parameters()
    )
    report = {
        "pipeline_parameters": pipeline_parameters,
        "unet_parameters": sum(parameter.numel() for parameter in pipeline.unet.parameters()),
        "transformer2d_modules": len(targets),
        "channel_distribution": dict(sorted(channels.items())),
        "cuda_allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "cuda_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
        "target_names": [name for name, _ in targets],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
