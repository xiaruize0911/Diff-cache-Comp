#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import torch

from flux_residual_delta.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--write", default="runs/environment.json")
    args = parser.parse_args()
    config = load_config(args.config)
    packages = {}
    for name in ("torch", "diffusers", "transformers", "accelerate", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
        if torch.cuda.is_available()
        else None,
        "hf_token_present": bool(os.environ.get("HF_TOKEN")),
        "model_id": config["model"]["model_id"],
    }
    output = Path(args.write)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    missing = [name for name, version in packages.items() if version is None]
    if missing:
        raise SystemExit(f"Missing packages: {', '.join(missing)}")


if __name__ == "__main__":
    main()
