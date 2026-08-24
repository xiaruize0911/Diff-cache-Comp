#!/usr/bin/env python
"""Count real FLOPs per generated image for each variant.

Uses torch's FlopCounterMode on an actual pipeline call, so VAE decode, the
patch embedding and every projection are included -- no hand-derived formula to
get wrong. Reported speedup is exact_flops / variant_flops.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

from dit_residual_delta.config import load_config
from dit_residual_delta.pipeline import load_pixart_pipeline, load_prompt_embeddings
from dit_residual_delta.runtime import DiTSegmentRuntime, uniform_segments
from dit_residual_delta.surrogate import load_surrogate_bank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pixart_sigma_512.toml")
    parser.add_argument("--variants-file", required=True)
    parser.add_argument("--embeddings", default="data/embeddings/design4.pt")
    parser.add_argument("--prompt-file", default="data/prompts/design4.txt")
    parser.add_argument("--output", default="runs/flops.json")
    args = parser.parse_args()

    cfg = load_config(args.config)["model"]
    prompt = [l.strip() for l in Path(args.prompt_file).read_text().splitlines() if l.strip()][0]
    pipeline = load_pixart_pipeline(with_text_encoder=False).to("cuda")
    embeddings = load_prompt_embeddings(args.embeddings)
    depth = len(pipeline.transformer.transformer_blocks)
    steps = int(cfg["num_inference_steps"])
    common = {"height": int(cfg["height"]), "width": int(cfg["width"]),
              "num_inference_steps": steps, "guidance_scale": float(cfg["guidance_scale"]),
              **embeddings[prompt]}
    specs = json.loads(Path(args.variants_file).read_text())
    banks = {}
    for spec in specs:
        path = spec.get("surrogate_checkpoint")
        if path and path not in banks:
            banks[path] = load_surrogate_bank(path)

    def measure(spec):
        counter = FlopCounterMode(display=False)
        generator = torch.Generator(device="cuda").manual_seed(4242)
        # NOT inference_mode: inference tensors bypass the Python dispatch
        # mode, so FlopCounterMode would silently report zero
        with torch.no_grad(), counter:
            if spec is None:
                pipeline(**common, generator=generator)
            else:
                interval = int(spec.get("cache_interval", 2))
                anchors = {s for s in range(steps) if s % interval == 0}
                with DiTSegmentRuntime(
                    pipeline.transformer,
                    segments=uniform_segments(depth, int(spec.get("num_segments", 1))),
                    anchor_steps=anchors,
                    surrogate_bank=banks.get(spec.get("surrogate_checkpoint")),
                    surrogate_scale=float(spec.get("surrogate_scale", 1.0)),
                ):
                    pipeline(**common, generator=generator)
        return counter.get_total_flops()

    exact = measure(None)
    print(f"{'variant':<16} {'TFLOPs/image':>13} {'vs exact':>10}")
    print(f"{'exact':<16} {exact/1e12:13.3f} {1.0:9.3f}x")
    results = {"exact": {"flops": exact, "speedup": 1.0}}
    for spec in specs:
        f = measure(spec)
        results[spec["name"]] = {"flops": f, "speedup": exact / f}
        print(f"{spec['name']:<16} {f/1e12:13.3f} {exact/f:9.3f}x", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
