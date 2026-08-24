#!/usr/bin/env python
"""Clean wall-clock benchmark: exact and variants interleaved, many repeats.

Speedups taken from the evaluation runs are unreliable -- those runs shared the
GPU with other jobs, and the exact reference drifted from 0.99 s to 2.18 s within
a single run. This measures each configuration A/B interleaved against the exact
reference in one exclusive process, and reports the per-pair ratio so any residual
drift cancels.

Refuses to start if another process holds GPU memory.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import time
from pathlib import Path

import torch

from dit_residual_delta.config import load_config
from dit_residual_delta.pipeline import load_pixart_pipeline, load_prompt_embeddings
from dit_residual_delta.runtime import DiTSegmentRuntime, uniform_segments
from dit_residual_delta.surrogate import load_surrogate_bank


def other_gpu_processes() -> list[str]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return []
    import os
    mine = str(os.getpid())
    return [l for l in out.splitlines() if l.strip() and l.split(",")[0].strip() != mine]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pixart_sigma_512.toml")
    parser.add_argument("--variants-file", required=True)
    parser.add_argument("--embeddings", default="data/embeddings/design4.pt")
    parser.add_argument("--prompt-file", default="data/prompts/design4.txt")
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", default="runs/speed_benchmark.json")
    parser.add_argument("--allow-contention", action="store_true")
    args = parser.parse_args()

    busy = other_gpu_processes()
    if busy and not args.allow_contention:
        raise SystemExit(f"another process is on the GPU, timings would be junk: {busy}")

    cfg = load_config(args.config)["model"]
    prompt = [l.strip() for l in Path(args.prompt_file).read_text().splitlines() if l.strip()][0]
    pipeline = load_pixart_pipeline(with_text_encoder=False).to("cuda")
    embeddings = load_prompt_embeddings(args.embeddings)
    depth = len(pipeline.transformer.transformer_blocks)
    common = {**{k: v for k, v in (("height", int(cfg["height"])), ("width", int(cfg["width"])),
                                   ("num_inference_steps", int(cfg["num_inference_steps"])),
                                   ("guidance_scale", float(cfg["guidance_scale"])))},
              **embeddings[prompt]}
    specs = json.loads(Path(args.variants_file).read_text())
    banks = {}

    def run(spec):
        generator = torch.Generator(device="cuda").manual_seed(4242)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            if spec is None:
                pipeline(**common, generator=generator)
            else:
                interval = int(spec.get("cache_interval", 2))
                anchors = {s for s in range(int(cfg["num_inference_steps"])) if s % interval == 0}
                path = spec.get("surrogate_checkpoint")
                if path and path not in banks:
                    banks[path] = load_surrogate_bank(path)
                with DiTSegmentRuntime(
                    pipeline.transformer,
                    segments=uniform_segments(depth, int(spec.get("num_segments", 1))),
                    anchor_steps=anchors,
                    surrogate_bank=banks.get(path),
                    surrogate_scale=float(spec.get("surrogate_scale", 1.0)),
                ):
                    pipeline(**common, generator=generator)
        torch.cuda.synchronize()
        return time.perf_counter() - start

    # preload every checkpoint BEFORE timing: /workspace reads at ~25 MB/s, and a
    # lazy load inside the timed region showed up as 4-7 s "variant time"
    for spec in specs:
        path = spec.get("surrogate_checkpoint")
        if path and path not in banks:
            banks[path] = load_surrogate_bank(path)
    print(f"preloaded {len(banks)} checkpoint(s)", flush=True)

    for _ in range(args.warmup):
        run(None)
        run(specs[0])

    results = {}
    for spec in specs:
        ratios, exact_times, variant_times = [], [], []
        for _ in range(args.repeats):
            e = run(None)          # interleaved: any clock drift hits both
            v = run(spec)
            ratios.append(e / v); exact_times.append(e); variant_times.append(v)
        results[spec["name"]] = {
            "speedup_mean": st.mean(ratios), "speedup_sd": st.stdev(ratios),
            "variant_seconds": st.mean(variant_times), "exact_seconds": st.mean(exact_times),
            "repeats": args.repeats,
        }
        r = results[spec["name"]]
        print(f"{spec['name']:<14} {r['speedup_mean']:6.3f}x +-{r['speedup_sd']:.3f}  "
              f"({r['variant_seconds']*1000:6.1f} ms vs exact {r['exact_seconds']*1000:6.1f} ms)", flush=True)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"results": results, "gpu_exclusive": not busy}, indent=2))


if __name__ == "__main__":
    main()
