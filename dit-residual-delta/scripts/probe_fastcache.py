#!/usr/bin/env python
"""Skip rate and delta distribution for FastCache thresholds -- no image metrics.

The block-level skip rate is what sets FastCache's FLOPs, and the evaluate script
does not surface runtime stats, so this reports them directly.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from dit_residual_delta.config import load_config
from dit_residual_delta.pipeline import load_pixart_pipeline, load_prompt_embeddings
from dit_residual_delta.runtime import FastCacheRuntime, load_fastcache_maps

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--prompt-file", required=True)
ap.add_argument("--embeddings", required=True)
ap.add_argument("--maps", required=True)
ap.add_argument("--seeds", nargs="+", type=int, required=True)
ap.add_argument("--thresholds", nargs="+", type=float, required=True)
ap.add_argument("--output", required=True)
a = ap.parse_args()

cfg = load_config(a.config)["model"]
pipe = load_pixart_pipeline(with_text_encoder=False).to("cuda")
prompts = [l.strip() for l in Path(a.prompt_file).read_text().splitlines() if l.strip()]
emb = load_prompt_embeddings(a.embeddings)
maps = load_fastcache_maps(a.maps)
common = {"height": int(cfg["height"]), "width": int(cfg["width"]),
          "num_inference_steps": int(cfg["num_inference_steps"]),
          "guidance_scale": float(cfg["guidance_scale"])}

out = {}
for thr in a.thresholds:
    ex = re = 0
    deltas = []
    for p in prompts:
        for seed in a.seeds:
            gen = torch.Generator(device="cuda").manual_seed(seed)
            with torch.inference_mode(), FastCacheRuntime(pipe.transformer, maps, thr) as rt:
                pipe(**common, **emb[p], generator=gen)
                ex += rt.stats.exact_block_calls
                re += rt.stats.reused_block_calls
                deltas += rt.deltas
    total = ex + re
    deltas.sort()
    out[f"{thr:.4f}"] = {
        "skip_rate": re / total, "exact_block_calls": ex, "reused_block_calls": re,
        "delta_median": deltas[len(deltas) // 2] if deltas else None,
        "delta_p10": deltas[int(0.1 * len(deltas))] if deltas else None,
        "delta_p90": deltas[int(0.9 * len(deltas))] if deltas else None,
    }
    print(json.dumps({"threshold": thr, "skip_rate": round(re / total, 4)}), flush=True)
Path(a.output).write_text(json.dumps(out, indent=1))
