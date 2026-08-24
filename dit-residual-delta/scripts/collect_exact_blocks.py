#!/usr/bin/env python
"""Capture (block input, block residual) on an EXACT rollout, for FastCache maps.

FastCache has no anchor schedule: it runs a mostly-exact trajectory and replaces
individual blocks. Fitting its linear stand-in on slots captured from a cached
rollout would hand it a distribution it never sees at deployment, so we capture
on the exact trajectory instead. Tokens are subsampled per (step, block).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from dit_residual_delta.config import load_config
from dit_residual_delta.pipeline import load_pixart_pipeline, load_prompt_embeddings

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--prompt-file", required=True)
ap.add_argument("--embeddings", required=True)
ap.add_argument("--seeds", nargs="+", type=int, required=True)
ap.add_argument("--tokens-per-slot", type=int, default=48)
ap.add_argument("--shard-size", type=int, default=8, help="prompts per shard")
ap.add_argument("--output-dir", required=True)
a = ap.parse_args()

cfg = load_config(a.config)["model"]
pipe = load_pixart_pipeline(with_text_encoder=False).to("cuda")
prompts = [l.strip() for l in Path(a.prompt_file).read_text().splitlines() if l.strip()]
emb = load_prompt_embeddings(a.embeddings)
common = {"height": int(cfg["height"]), "width": int(cfg["width"]),
          "num_inference_steps": int(cfg["num_inference_steps"]),
          "guidance_scale": float(cfg["guidance_scale"])}
out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
blocks = list(pipe.transformer.transformer_blocks)
gen_index = torch.Generator(device="cuda").manual_seed(12345)

store = {"hidden": [], "residual": [], "block_id": [], "step": []}
step_box = {"i": 0}

def make_hook(block_id):
    def hook(module, args, kwargs, output):
        x = args[0] if args else kwargs["hidden_states"]
        flat_in = x.detach().reshape(-1, x.shape[-1])
        flat_res = (output.detach() - x.detach()).reshape(-1, x.shape[-1])
        n = min(a.tokens_per_slot, flat_in.shape[0])
        idx = torch.randperm(flat_in.shape[0], device=flat_in.device, generator=gen_index)[:n]
        store["hidden"].append(flat_in[idx].to("cpu", torch.float16))
        store["residual"].append(flat_res[idx].to("cpu", torch.float16))
        store["block_id"].append(block_id)
        store["step"].append(step_box["i"])
        return output
    return hook

def step_hook(module, args, kwargs, output):
    step_box["i"] += 1
    return output

handles = [b.register_forward_hook(make_hook(i), with_kwargs=True) for i, b in enumerate(blocks)]
handles.append(pipe.transformer.register_forward_hook(step_hook, with_kwargs=True))

shard, in_shard, manifest = 0, 0, {"shards": [], "tokens_per_slot": a.tokens_per_slot,
                                   "prompt_file": a.prompt_file, "seeds": a.seeds,
                                   "trajectory": "exact"}
try:
    for pi, p in enumerate(prompts):
        for seed in a.seeds:
            step_box["i"] = 0
            g = torch.Generator(device="cuda").manual_seed(seed + pi)
            with torch.inference_mode():
                pipe(**common, **emb[p], generator=g)
            in_shard += 1
        if in_shard >= a.shard_size or pi == len(prompts) - 1:
            payload = {"hidden": torch.stack(store["hidden"]),
                       "residual": torch.stack(store["residual"]),
                       "block_id": torch.tensor(store["block_id"]),
                       "step": torch.tensor(store["step"])}
            torch.save(payload, out / f"shard_{shard:04d}.pt")
            print(json.dumps({"shard": shard, "slots": payload["hidden"].shape[0]}), flush=True)
            manifest["shards"].append(f"shard_{shard:04d}.pt")
            for k in store: store[k].clear()
            shard += 1; in_shard = 0
finally:
    for h in handles: h.remove()
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(json.dumps({"shards": shard, "output": str(out)}))
