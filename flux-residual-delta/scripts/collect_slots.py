#!/usr/bin/env python
"""On-policy whole-stack slot capture for FLUX (K=1).

At every reuse slot the real 57-block stack runs once to get the true residual while
propagation keeps using the cached value, so recorded states are the ones the
corrector meets at deployment. This is the PixArt arm's collection design; that arm
showed teacher-forced capture (the SD1.5 approach) creates exposure bias, and also
that the residual distribution shift from going fully on-policy is only ~4%, so this
is the right and sufficient level of care.

Features go to /workspace (304 MB/s measured) not /dev/shm: a 5376-slot FLUX split is
~17 GB at dim 3072, which does not fit in the 4 GB left on tmpfs.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from flux_residual_delta.slot_capture import SlotRecorder
from flux_residual_delta.whole_stack import FluxWholeStackRuntime, anchor_steps_for


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/workspace/models/flux1_dev")
    p.add_argument("--embeddings", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[4001])
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--guidance", type=float, default=3.5)
    p.add_argument("--interval", type=int, default=5)
    p.add_argument("--tokens-per-slot", type=int, default=128)
    p.add_argument("--shard-size", type=int, default=8, help="rollouts per shard file")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--surrogate-checkpoint", default=None)
    p.add_argument("--surrogate-scale", type=float, default=1.0)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    from diffusers import FluxPipeline
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    store = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    prompts = store["prompts"][: args.limit] if args.limit else store["prompts"]

    pipe = FluxPipeline.from_pretrained(args.root, torch_dtype=torch.bfloat16,
                                        text_encoder=None, text_encoder_2=None,
                                        tokenizer=None, tokenizer_2=None).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    bank = None
    if args.surrogate_checkpoint:
        from flux_residual_delta.surrogate import load_surrogate_bank
        bank = load_surrogate_bank(args.surrogate_checkpoint)

    anchors = anchor_steps_for(args.steps, args.interval)
    recorder = SlotRecorder(tokens_per_slot=args.tokens_per_slot)
    manifest = {"anchor_steps": sorted(anchors), "interval": args.interval,
                "steps": args.steps, "height": args.height, "width": args.width,
                "guidance": args.guidance, "tokens_per_slot": args.tokens_per_slot,
                "seeds": args.seeds, "prompt_file": args.embeddings,
                "surrogate_checkpoint": args.surrogate_checkpoint,
                "surrogate_scale": args.surrogate_scale, "shards": []}
    shard_index = in_shard = 0
    start = time.perf_counter()
    for idx, prompt in enumerate(prompts):
        pe = store["prompt_embeds"][idx : idx + 1].to("cuda", torch.bfloat16)
        pooled = store["pooled_prompt_embeds"][idx : idx + 1].to("cuda", torch.bfloat16)
        for seed in args.seeds:
            g = torch.Generator("cuda").manual_seed(seed + idx)
            rt = FluxWholeStackRuntime(pipe.transformer, anchors, surrogate=bank,
                                       surrogate_scale=args.surrogate_scale,
                                       reuse_observer=recorder)
            with torch.no_grad(), rt:
                pipe(prompt_embeds=pe, pooled_prompt_embeds=pooled,
                     height=args.height, width=args.width,
                     num_inference_steps=args.steps, guidance_scale=args.guidance,
                     generator=g, output_type="latent")
            in_shard += 1
        if in_shard >= args.shard_size or idx == len(prompts) - 1:
            path = out / f"shard_{shard_index:04d}.pt"
            info = recorder.save(path)
            recorder.reset()
            manifest["shards"].append({"file": path.name, **info})
            print(json.dumps({"shard": shard_index, "prompts_done": idx + 1,
                              "seconds": round(time.perf_counter() - start, 1), **info}),
                  flush=True)
            shard_index += 1
            in_shard = 0
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(s["slots"] for s in manifest["shards"])
    print(json.dumps({"shards": len(manifest["shards"]), "total_slots": total,
                      "output": str(out), "seconds": round(time.perf_counter()-start, 1)}))


if __name__ == "__main__":
    main()
