#!/usr/bin/env python
"""Precompute FLUX prompt embeddings once, so every later run loads no text encoder.

Same pattern as the PixArt arm: T5 is 9.5 GB and only needed here. After this the
transformer (24 GB) runs alone, which is what makes the experiments fit on one A40.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/workspace/models/flux1_dev")
    p.add_argument("--prompt-file", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-sequence-length", type=int, default=512)
    args = p.parse_args()

    from diffusers import FluxPipeline
    prompts = [l.strip() for l in Path(args.prompt_file).read_text().splitlines() if l.strip()]
    pipe = FluxPipeline.from_pretrained(args.root, torch_dtype=torch.bfloat16,
                                        transformer=None, vae=None)
    pipe.to("cuda")
    store = {"prompts": prompts, "prompt_embeds": [], "pooled_prompt_embeds": []}
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            pe, pooled, _ = pipe.encode_prompt(
                prompt=prompt, prompt_2=prompt, device="cuda",
                max_sequence_length=args.max_sequence_length)
            store["prompt_embeds"].append(pe.squeeze(0).cpu())
            store["pooled_prompt_embeds"].append(pooled.squeeze(0).cpu())
            print(f"{i+1}/{len(prompts)}", flush=True)
    store["prompt_embeds"] = torch.stack(store["prompt_embeds"])
    store["pooled_prompt_embeds"] = torch.stack(store["pooled_prompt_embeds"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(store, args.output)
    print({"saved": args.output, "prompt_embeds": list(store["prompt_embeds"].shape),
           "pooled": list(store["pooled_prompt_embeds"].shape)})


if __name__ == "__main__":
    main()
