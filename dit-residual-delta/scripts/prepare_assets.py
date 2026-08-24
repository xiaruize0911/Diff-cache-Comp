#!/usr/bin/env python
"""One-time asset prep: precompute T5 prompt embeddings, then shrink the weights.

After this runs, no experiment needs the 19 GB fp32 T5 again -- which matters
because it only fits in tmpfs and tmpfs does not survive a pod restart.

Outputs:
  data/embeddings/<name>.pt                     prompt + negative embeddings
  /workspace/models/pixart_sigma_512_fp16/      durable fp16 transformer + vae
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from dit_residual_delta.pipeline import FAST_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-files", nargs="+", required=True)
    parser.add_argument("--root", default=str(FAST_ROOT))
    parser.add_argument("--embeddings-dir", default="data/embeddings")
    parser.add_argument("--durable-root", default="/workspace/models/pixart_sigma_512_fp16")
    parser.add_argument("--skip-durable-copy", action="store_true")
    args = parser.parse_args()

    from diffusers import PixArtSigmaPipeline

    pipeline = PixArtSigmaPipeline.from_pretrained(args.root, torch_dtype=torch.float16)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.text_encoder.to("cuda")

    embeddings_dir = Path(args.embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for prompt_file in args.prompt_files:
        path = Path(prompt_file)
        prompts = [l.strip() for l in path.read_text().splitlines() if l.strip()]
        payload = {"prompts": {}, "source": str(path)}
        for prompt in prompts:
            with torch.no_grad():
                pe, pm, ne, nm = pipeline.encode_prompt(prompt, do_classifier_free_guidance=True, device="cuda")
            payload["prompts"][prompt] = {
                "prompt_embeds": pe.detach().cpu(),
                "prompt_attention_mask": pm.detach().cpu(),
            }
            payload["negative"] = {
                "prompt_embeds": ne.detach().cpu(),
                "prompt_attention_mask": nm.detach().cpu(),
            }
        destination = embeddings_dir / f"{path.stem}.pt"
        torch.save(payload, destination)
        manifest[path.stem] = {"file": str(destination), "count": len(prompts)}
        print(f"[embeddings] {destination} ({len(prompts)} prompts)", flush=True)

    if not args.skip_durable_copy:
        durable = Path(args.durable_root)
        durable.mkdir(parents=True, exist_ok=True)
        pipeline.transformer.to(torch.float16).save_pretrained(durable / "transformer")
        pipeline.vae.to(torch.float16).save_pretrained(durable / "vae")
        pipeline.scheduler.save_pretrained(durable / "scheduler")
        index = json.loads((Path(args.root) / "model_index.json").read_text())
        index["text_encoder"] = [None, None]
        index["tokenizer"] = [None, None]
        (durable / "model_index.json").write_text(json.dumps(index, indent=2))
        print(f"[durable] fp16 transformer + vae -> {durable}", flush=True)

    (embeddings_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
