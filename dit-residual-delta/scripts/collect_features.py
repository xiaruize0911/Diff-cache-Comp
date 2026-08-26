#!/usr/bin/env python
"""Capture on-policy residual-delta pairs from cached rollouts."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from dit_residual_delta.capture import SlotRecorder
from dit_residual_delta.config import load_config
from dit_residual_delta.pipeline import load_pixart_pipeline, load_prompt_embeddings
from dit_residual_delta.runtime import DiTBlockRuntime, DiTSegmentRuntime, uniform_segments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pixart_sigma_512.toml")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[4001])
    parser.add_argument("--slot-seed", type=int, default=7717,
                        help="seeds the token subsample, which was previously drawn "
                             "from the unseeded global RNG (banks were irreproducible)")
    parser.add_argument("--anchor-steps", nargs="+", type=int, default=[0, 5, 10, 15])
    parser.add_argument("--tokens-per-slot", type=int, default=48)
    parser.add_argument("--shard-size", type=int, default=8, help="prompts per shard file")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--surrogate-checkpoint", default=None,
                        help="capture ON the trajectory this C produces (DAgger round)")
    parser.add_argument("--surrogate-scale", type=float, default=1.0)
    parser.add_argument("--segment", nargs=2, type=int, default=None,
                        help="cache blocks [start, end) as ONE unit: one slot per reuse step")
    parser.add_argument("--num-segments", type=int, default=None,
                        help="split the whole stack into K uniform cached segments")
    args = parser.parse_args()

    cfg = load_config(args.config)["model"]
    prompts = [l.strip() for l in Path(args.prompt_file).read_text().splitlines() if l.strip()]
    if args.limit:
        prompts = prompts[: args.limit]
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pipeline = load_pixart_pipeline(with_text_encoder=False).to("cuda")
    embeddings = load_prompt_embeddings(args.embeddings)
    anchors = set(args.anchor_steps)
    common = {
        "height": int(cfg["height"]), "width": int(cfg["width"]),
        "num_inference_steps": int(cfg["num_inference_steps"]),
        "guidance_scale": float(cfg["guidance_scale"]),
    }

    bank = None
    if args.surrogate_checkpoint:
        from dit_residual_delta.surrogate import load_surrogate_bank
        bank = load_surrogate_bank(args.surrogate_checkpoint)
        print(json.dumps({"capture_on_policy_with": args.surrogate_checkpoint,
                          "scale": args.surrogate_scale, **bank.checkpoint_info}), flush=True)

    # The token subsample inside SlotRecorder draws from its own generator; leaving
    # it None meant it drew from the global CUDA RNG, which nothing here seeds, so
    # two runs of the same command produced different banks -- 0.3% apart in the
    # scale statistics and 1.5% in downstream rel-MSE, the same order as several
    # contrasts this project reports. Seeded explicitly and recorded in the manifest.
    slot_generator = torch.Generator(device="cuda").manual_seed(args.slot_seed)
    recorder = SlotRecorder(tokens_per_slot=args.tokens_per_slot,
                            generator=slot_generator)
    manifest = {"shards": [], "anchor_steps": sorted(anchors), "segment": args.segment,
                "num_segments": args.num_segments,
                "surrogate_checkpoint": args.surrogate_checkpoint,
                "surrogate_scale": args.surrogate_scale,
                "tokens_per_slot": args.tokens_per_slot,
                "prompt_file": args.prompt_file, "seeds": args.seeds,
                "slot_seed": args.slot_seed}
    shard_index, in_shard, start = 0, 0, time.perf_counter()
    for prompt_index, prompt in enumerate(prompts):
        for seed in args.seeds:
            generator = torch.Generator(device="cuda").manual_seed(seed + prompt_index)
            with torch.inference_mode():
                if args.num_segments:
                    runtime = DiTSegmentRuntime(
                        pipeline.transformer,
                        segments=uniform_segments(len(pipeline.transformer.transformer_blocks),
                                                  args.num_segments),
                        anchor_steps=anchors, reuse_observer=recorder,
                        surrogate_bank=bank, surrogate_scale=args.surrogate_scale)
                elif args.segment is None:
                    runtime = DiTBlockRuntime(pipeline.transformer, cache_interval=999,
                                              anchor_steps=anchors, reuse_observer=recorder)
                else:
                    runtime = DiTSegmentRuntime(pipeline.transformer, segment=tuple(args.segment),
                                                anchor_steps=anchors, reuse_observer=recorder,
                                                surrogate_bank=bank,
                                                surrogate_scale=args.surrogate_scale)
                with runtime:
                    pipeline(**common, **embeddings[prompt], generator=generator)
            in_shard += 1
        if in_shard >= args.shard_size or prompt_index == len(prompts) - 1:
            path = out / f"shard_{shard_index:04d}.pt"
            info = recorder.save(path)
            manifest["shards"].append({"file": path.name, "rollouts": in_shard, **info})
            print(json.dumps({"shard": shard_index, "rollouts_done": prompt_index + 1,
                              "seconds": round(time.perf_counter() - start, 1), **info}), flush=True)
            recorder.reset()
            shard_index += 1
            in_shard = 0
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"shards": len(manifest['shards']), "output": str(out)}))


if __name__ == "__main__":
    main()
