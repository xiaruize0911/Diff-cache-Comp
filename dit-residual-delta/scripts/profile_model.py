#!/usr/bin/env python
"""G0 -- speed ceiling. Where does PixArt-Sigma's wall clock actually go?

SD1.5's fatal constraint was that the cacheable units (attention) were only 41%
of end-to-end time, capping any all-cache scheme at 1.70x. This measures the
same quantity for the DiT stack before anything else is built.

Attribution uses CUDA events on forward hooks; a separate hook-free run gives the
honest absolute latency (hooks add per-call overhead).
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from dit_residual_delta.pipeline import load_pixart_pipeline, load_prompt_embeddings
from dit_residual_delta.runtime import find_transformer_blocks

PROMPT = "a highly detailed photograph of a red fox standing in fresh snow"


class EventTimer:
    """Accumulate GPU time per named module using CUDA events."""

    def __init__(self) -> None:
        self.pairs: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = defaultdict(list)
        self._open: dict[str, torch.cuda.Event] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []

    def attach(self, name: str, module: torch.nn.Module) -> None:
        def pre_hook(_m, _a, _k=None):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            self._open[name] = start

        def post_hook(_m, _a, _o):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            start = self._open.pop(name, None)
            if start is not None:
                self.pairs[name].append((start, end))

        self.handles.append(module.register_forward_pre_hook(pre_hook))
        self.handles.append(module.register_forward_hook(post_hook))

    def totals_ms(self) -> dict[str, float]:
        torch.cuda.synchronize()
        return {
            name: float(sum(s.elapsed_time(e) for s, e in pairs))
            for name, pairs in self.pairs.items()
        }

    def counts(self) -> dict[str, int]:
        return {name: len(pairs) for name, pairs in self.pairs.items()}

    def detach(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def timed_generate(pipeline, steps: int, seed: int, call_kwargs: dict) -> float:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        pipeline(num_inference_steps=steps, generator=generator, **call_kwargs)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7101)
    parser.add_argument("--output", default="runs/model_profile.json")
    parser.add_argument("--embeddings", help="precomputed prompt embeddings (.pt); omit to run the T5")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    args = parser.parse_args()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    use_embeddings = args.embeddings is not None
    pipeline = load_pixart_pipeline(with_text_encoder=not use_embeddings).to("cuda")
    if use_embeddings:
        table = load_prompt_embeddings(args.embeddings)
        call_kwargs = dict(next(iter(table.values())))
    else:
        call_kwargs = {"prompt": PROMPT}
    call_kwargs.update({"height": args.height, "width": args.width,
                        "guidance_scale": args.guidance_scale})
    transformer = pipeline.transformer
    blocks = find_transformer_blocks(transformer)

    static = {
        "model": "PixArt-Sigma-XL-2-512-MS",
        "num_blocks": len(blocks),
        "hidden_dim": int(transformer.config.num_attention_heads * transformer.config.attention_head_dim),
        "num_heads": int(transformer.config.num_attention_heads),
        "patch_size": int(transformer.config.patch_size),
        "transformer_parameters": sum(p.numel() for p in transformer.parameters()),
        "vae_parameters": sum(p.numel() for p in pipeline.vae.parameters()),
        "text_encoder_parameters": (
            0 if pipeline.text_encoder is None
            else sum(p.numel() for p in pipeline.text_encoder.parameters())
        ),
        "text_encoder_in_run": not use_embeddings,
    }

    # 1. Clean latency, no hooks.
    timed_generate(pipeline, args.steps, args.seed, call_kwargs)  # warm-up
    clean = [timed_generate(pipeline, args.steps, args.seed + i, call_kwargs) for i in range(args.repeats)]
    clean_mean = sum(clean) / len(clean)

    # 2. Attribution run with hooks.
    timer = EventTimer()
    timer.attach("transformer", transformer)
    timer.attach("vae_decode", pipeline.vae.decoder)
    if pipeline.text_encoder is not None:
        timer.attach("text_encoder", pipeline.text_encoder)
    for name, block in blocks:
        timer.attach(f"block/{name}", block)
        timer.attach(f"attn1/{name}", block.attn1)
        if getattr(block, "attn2", None) is not None:
            timer.attach(f"attn2/{name}", block.attn2)
        timer.attach(f"ff/{name}", block.ff)
    hooked_seconds = timed_generate(pipeline, args.steps, args.seed, call_kwargs)
    totals = timer.totals_ms()
    counts = timer.counts()
    timer.detach()

    def group(prefix: str) -> float:
        return sum(v for k, v in totals.items() if k.startswith(prefix))

    blocks_ms = group("block/")
    attn1_ms, attn2_ms, ff_ms = group("attn1/"), group("attn2/"), group("ff/")
    transformer_ms = totals.get("transformer", 0.0)
    hooked_ms = hooked_seconds * 1000.0

    per_block_ms = {
        name.split("/", 1)[1]: value / max(counts[name], 1)
        for name, value in totals.items()
        if name.startswith("block/")
    }

    report = {
        **static,
        "steps": args.steps,
        "clean_latency_seconds": {"mean": clean_mean, "runs": clean},
        "hooked_latency_seconds": hooked_seconds,
        "hook_overhead_ratio": hooked_seconds / clean_mean,
        "gpu_ms": {
            "transformer_total": transformer_ms,
            "blocks_total": blocks_ms,
            "attn1_self_total": attn1_ms,
            "attn2_cross_total": attn2_ms,
            "ff_total": ff_ms,
            "vae_decode": totals.get("vae_decode", 0.0),
            "text_encoder": totals.get("text_encoder", 0.0),
        },
        "share_of_hooked_wall": {
            "transformer": transformer_ms / hooked_ms,
            "blocks": blocks_ms / hooked_ms,
            "attention_only": (attn1_ms + attn2_ms) / hooked_ms,
            "vae_decode": totals.get("vae_decode", 0.0) / hooked_ms,
            "text_encoder": totals.get("text_encoder", 0.0) / hooked_ms,
        },
        "blocks_share_of_transformer": blocks_ms / transformer_ms if transformer_ms else None,
        "all_cache_speedup_ceiling": hooked_ms / (hooked_ms - blocks_ms) if hooked_ms > blocks_ms else None,
        "transformer_calls": counts.get("transformer", 0),
        "per_block_ms_mean": per_block_ms,
        "cuda_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_block_ms_mean"}, indent=2))


if __name__ == "__main__":
    main()
