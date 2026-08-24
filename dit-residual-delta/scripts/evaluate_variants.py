#!/usr/bin/env python
"""Image-level evaluation of block-reuse variants against exact same-seed output.

One entry point covers both pre-registered gates:

  G1 fixed-cache Pareto  -- variants that only set `cache_interval`/`anchor_steps`
  G2 oracle ceiling      -- variants that add `oracle_block_ids`, i.e. the true
                            residual is recomputed at those (step, block) slots.
                            That is the best any corrector could ever do there,
                            so a low ceiling rules the whole approach out
                            independent of predictor capacity or training data.

Every variant is measured on the same prompts and seeds as the exact reference.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import lpips
import torch
import torch.nn.functional as functional

from dit_residual_delta.config import load_config
from dit_residual_delta.metrics import (
    image_tensor,
    lpips_alex,
    numeric_summary,
    psnr,
    skimage_ssim,
)
from dit_residual_delta.pipeline import load_pixart_pipeline, load_prompt_embeddings
from dit_residual_delta.runtime import DiTBlockRuntime, DiTSegmentRuntime, uniform_segments
from dit_residual_delta.surrogate import load_surrogate_bank

METRICS = (
    "clip_score",
    "clip_score_vs_exact",
    "speedup_vs_exact",
    "mse_vs_exact",
    "psnr_vs_exact",
    "ssim_gaussian_vs_exact",
    "lpips_alex_vs_exact",
)


def resolve_anchor_steps(spec: dict, num_steps: int) -> set[int]:
    if "anchor_steps" in spec:
        return {int(s) for s in spec["anchor_steps"]}
    interval = int(spec.get("cache_interval", 2))
    return {s for s in range(num_steps) if s % interval == 0}


def build_variant(spec: dict, num_steps: int, num_blocks: int) -> dict:
    anchors = resolve_anchor_steps(spec, num_steps)
    reuse_steps = [s for s in range(num_steps) if s not in anchors]
    oracle_steps = [int(s) for s in spec.get("oracle_steps", reuse_steps)]
    oracle_blocks = [int(b) for b in spec.get("oracle_block_ids", [])]
    oracle = {(s, b) for b in oracle_blocks for s in oracle_steps} or None
    forced = {
        (int(s), int(b))
        for b in spec.get("forced_cache_block_ids", [])
        for s in spec.get("forced_cache_steps", sorted(anchors))
    } or None
    cached_blocks = spec.get("cached_block_ids")
    surrogate_blocks = spec.get("surrogate_block_ids")
    segment = spec.get("segment")
    num_segments = spec.get("num_segments")
    return {
        "segment": tuple(int(v) for v in segment) if segment else None,
        "num_segments": int(num_segments) if num_segments else None,
        "surrogate_checkpoint": spec.get("surrogate_checkpoint"),
        "surrogate_scale": float(spec.get("surrogate_scale", 1.0)),
        "adaptive_threshold": (float(spec["adaptive_threshold"])
                               if spec.get("adaptive_threshold") is not None else None),
        "taylor_order": int(spec.get("taylor_order", 0)),
        "surrogate_block_ids": {int(b) for b in surrogate_blocks} if surrogate_blocks else None,
        "name": spec["name"],
        "oracle_blend": float(spec.get("oracle_blend", 1.0)),
        "oracle_blend_or_none": (float(spec["oracle_blend"]) if "oracle_blend" in spec
                                 else (1.0 if spec.get("segment_oracle") else None)),
        "anchor_steps": anchors,
        "reuse_steps": reuse_steps,
        "oracle_refresh_keys": oracle,
        "forced_cache_keys": forced,
        "cached_block_ids": {int(b) for b in cached_blocks} if cached_blocks else None,
        "num_oracle_slots": 0 if oracle is None else len(oracle),
        "num_reuse_slots": len(reuse_steps) * num_blocks,
    }


_BANK_CACHE: dict[str, object] = {}


def surrogate_for(variant):
    path = variant.get("surrogate_checkpoint")
    if path is None:
        return None
    if path not in _BANK_CACHE:
        _BANK_CACHE[path] = load_surrogate_bank(path)
        info = getattr(_BANK_CACHE[path], "checkpoint_info", {})
        print(json.dumps({"loaded_surrogate": path, **info}), flush=True)
    return _BANK_CACHE[path]


def generate(pipeline, common, seed, variant=None):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        if variant is None:
            result = pipeline(**common, generator=generator)
            stats = None
        elif variant["segment"] is not None or variant["num_segments"] is not None:
            depth = len(pipeline.transformer.transformer_blocks)
            with DiTSegmentRuntime(
                pipeline.transformer,
                segment=variant["segment"] if variant["num_segments"] is None else None,
                segments=(uniform_segments(depth, variant["num_segments"])
                          if variant["num_segments"] else None),
                anchor_steps=variant["anchor_steps"],
                surrogate_bank=surrogate_for(variant),
                surrogate_scale=variant["surrogate_scale"],
                oracle_blend=variant["oracle_blend_or_none"],
                adaptive_threshold=variant["adaptive_threshold"],
                taylor_order=variant["taylor_order"],
            ) as runtime:
                result = pipeline(**common, generator=generator)
                stats = vars(runtime.stats)
        else:
            with DiTBlockRuntime(
                pipeline.transformer,
                cache_interval=999,
                anchor_steps=variant["anchor_steps"],
                cached_block_ids=variant["cached_block_ids"],
                oracle_refresh_keys=variant["oracle_refresh_keys"],
                oracle_blend=variant["oracle_blend"],
                forced_cache_keys=variant["forced_cache_keys"],
                surrogate_bank=surrogate_for(variant),
                surrogate_block_ids=variant["surrogate_block_ids"],
                surrogate_scale=variant["surrogate_scale"],
                taylor_order=variant["taylor_order"],
            ) as runtime:
                result = pipeline(**common, generator=generator)
                stats = vars(runtime.stats)
    torch.cuda.synchronize()
    return result.images[0], time.perf_counter() - start, stats


class _ClipScorer:
    """Cosine similarity between CLIP image and text embeddings, in [-1, 1]."""

    def __init__(self, name: str = "openai/clip-vit-base-patch32"):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(name).to("cuda").eval()
        self.processor = CLIPProcessor.from_pretrained(name)
        self._text_cache: dict[str, "torch.Tensor"] = {}

    def _text(self, prompt: str):
        if prompt not in self._text_cache:
            batch = self.processor(text=[prompt], return_tensors="pt",
                                   padding=True, truncation=True).to("cuda")
            with torch.no_grad():
                feat = self.model.get_text_features(**batch)
            self._text_cache[prompt] = feat / feat.norm(dim=-1, keepdim=True)
        return self._text_cache[prompt]

    def __call__(self, image, prompt: str) -> float:
        batch = self.processor(images=image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            feat = self.model.get_image_features(**batch)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return float((feat @ self._text(prompt).T).squeeze().item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--variants-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--clip-score", action="store_true",
                        help="also score prompt alignment. SSIM/LPIPS measure fidelity to\n                             the exact output; CLIP asks the different question of whether\n                             the accelerated image still matches the prompt as well.")
    parser.add_argument("--embeddings", required=True,
                        help="precomputed prompt embeddings (.pt) from prepare_assets.py")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_cfg = cfg["model"]
    num_steps = int(model_cfg["num_inference_steps"])
    prompts = [l.strip() for l in Path(args.prompt_file).read_text().splitlines() if l.strip()]
    specs = json.loads(Path(args.variants_file).read_text())

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pipeline = load_pixart_pipeline(with_text_encoder=False).to("cuda")
    embeddings = load_prompt_embeddings(args.embeddings)
    missing = [p for p in prompts if p not in embeddings]
    if missing:
        raise SystemExit(f"missing embeddings for {len(missing)} prompt(s): {missing[:2]}")
    num_blocks = len(pipeline.transformer.transformer_blocks)
    variants = [build_variant(s, num_steps, num_blocks) for s in specs]
    perceptual = lpips.LPIPS(net="alex").to("cuda").eval()
    clip_scorer = _ClipScorer() if args.clip_score else None

    common_base = {
        "height": int(model_cfg["height"]),
        "width": int(model_cfg["width"]),
        "num_inference_steps": num_steps,
        "guidance_scale": float(model_cfg["guidance_scale"]),
    }

    warm = {**common_base, **embeddings[prompts[0]]}
    generate(pipeline, warm, 999_001)
    generate(pipeline, warm, 999_001, variants[0])

    cases = []
    for prompt_index, prompt in enumerate(prompts):
        for seed in args.seeds:
            common = {**common_base, **embeddings[prompt]}
            exact_image, exact_seconds, _ = generate(pipeline, common, seed)
            exact_tensor = image_tensor(exact_image)
            exact_clip = clip_scorer(exact_image, prompt) if clip_scorer else None
            case_id = f"prompt-{prompt_index:02d}-seed-{seed}"
            if args.save_images:
                exact_image.save(out / f"{case_id}-exact.png")
            results = {}
            for variant in variants:
                image, elapsed, stats = generate(pipeline, common, seed, variant)
                candidate = image_tensor(image)
                results[variant["name"]] = {
                    "seconds": float(elapsed),
                    "speedup_vs_exact": float(exact_seconds / elapsed),
                    "mse_vs_exact": float(functional.mse_loss(candidate, exact_tensor).item()),
                    "psnr_vs_exact": psnr(candidate, exact_tensor),
                    "ssim_gaussian_vs_exact": skimage_ssim(image, exact_image, True),
                    "lpips_alex_vs_exact": lpips_alex(perceptual, candidate, exact_tensor),
                    "runtime": stats,
                }
                if clip_scorer is not None:
                    cs = clip_scorer(image, prompt)
                    results[variant["name"]]["clip_score"] = cs
                    results[variant["name"]]["clip_score_vs_exact"] = cs - exact_clip
                if args.save_images:
                    image.save(out / f"{case_id}-{variant['name']}.png")
            cases.append({
                "case": case_id, "prompt": prompt, "seed": seed,
                "exact_seconds": exact_seconds, "exact_clip_score": exact_clip,
                "variants": results,
            })
            (out / "cases.partial.json").write_text(json.dumps(cases, indent=2))
            print(json.dumps({"completed": len(cases), "case": case_id}), flush=True)

    # clip_score* are only present when --clip-score is passed
    def _summarise(name: str) -> dict:
        out = {}
        for metric in METRICS:
            vals = [c["variants"][name][metric] for c in cases if metric in c["variants"][name]]
            if vals:
                out[metric] = numeric_summary([float(v) for v in vals])
        return out

    aggregate = {variant["name"]: _summarise(variant["name"]) for variant in variants}
    payload = {
        "config": args.config,
        "prompt_file": args.prompt_file,
        "seeds": args.seeds,
        "num_steps": num_steps,
        "num_blocks": num_blocks,
        "variants": [
            {
                "name": v["name"],
                "anchor_steps": sorted(v["anchor_steps"]),
                "reuse_steps": v["reuse_steps"],
                "num_reuse_slots": v["num_reuse_slots"],
                "num_oracle_slots": v["num_oracle_slots"],
                "oracle_blend": v["oracle_blend"],
                "surrogate_checkpoint": v["surrogate_checkpoint"],
                "surrogate_scale": v["surrogate_scale"],
                "taylor_order": v["taylor_order"],
                "segment": v["segment"],
                "num_segments": v["num_segments"],
                "oracle_fraction": v["num_oracle_slots"] / v["num_reuse_slots"] if v["num_reuse_slots"] else 0.0,
            }
            for v in variants
        ],
        "aggregate": aggregate,
        "cases": cases,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({n: {m: round(v["mean"], 6) for m, v in s.items()} for n, s in aggregate.items()}, indent=1))


if __name__ == "__main__":
    main()
