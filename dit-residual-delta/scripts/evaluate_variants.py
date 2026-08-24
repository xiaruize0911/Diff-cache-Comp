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
from dit_residual_delta.surrogate import load_surrogate_bank
from dit_residual_delta.variants import build_variant, runtime_for_variant

METRICS = (
    "clip_score",
    "clip_score_vs_exact",
    "speedup_vs_exact",
    "mse_vs_exact",
    "psnr_vs_exact",
    "ssim_gaussian_vs_exact",
    "lpips_alex_vs_exact",
)


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
        else:
            with runtime_for_variant(pipeline.transformer, variant,
                                     surrogate_for(variant)) as runtime:
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
