#!/usr/bin/env python
"""Compare configurations across DIFFERENT step counts against ONE fixed reference.

Every other evaluation in this repo measures fidelity to the exact output at the
SAME step count, which is the right target for an accelerator meant to reproduce
what the model would have produced. It structurally cannot compare across step
counts: a 20-step variant is scored against a 20-step reference and a 50-step
variant against a 50-step one, so equal SSIM does not mean equal quality.

That makes one question unanswerable with the existing harness -- whether a budget
is better spent on more steps with more aggressive caching than on fewer steps with
milder caching. This scores everything against a single high-step exact reference
so the comparison is about absolute quality per FLOP.

Each variant carries its own `steps`, so the pipeline is re-driven per variant.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import lpips
import torch

from dit_residual_delta.config import load_config
from dit_residual_delta.metrics import image_tensor, lpips_alex, numeric_summary, psnr, skimage_ssim
from dit_residual_delta.pipeline import load_pixart_pipeline, load_prompt_embeddings
from dit_residual_delta.surrogate import load_surrogate_bank
from dit_residual_delta.variants import build_variant, runtime_for_variant

_BANKS: dict[str, object] = {}


def bank_for(spec):
    path = spec.get("surrogate_checkpoint")
    if path is None:
        return None
    if path not in _BANKS:
        _BANKS[path] = load_surrogate_bank(path)
    return _BANKS[path]


def generate(pipeline, embed, common, steps, seed, spec=None):
    kwargs = {**common, **embed, "num_inference_steps": steps}
    generator = torch.Generator(device="cuda").manual_seed(seed)
    torch.cuda.synchronize(); start = time.perf_counter()
    with torch.inference_mode():
        if spec is None:
            result = pipeline(**kwargs, generator=generator)
        else:
            variant = build_variant(spec, steps, len(pipeline.transformer.transformer_blocks))
            with runtime_for_variant(pipeline.transformer, variant, bank_for(spec)):
                result = pipeline(**kwargs, generator=generator)
    torch.cuda.synchronize()
    return result.images[0], time.perf_counter() - start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="only height/width/guidance are read")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--reference-steps", type=int, default=50)
    ap.add_argument("--variants-file", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)["model"]
    common = {"height": int(cfg["height"]), "width": int(cfg["width"]),
              "guidance_scale": float(cfg["guidance_scale"])}
    pipeline = load_pixart_pipeline(with_text_encoder=False).to("cuda")
    prompts = [l.strip() for l in Path(args.prompt_file).read_text().splitlines() if l.strip()]
    embeddings = load_prompt_embeddings(args.embeddings)
    specs = json.loads(Path(args.variants_file).read_text())
    perceptual = lpips.LPIPS(net="alex").to("cuda").eval()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    warm = embeddings[prompts[0]]
    generate(pipeline, warm, common, args.reference_steps, 999_001)

    cases = []
    for pi, prompt in enumerate(prompts):
        for seed in args.seeds:
            embed = embeddings[prompt]
            ref_image, ref_seconds = generate(pipeline, embed, common,
                                              args.reference_steps, seed)
            ref = image_tensor(ref_image)
            row = {"case": f"prompt-{pi:02d}-seed-{seed}", "prompt": prompt, "seed": seed,
                   "reference_seconds": ref_seconds, "variants": {}}
            for spec in specs:
                steps = int(spec["steps"])
                image, seconds = generate(pipeline, embed, common, steps, seed,
                                          None if spec.get("exact") else spec)
                cand = image_tensor(image)
                row["variants"][spec["name"]] = {
                    # skimage_ssim takes PIL images; psnr and lpips take tensors.
                    "ssim_gaussian_vs_reference": skimage_ssim(image, ref_image),
                    "lpips_alex_vs_reference": lpips_alex(perceptual, cand, ref),
                    "psnr_vs_reference": psnr(cand, ref),
                    "seconds": seconds,
                    "speedup_vs_reference": ref_seconds / seconds,
                }
            cases.append(row)
            print(json.dumps({"completed": len(cases), "case": row["case"]}), flush=True)

    metrics = ("ssim_gaussian_vs_reference", "lpips_alex_vs_reference",
               "psnr_vs_reference", "speedup_vs_reference")
    aggregate = {spec["name"]: {m: numeric_summary(
        [c["variants"][spec["name"]][m] for c in cases]) for m in metrics} for spec in specs}
    (out / "results.json").write_text(json.dumps(
        {"reference_steps": args.reference_steps, "config": args.config,
         "prompt_file": args.prompt_file, "seeds": args.seeds,
         "variants": specs, "aggregate": aggregate, "cases": cases}, indent=1))
    print(json.dumps({"written": str(out / "results.json")}))


if __name__ == "__main__":
    main()
