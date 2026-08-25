#!/usr/bin/env python
"""Score ImageReward and same-seed fidelity together, on one set of images.

TaylorSeer's "almost lossless at 4.99x" is measured by ImageReward, and it reports
no fidelity-to-exact-output metric at all. Every number in this repository is the
other kind -- SSIM/LPIPS against the same-seed exact output. Those measure different
things: an image that differs from what the model would have produced but is equally
good scores well on ImageReward and badly on fidelity. So the two claims cannot be
compared, and conceding on the basis of that comparison is unfounded.

This measures BOTH on the same images, which settles what "lossless by ImageReward"
is actually compatible with in this setting. If ImageReward barely moves while SSIM
falls to a level where the image is visibly different, then the claim is weak
evidence of preservation, independent of anyone's specific number.

Fidelity reference is the exact output at the model's normal step count, so
fewer-step exact variants are scored against it too.
"""
from __future__ import annotations

import argparse, json, statistics as st, time
from pathlib import Path

import lpips
import torch

from dit_residual_delta.config import load_config
from dit_residual_delta.metrics import image_tensor, lpips_alex, numeric_summary, skimage_ssim
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
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--variants-file", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    import ImageReward as RM
    reward = RM.load("ImageReward-v1.0", device="cuda")

    cfg = load_config(args.config)["model"]
    ref_steps = int(cfg["num_inference_steps"])
    common = {"height": int(cfg["height"]), "width": int(cfg["width"]),
              "guidance_scale": float(cfg["guidance_scale"])}
    pipeline = load_pixart_pipeline(with_text_encoder=False).to("cuda")
    prompts = [l.strip() for l in Path(args.prompt_file).read_text().splitlines() if l.strip()]
    if args.limit:
        prompts = prompts[: args.limit]
    embeddings = load_prompt_embeddings(args.embeddings)
    specs = json.loads(Path(args.variants_file).read_text())
    perceptual = lpips.LPIPS(net="alex").to("cuda").eval()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    scratch = out / "_img"; scratch.mkdir(exist_ok=True)

    generate(pipeline, embeddings[prompts[0]], common, ref_steps, 999_001)

    cases = []
    for pi, prompt in enumerate(prompts):
        for seed in args.seeds:
            embed = embeddings[prompt]
            ref_image, _ = generate(pipeline, embed, common, ref_steps, seed)
            ref_path = scratch / "ref.png"; ref_image.save(ref_path)
            ref = image_tensor(ref_image)
            row = {"case": f"prompt-{pi:03d}-seed-{seed}", "prompt": prompt,
                   "exact_reward": reward.score(prompt, str(ref_path)), "variants": {}}
            for spec in specs:
                steps = int(spec["steps"])
                image, seconds = generate(pipeline, embed, common, steps, seed,
                                          None if spec.get("exact") else spec)
                path = scratch / "cand.png"; image.save(path)
                cand = image_tensor(image)
                row["variants"][spec["name"]] = {
                    "reward": reward.score(prompt, str(path)),
                    "ssim_vs_exact": skimage_ssim(image, ref_image),
                    "lpips_vs_exact": lpips_alex(perceptual, cand, ref),
                    "seconds": seconds}
            cases.append(row)
            if len(cases) % 20 == 0:
                print(json.dumps({"completed": len(cases)}), flush=True)

    exact_reward = [c["exact_reward"] for c in cases]
    aggregate = {"exact_reward": numeric_summary(exact_reward)}
    for spec in specs:
        n = spec["name"]
        aggregate[n] = {m: numeric_summary([c["variants"][n][m] for c in cases])
                        for m in ("reward", "ssim_vs_exact", "lpips_vs_exact", "seconds")}
        # paired reward delta against the same-seed exact output
        dif = [c["variants"][n]["reward"] - c["exact_reward"] for c in cases]
        aggregate[n]["reward_minus_exact"] = numeric_summary(dif)
    (out / "results.json").write_text(json.dumps(
        {"reference_steps": ref_steps, "n": len(cases), "variants": specs,
         "aggregate": aggregate, "cases": cases}, indent=1))
    print(json.dumps({"written": str(out / "results.json"), "n": len(cases)}))


if __name__ == "__main__":
    main()
