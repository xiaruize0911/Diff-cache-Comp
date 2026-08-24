#!/usr/bin/env python
"""Distributional evaluation on COCO captions: FID against the exact model's own
output distribution, plus CLIP alignment and per-image fidelity.

Why FID against *exact* rather than against real photographs: an accelerator cannot
improve on the model it accelerates, so the question that matters is whether the
accelerated output distribution still matches the one the unaccelerated model would
have produced. Both sets here are generated from the same captions and the same seed,
so the reference is exactly that. We do not measure FID against real COCO images and
therefore make no claim about absolute generation quality.

Inception features are accumulated on the fly; no images are written to disk.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from scipy import linalg

from dit_residual_delta.config import load_config
from dit_residual_delta.metrics import image_tensor, lpips_alex, psnr, skimage_ssim
from dit_residual_delta.pipeline import load_pixart_pipeline, load_prompt_embeddings
from dit_residual_delta.runtime import DiTSegmentRuntime, uniform_segments


def kid(x, y, subsets=100, subset_size=500, seed=0):
    """Unbiased KID (polynomial-kernel MMD^2) -- the right statistic at this sample size.

    FID's covariance estimate is rank-deficient when n < 2048 and its small-sample bias
    is large, so we report FID only for like-for-like comparison against a common
    reference and lean on KID, whose estimator is unbiased.
    """
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    m = min(subset_size, x.shape[0], y.shape[0])
    vals = []
    for _ in range(subsets):
        a = x[rng.choice(x.shape[0], m, replace=False)]
        b = y[rng.choice(y.shape[0], m, replace=False)]
        kxx = (a @ a.T / d + 1) ** 3
        kyy = (b @ b.T / d + 1) ** 3
        kxy = (a @ b.T / d + 1) ** 3
        np.fill_diagonal(kxx, 0)
        np.fill_diagonal(kyy, 0)
        vals.append(kxx.sum() / (m * (m - 1)) + kyy.sum() / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def frechet(mu1, s1, mu2, s2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + offset).dot(s2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


class _Feats:
    """Streaming accumulator for Inception pool3 activations."""

    def __init__(self, model):
        self.model, self.buf = model, []

    def add(self, pil):
        x = torch.from_numpy(np.array(pil.convert("RGB"))).permute(2, 0, 1)[None].float() / 255.0
        with torch.no_grad():
            f = self.model(x.cuda())[0].squeeze(-1).squeeze(-1)
        self.buf.append(f.cpu().numpy().astype(np.float64))

    def raw(self):
        return np.concatenate(self.buf, 0)

    def stats(self):
        a = self.raw()
        return a.mean(0), np.cov(a, rowvar=False), a.shape[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pixart_sigma_512.toml")
    p.add_argument("--embeddings", required=True)
    p.add_argument("--variants-file", required=True)
    p.add_argument("--seed", type=int, default=7001)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    import lpips as lpips_pkg
    from pytorch_fid.inception import InceptionV3
    from transformers import CLIPModel, CLIPProcessor

    cfg = load_config(args.config)["model"]
    steps = int(cfg["num_inference_steps"])
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    specs = json.loads(Path(args.variants_file).read_text())

    pipe = load_pixart_pipeline(with_text_encoder=False).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    emb = load_prompt_embeddings(args.embeddings)
    prompts = list(emb)[: args.limit] if args.limit else list(emb)
    nblocks = len(pipe.transformer.transformer_blocks)

    incep = InceptionV3([3]).cuda().eval()
    perceptual = lpips_pkg.LPIPS(net="alex").cuda().eval()
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").cuda().eval()
    clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def clip_score(img, text):
        b = clip_proc(text=[text], images=img, return_tensors="pt",
                      padding=True, truncation=True).to("cuda")
        with torch.no_grad():
            o = clip(**b)
        i = o.image_embeds / o.image_embeds.norm(dim=-1, keepdim=True)
        t = o.text_embeds / o.text_embeds.norm(dim=-1, keepdim=True)
        return float((i * t).sum())

    banks = {}
    for s in specs:
        ck = s.get("surrogate_checkpoint")
        if ck and ck not in banks:
            from dit_residual_delta.surrogate import load_surrogate_bank
            banks[ck] = load_surrogate_bank(ck)

    feats = {"exact": _Feats(incep), **{s["name"]: _Feats(incep) for s in specs}}
    acc = {s["name"]: {"ssim": [], "lpips": [], "psnr": [], "clip": []} for s in specs}
    acc_exact_clip = []
    common = {"height": int(cfg["height"]), "width": int(cfg["width"]),
              "num_inference_steps": steps, "guidance_scale": float(cfg["guidance_scale"])}
    t0 = time.perf_counter()
    for i, prompt in enumerate(prompts):
        kw = {**common, **emb[prompt]}
        g = torch.Generator("cuda").manual_seed(args.seed + i)
        with torch.inference_mode():
            ex = pipe(generator=g, **kw).images[0]
        feats["exact"].add(ex)
        acc_exact_clip.append(clip_score(ex, prompt))
        ex_t = image_tensor(ex).cuda()
        for s in specs:
            anchors = {t for t in range(steps) if t % int(s.get("cache_interval", 5)) == 0}
            rt = DiTSegmentRuntime(
                pipe.transformer,
                segments=uniform_segments(nblocks, int(s.get("num_segments", 1))),
                anchor_steps=anchors,
                surrogate_bank=banks.get(s.get("surrogate_checkpoint")),
                surrogate_scale=float(s.get("surrogate_scale", 1.0)))
            g = torch.Generator("cuda").manual_seed(args.seed + i)
            with torch.inference_mode(), rt:
                im = pipe(generator=g, **kw).images[0]
            feats[s["name"]].add(im)
            c = image_tensor(im).cuda()
            a = acc[s["name"]]
            a["ssim"].append(skimage_ssim(im, ex, True))
            a["lpips"].append(lpips_alex(perceptual, c, ex_t))
            a["psnr"].append(psnr(c, ex_t))
            a["clip"].append(clip_score(im, prompt))
        if (i + 1) % 50 == 0:
            el = time.perf_counter() - t0
            print(json.dumps({"done": i + 1, "of": len(prompts),
                              "min_elapsed": round(el / 60, 1),
                              "min_left": round(el / (i + 1) * (len(prompts) - i - 1) / 60, 1)}),
                  flush=True)

    # persist raw features so FID/KID can be recomputed or audited without regenerating
    np.savez_compressed(out / "inception_features.npz",
                        **{k: v.raw().astype(np.float32) for k, v in feats.items()})
    raw_e = feats["exact"].raw()
    mu_e, s_e, n_e = feats["exact"].stats()
    res = {"n": n_e, "captions": len(prompts), "exact_clip": float(np.mean(acc_exact_clip)),
           "variants": {}}
    for s in specs:
        mu, sg, n = feats[s["name"]].stats()
        a = acc[s["name"]]
        km, ks = kid(feats[s["name"]].raw(), raw_e)
        res["variants"][s["name"]] = {
            "fid_vs_exact": frechet(mu, sg, mu_e, s_e),
            "kid_vs_exact": km, "kid_subset_std": ks, "n": n,
            "ssim": float(np.mean(a["ssim"])), "lpips": float(np.mean(a["lpips"])),
            "psnr": float(np.mean(a["psnr"])), "clip": float(np.mean(a["clip"])),
            "clip_vs_exact": float(np.mean(a["clip"]) - np.mean(acc_exact_clip))}
    (out / "results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2)[:1400])


if __name__ == "__main__":
    main()
