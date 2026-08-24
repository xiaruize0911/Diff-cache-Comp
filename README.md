# Residual-Delta Cache Correction for Diffusion Transformers

Research code for the question: **when you add a learned corrector on top of step
caching in a diffusion model, what objective should train and select it?**

The short answer we arrived at is that the natural objective — reconstruction error in
residual space — does not merely correlate weakly with deployed image quality, it can
rank two correctors in the *opposite* order. The paper in [`paper/`](paper/) documents
that, gives a mechanism, proposes a fix, and reports several negative results.

## Layout

| directory | what it is | status |
|---|---|---|
| [`dit-residual-delta/`](dit-residual-delta/) | primary arm: PixArt-Σ-XL-2-512 (0.6B, 28 blocks) | main results |
| [`flux-residual-delta/`](flux-residual-delta/) | secondary arm: FLUX.1-dev (12B, 57 blocks, flow matching) | gates pass; loss fix does **not** replicate |
| [`sd15-residual-delta/`](sd15-residual-delta/) | first arm: SD1.5 U-Net attention modules | closed, negative result |
| [`paper/`](paper/) | LaTeX source + compiled PDF | see status below |

## Headline results (primary arm)

- **Correction works.** At $3.97\times$ FLOPs reduction a whole-stack corrector gains
  $+0.034$ to $+0.054$ SSIM over plain caching at the same anchor schedule
  (paired $t = 9.9$–$19.3$, 149–182 of 192 held-out images improve), for $+5.1\%$ FLOPs.
- **Injection granularity inverts the metrics.** Correcting each of 28 blocks scores
  6× better residual accuracy than correcting the stack once, and destroys the image.
  No scalar gain calibration rescues it (`runs/k28_calibration`).
- **The training objective matters and residual error gets its sign wrong.** A per-site
  scale-invariant loss deploys better ($+0.008$ SSIM, $-0.009$ LPIPS, both significant
  under prompt-subject clustering) while scoring *worse* on residual rel-MSE in all
  three training seeds.
- **On-policy (DAgger) data collection buys nothing** at the image level over an equal
  volume of ordinary data, despite improving rel-MSE by 11.9%.

## Honest status

The paper was reviewed by four independent adversarial reviewers during development.
The final meta-review scored it **weak reject (4/10)** for a top-tier venue. The
reproducibility audit scored **9/10**, having recomputed ~70 of its numbers from the raw
artifacts in `runs/`. Known remaining gaps are listed in the paper's Limitations and
were not papered over: at time of writing, no FID on a standard benchmark, a narrow
prompt distribution, and no re-implemented published baseline. Work on all three is
in progress on top of this commit.

Our own estimate of the headline effect moved four times as designs got cleaner
(+0.0217 → +0.0115 → +0.0266 → +0.0077), always toward zero. The paper claims the
*ordering*, which survived every design, and not the magnitude, which did not.

## What is not in this repository

Model weights, captured feature banks (`*.pt`, 6–17 GB per split), generated images and
Inception feature caches are all excluded — every one is regenerable from the scripts.
Each arm's README gives the commands; a feature split takes ~6 min on PixArt and ~41 min
on FLUX, and a corrector trains in ~10 min.

## Reproducing

```bash
cd dit-residual-delta && pip install -e .
python -m pytest tests -q                       # 16 tests, no GPU needed
python scripts/collect_features.py --help       # capture on-policy slots
python scripts/train_surrogate.py --help        # --normalize-loss is the fix
python scripts/eval_checkpoint.py --help        # rel-MSE *and* per-site median
python scripts/evaluate_variants.py --help      # image-level, same-seed vs exact
```

Protocol we would ask others to keep, learned the hard way: select injection strength on
validation and never on test; replicate the decisive contrast across training seeds and
report the seed spread next to any effect; do not select checkpoints with residual
error; verify the deployed low-precision path reproduces the training path before
trusting a rollout; inspect artefacts at native resolution, never on resampled
thumbnails; and treat residual-space error strictly as a diagnostic.
