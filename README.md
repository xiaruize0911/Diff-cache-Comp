# Residual Accuracy Does Not Predict Deployed Quality in Cached Diffusion Transformers

Research code for the question: **when you add a learned corrector on top of step
caching in a diffusion model, what objective should train and select it?**

The short answer we arrived at is that the natural objective — reconstruction error in
residual space — is unreliable as a proxy for deployed image quality. In one regime it
is worse than unreliable: across injection granularity it ranks two designs in the
*opposite* order. We are deliberate about how far that extends. Our own attempt to
exploit the insight, a per-site scale-invariant training loss, helps on our templated
prompt split and then **fails to replicate on a standard benchmark** — so we report the
null and do not claim the fix. The paper in [`paper/`](paper/) documents all of this,
gives a mechanism, and reports several negative results.

## Layout

| directory | what it is | status |
|---|---|---|
| [`dit-residual-delta/`](dit-residual-delta/) | primary arm: PixArt-Σ-XL-2-512 (0.6B, 28 blocks) | main results |
| [`flux-residual-delta/`](flux-residual-delta/) | secondary arm: FLUX.1-dev (12B, 57 blocks, flow matching) | gates pass; loss fix does **not** replicate |
| [`sd15-residual-delta/`](sd15-residual-delta/) | first arm: SD1.5 U-Net attention modules | closed, negative result |
| [`paper/`](paper/) | LaTeX source + compiled PDF | see status below |

## Headline results (primary arm)

- **Correction works, and it survives a standard benchmark.** On 1989 COCO captions a
  whole-stack corrector halves the distributional divergence from the exact model
  (KID $24.1 \to 12.5 \times 10^{-3}$) for $+5.1$% FLOPs, at unchanged latency relative
  to an equally-sized corrector. On our held-out templated split it gains $+0.034$ to
  $+0.054$ SSIM over plain caching at the same anchor schedule (paired $t = 9.9$ – $19.3$,
  149–182 of 192 images improve).
- **Injection granularity inverts the metrics.** Correcting each of 28 blocks scores
  6× better residual accuracy than correcting the stack once, and destroys the image.
  No scalar gain calibration rescues it (`runs/k28_calibration`).
- **Our proposed loss fix did not survive its own benchmark.** A per-site
  scale-invariant objective, which the mechanism predicts should help, does help on the
  templated split ($+0.0077$ SSIM, $-0.0091$ LPIPS, both significant under
  prompt-subject clustering) but produces **no resolvable effect on COCO**, where four
  fidelity metrics disagree on its sign. We report the null rather than the split that
  favours us. An earlier draft of this paper headlined this as a positive result.
- **On-policy (DAgger) data collection buys nothing** at the image level over an equal
  volume of ordinary data, despite improving rel-MSE by 11.9%.
- **FID is unusable at the sample sizes this setting affords.** Two halves of our own
  exact set score FID $50.7$ against each other — worse than the accelerated variants
  score against the whole set. Every distributional claim here rests on KID, validated
  against a same-distribution baseline where it correctly returns zero.

Three of the five findings are negative, including one that contradicts an earlier draft.

## Honest status

The paper was reviewed by four independent adversarial reviewers during development.
That round scored it **weak reject (4/10)** for a top-tier venue; a reproducibility
audit scored **9/10**, having recomputed ~70 of its numbers from the raw artifacts in
`runs/`. Both scores predate the COCO benchmark and the retraction above, and the
revised paper has not been re-scored.

**We are not a competitive accelerator, and do not claim to be.** A training-free
method, TaylorSeer, reports near-lossless generation at $4.99\times$ compression on
FLUX.1-dev, where our corrected variant at $4.7\times$ reaches SSIM $0.585$ against the
exact output. Our operating points were chosen to make a clean diagnostic comparison
against an equal-cost correction-free baseline, not to win a trade-off.

Known remaining gaps, stated in full in the paper's Limitations rather than papered
over: a narrow prompt distribution; no re-implementation of any *specific* published
method (we do implement the *idea* of timestep-aware adaptive caching as a baseline,
Sec. "An adaptive-schedule baseline"); and no test of correction in the conservative
caching regime — our correctors are trained and evaluated at anchor intervals 4 and 5
only, where absolute fidelity is SSIM $\approx 0.53$ against the exact output.

Our own estimate of the loss effect moved four times as designs got cleaner
(+0.0217 → +0.0115 → +0.0266 → +0.0077), always toward zero, and then to no resolvable
effect on COCO. Every uncontrolled degree of freedom we removed made the effect smaller.
The claim that survived every design is the *ordering* under granularity, not the
magnitude of the loss contrast.

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
python scripts/train_surrogate.py --help        # --normalize-loss is the per-site objective
python scripts/eval_checkpoint.py --help        # rel-MSE *and* per-site median
python scripts/evaluate_variants.py --help      # image-level, same-seed vs exact
```

Protocol we would ask others to keep, learned the hard way: select injection strength on
validation and never on test; replicate the decisive contrast across training seeds and
report the seed spread next to any effect; confirm any effect on a standard benchmark
before claiming it, because ours did not survive one; do not select checkpoints with
residual error; verify the deployed low-precision path reproduces the training path
before trusting a rollout; inspect artefacts at native resolution, never on resampled
thumbnails; and treat residual-space error strictly as a diagnostic.
