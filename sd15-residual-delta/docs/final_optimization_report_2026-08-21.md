# SD1.5 residual-delta / cache strategy final optimization report

Date: 2026-08-21

## 1. Outcome

The preselected final strategy passed all three declared mean-metric gates on a
second, untouched 12-case validation set:

| Strategy | Mean speedup | Gaussian SSIM | AlexNet LPIPS | MSE | Gate result |
|---|---:|---:|---:|---:|---|
| `base_front9_i4` | 1.3073x | 0.7955 | 0.1343 | 0.005137 | fail quality |
| `preselected_extra_10` | **1.2905x** | **0.8601** | **0.0830** | **0.002966** | **pass** |

Declared gates:

- mean end-to-end speedup at least 1.25x;
- mean Gaussian-weighted SSIM greater than 0.850;
- mean AlexNet LPIPS less than 0.125.

The final strategy therefore gives about 29.1% mean end-to-end acceleration
while numerically exceeding the OnlineCache 2.07x FLUX reference precision
values (SSIM 0.850 and LPIPS 0.125) under this project's SD1.5 evaluation
protocol. This is a cross-model numerical comparison, not a controlled claim
that SD1.5 is better than FLUX or that the methods have identical speed/quality
operating points.

## 2. Final strategy

Model and generation settings:

- Stable Diffusion v1.5;
- 512 x 512;
- 30 DDIM denoising steps;
- guidance scale 7.5;
- cache target: all 16 `Transformer2DModel` attention modules.

Exact anchor steps:

```text
0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 28
```

All other steps reuse the cached attention residual. The decisive change over
`front9_interval4` is the additional exact anchor at denoising step 10. This
early reset prevents error from the first long reuse span from propagating into
later spans.

The final selected strategy does not invoke the learned correction model. The
old correction checkpoints were tested with sparse `(module, horizon)` gates,
but their improvement on the new schedule was too small and inconsistent to
justify their latency. The correction infrastructure remains available for
future on-policy retraining on the final schedule.

## 3. Optimization history and decisions

### 3.1 Explicit nonuniform schedules

The runtime was extended with explicit `anchor_steps`. Remote regression tests
passed 10/10 at this point. A 20-schedule, 12-case search showed that the best
initial candidate was:

| Schedule | Speedup | Gaussian SSIM | LPIPS |
|---|---:|---:|---:|
| `front9_interval4` | 1.2973x | 0.8419 | 0.1044 |

This passed speed and LPIPS but missed SSIM by 0.0081.

### 3.2 Existing learned correction

Three rollout checkpoints (steps 25, 50, and 100) and 12 sparse correction
policies were tested. The best two-case result improved MSE by only about 0.24%
and SSIM by about 0.00029. Most policies slightly worsened the output. The old
predictor had been trained for another rollout distribution and was not used in
the final method.

### 3.3 Module-selective cache

The runtime was extended with `cached_module_ids`, allowing selected attention
modules to remain exact. Remote regression tests passed 11/11. A module-mask
search found useful design-set points, including:

| Strategy | Speedup | Gaussian SSIM | LPIPS |
|---|---:|---:|---:|
| exact modules 8/9/14 | 1.2511x | 0.8557 | 0.0955 |
| exact modules 8/9/10 plus module 14 refresh at steps 15/27 | 1.2794x | 0.8526 | 0.0973 |

The latter passed all design gates, but a first untouched validation set reached
only 1.2554x / 0.8400 SSIM / 0.1013 LPIPS. That failed validation set was saved,
then retired from final evaluation.

### 3.4 Single extra anchor

Returning to the original design set, each reuse step was independently
promoted to a full anchor. Step 10 produced a substantially larger and more
consistent gain than module-local refreshes:

| Schedule | Speedup | Gaussian SSIM | LPIPS | MSE |
|---|---:|---:|---:|---:|
| base | 1.3291x | 0.8419 | 0.1044 | 0.004614 |
| extra step 9 | 1.3200x | 0.8811 | 0.0749 | 0.003218 |
| **extra step 10** | **1.3302x** | **0.8842** | **0.0709** | **0.003010** |
| extra step 11 | 1.3264x | 0.8666 | 0.0868 | 0.003744 |

`extra_10` was locked before the second untouched validation set was generated.

## 4. Final validation protocol

The second final validation used four prompts absent from all preceding search,
with seeds 5101, 5102, and 5103, for 12 cases. Each case generated exact,
`base_front9_i4`, and the already selected `preselected_extra_10` output from
the same initial latent.

Metrics:

- end-to-end wall-clock speedup relative to exact SD1.5;
- MSE and PSNR on RGB image tensors;
- project SSIM;
- `skimage` channel-aware SSIM;
- Gaussian-weighted `skimage` SSIM (`sigma=1.5`, population covariance);
- LPIPS 0.1 with AlexNet backbone.

Final aggregate details for `preselected_extra_10`:

| Metric | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|
| Speedup | 1.2905x | 0.0508 | 1.2262x | 1.3873x |
| Gaussian SSIM | 0.8601 | 0.0699 | 0.7632 | 0.9504 |
| LPIPS | 0.0830 | 0.0440 | 0.0189 | 0.1641 |
| MSE | 0.002966 | 0.002062 | 0.000450 | 0.006719 |

The declared gates apply to aggregate means. Per-case minima/maxima show that
hard prompts still exist; this result should not be described as a per-image
guarantee.

## 5. Workspace artifacts

Primary artifacts:

- `configs/final_extra10_schedule.json`
- `configs/final_unseen_prompts_v2.txt`
- `runs/final_unseen_validation_v2_2026-08-21/anchor_schedule_search.json`
- `runs/extra_anchor_refine_holdout4/anchor_schedule_search.json`
- `runs/final_unseen_validation_2026-08-21/module_cache_mask_search.json`
- `src/sd15_residual_delta/runtime.py`
- `scripts/search_anchor_schedules.py`
- `scripts/search_module_cache_masks.py`

Important SHA-256 values:

```text
a949debcb82a06a537011293b6bfb7223584a289b793cff20f639f5138931ea7  final unseen v2 metrics
14b2d5b7943fe6794f7631c5db380b6734bf83e5d6f1a3a496d1ce7bb8e0c890  extra-anchor design metrics
209681e5399bdbab972731b6220253fb9c49b89bdb2d9892bacecfb94aaa3ab4  first failed unseen validation
9cf9e1627acc9be48589a72e13efde02d65790704f1b920c96b276d03523818a  final runtime.py
b3d514cd27b7a8dabdc410718143f3bb7365006e3ff4d772c0281d05c007e6a4  runtime regression tests
88f499d6362ce5bd82f6c0f20d3eef9229af93b08ce8a57bc47ed8d3cbb14374  final schedule used for v2 validation
10fa5b58d40d8ba9a4041d9ec04233747807ccc4e6d22270b71d92343d877177  final v2 prompts
```

All listed artifacts and intermediate search results are stored under
`/workspace/sd15-residual-delta`.

## 6. Recommended next research step

If a learned corrector is revisited, collect on-policy pairs specifically from
the final `extra_10` schedule and train only on the 15 reuse steps that remain.
The acceptance test should require a measurable improvement over 0.8601 SSIM or
0.0830 LPIPS while retaining at least 1.25x speed on another untouched prompt
set. Until that gate is passed, the correction-free `extra_10` policy is the
recommended result.
