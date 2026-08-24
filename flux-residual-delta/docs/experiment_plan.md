# Detailed Experiment Plan: Current-Conditioned Residual-Delta Reuse for FLUX

## Research question

At equal measured latency, can a stronger surrogate conditioned on the current
prefix-state change reconstruct the residual of skipped FLUX blocks more
accurately than fixed cache reuse and a lightweight output corrector?

## Model and hardware boundary

- Backbone: FLUX.1-dev, approximately 12B parameters.
- Hardware: one NVIDIA A40 48 GB.
- Resolution: 512 x 512, batch size 1, 30 denoising steps.
- The backbone is always frozen. No gradient graph is retained through FLUX.
- The first implementation skips single-stream blocks `[8, 28)`, 20 of 38
  blocks. The 19 dual-stream blocks and final 10 single-stream blocks remain
  exact.

This cut keeps the cached and predicted tensors in the same 3072-dimensional
state space and gives exact downstream blocks a chance to absorb prediction
error.

## Method

Let `G` denote the selected 20-block segment. At denoising step `s`:

```text
R_s = G(h_s) - h_s
```

For anchor step `a` and later target step `t`:

```text
delta_h = h_t - h_a
delta_R = R_t - R_a
delta_R_hat = C(delta_h, R_a, timestep_t, t-a, segment)
h_after_segment_hat = h_t + R_a + delta_R_hat
```

The default surrogate uses 3072-to-512 input projections, two width-512
conditioned transformer blocks, and a zero-initialized 512-to-3072 output.

## Phase 0: readiness and baseline profiling (3-8 hours)

1. Install pinned dependencies and record the environment.
2. Confirm gated model access and download weights.
3. Run one 512 x 512, 30-step generation with fixed seed.
4. Record peak VRAM, end-to-end latency after one warm-up, and per-block timing.
5. Confirm exactly 19 dual-stream and 38 single-stream blocks.

Exit condition: deterministic generation succeeds without out-of-memory errors.

## Phase 1: pilot feature test (12-24 hours)

Data:

- 256 training prompts, 64 validation prompts, 128 held-out test prompts.
- Four pairs per prompt: `(2,4)`, `(6,10)`, `(12,16)`, `(20,24)`.
- BF16 tensors stored in four-example shards.
- Prompt-disjoint 80/10/10 split within each supplied prompt list.

Models:

1. Fixed residual reuse: predict `delta_R = 0`.
2. Time-only predictor.
3. Same-capacity predictor without `delta_h`.
4. Proposed predictor with `delta_h` and `R_a`.

Training:

- AdamW, learning rate `1e-4`, weight decay `0.01`.
- BF16, micro-batch 1, gradient accumulation 16.
- 5,000 optimizer steps maximum, validation every 250 steps.
- Relative residual MSE is the primary optimization and selection metric.

Gate 1: at horizons of four or more steps, the proposed model must reduce
relative residual MSE by at least 20% compared with fixed reuse. Otherwise stop
before full data collection and inspect representation/cut location.

## Phase 2: image-level BF16 experiment (2-4 additional GPU days)

After Gate 1, integrate the surrogate into the denoising rollout and evaluate:

- Exact FLUX.1-dev.
- Fixed residual reuse.
- First- and second-order residual extrapolation.
- Same-capacity surrogate without `delta_h`.
- Proposed surrogate.
- TeaCache/ERTACache where a reproducible FLUX implementation is available.
- Published OnlineCache numbers are reference-only unless its code is released.

Main data scale:

- 2,000 training, 200 validation, 800 held-out test prompts.
- Seven horizons spanning two to four denoising steps.
- A rollout-generated fine-tuning set is added after teacher-forced training to
  reduce exposure bias.

Metrics:

- Feature: relative MSE, cosine similarity, explained variance, error by horizon.
- Image fidelity to exact same-seed FLUX: LPIPS, SSIM, PSNR, CLIP and DINO.
- Efficiency: wall-clock latency, speedup, peak VRAM, surrogate latency, cache
  memory, and skipped-block fraction.
- Report 95% bootstrap confidence intervals. Repeat 200 prompts with three seeds.

Gate 2: at the same measured speed, improve LPIPS by at least 15% over fixed
reuse. The target operating region is 2.5-3.0x speedup with LPIPS below 0.245;
the research goal is 0.20 or lower.

## Phase 3: quantization (1-2 additional GPU days)

Only after BF16 passes Gate 2:

- BF16, W8A8, W4A8, W4A4, and W4A3 surrogate variants.
- MoDiff-style activation-delta quantization is a secondary extension.
- Fake quantization is used only for fidelity/BOP studies.
- Latency claims require a real kernel on the A40.

Gate 3: LPIPS degradation no larger than 0.01 and either surrogate latency
reduced by 25% or total generation latency reduced by 8%.

## Time budget and stopping policy

- Minimum feasibility answer: 1-2 GPU days.
- BF16 core result: 3-5 GPU days.
- Full quantized, paper-style evaluation: 5-8 continuous GPU days, normally
  7-12 calendar days including setup, failed runs, and analysis.

The project stops at each failed gate. This avoids spending several days on
image evaluation or quantization before the central `delta_h` hypothesis is
supported.
