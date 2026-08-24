# SD1.5 Attention Residual-Delta Reuse

Fast single-A40 validation of current-conditioned residual-delta reuse on
Stable Diffusion 1.5.

Instead of cutting through U-Net skip connections, this implementation targets
every `Transformer2DModel` attention module. Each target has identical input and
output shapes, so at denoising step `t`:

```text
R_t = AttentionModule(h_t) - h_t
delta_h = h_t - h_anchor
delta_R_hat = C(delta_h, R_anchor, timestep, horizon, module_id)
h_out_hat = h_t + R_anchor + delta_R_hat
```

The convolutional U-Net path continues to run at every step and supplies a real
current-state observation to each surrogate. Surrogates are shared by channel
width (320, 640, 1280) and conditioned on module identity.

## Phases

1. Exact/fixed-reuse runtime and latency profile.
2. Teacher-forced feature capture on prompt-disjoint splits.
3. Train BF16 shared surrogates and test the `delta_h` gate.
4. Image-level rollout evaluation and capacity ablation.
5. Quantization only after the BF16 fidelity gate passes.

The FLUX scaffold remains separately available in `/workspace/flux-residual-delta`
(its weights are gated). The follow-up DiT arm this project's report recommends is
running in `/workspace/dit-residual-delta` on PixArt-Sigma; see
`../dit-residual-delta/docs/dit_gate_report_2026-08-21.md`. It reports the DiT
counterparts of the two numbers that closed this arm: cacheable share of wall
clock 41% -> 91.6%, and all-cache speedup ceiling 1.70x -> 11.96x. It also
converts this arm's `rel_mse = 0.846` surrogate into an image-space prediction --
about 1% of the quality gap recovered, which matches what full deployment
measured here.

## Current experiment record

The detailed plan, completed ablations, early-stopping curve, independent
12-case final holdout, artifact layout, and next-stage gates are recorded in
[`docs/experiment_status_2026-08-20.md`](docs/experiment_status_2026-08-20.md).

The independent final holdout is negative: neither the on-policy checkpoint nor
the rollout-trained step-50 checkpoint improves over fixed cache on average.
This is recorded as a failed quality gate, not as a positive acceleration result.

A follow-up multi-seed, tail-weighted rollout run reduces validation variance and
worst-case error, but no checkpoint simultaneously passes the predeclared speed,
MSE, and SSIM gates. Its gate report therefore selects no operating point.

A downstream oracle-refresh pilot finds that module/horizon sensitivity is
strongly trajectory-dependent. A five-pair horizon-aware gate improves the
cost/quality tradeoff (1.367x speedup, 3.14% aggregate MSE improvement), but still
fails the predeclared 5% quality gate and is not promoted to final evaluation.
