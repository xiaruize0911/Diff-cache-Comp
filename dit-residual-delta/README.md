# DiT Block Residual-Delta Reuse

Second backbone arm of the residual-delta reuse study. The SD1.5 U-Net arm
(`/workspace/sd15-residual-delta`) is closed with a negative, structural result;
its own report recommends re-testing on an attention-dominant DiT backbone,
because on SD1.5 attention is only 41% of end-to-end time and the oracle
correction ceiling was 13%.

Backbone here: **PixArt-Sigma-XL-2-512-MS** (0.6B DiT, 28 blocks, 20 steps,
ungated). FLUX.1-dev and FLUX.1-schnell are both gated on Hugging Face; the
harness is backbone-generic, so swapping in FLUX later only needs a loader.

The cached unit is one `BasicTransformerBlock`, the DiT analogue of SD1.5's
`Transformer2DModel`. Input and output share a shape, so at step `t`:

```text
R_t = Block(h_t) - h_t
h_out_hat = h_t + R_anchor + delta_R_hat
```

`delta_R_hat = 0` is plain fixed reuse. `oracle_refresh_keys` recomputes the true
residual at selected `(step, block)` slots -- the upper bound any corrector can
reach there.

## Pre-registered gates

Run in order; stop at the first failure. See
[`docs/dit_pivot_plan.md`](docs/dit_pivot_plan.md) for why this order differs
from the older FLUX plan (which gated on residual MSE -- a metric SD1.5 showed
to be decoupled from image quality).

| Gate | Question | Pass condition | Result (2026-08-21) |
|---|---|---|---|
| G0 | What fraction of wall clock is the block stack? | blocks >= 75% of end-to-end (SD1.5: 41%) | **PASS** -- 91.6%, all-cache ceiling 11.96x |
| G1 | How does correction-free caching trade speed for quality? | records the baseline Pareto front | done -- 1.747x @ SSIM 0.676 is the reference point |
| G2 | Oracle ceiling on the top 25% of slots | >= 40% of the SSIM gap recovered (SD1.5: 13%) | **FAIL** -- 5.4% (design), 2.1% (holdout) |
| G2b | All slots, partial accuracy `alpha` | -- | alpha=0.90 recovers 50.6%; at interval 5 it beats the correction-free reference |
| G3 | Surrogate feasibility | see below -- the residual-space bar was the wrong bar | **done**, first positive result |

Full numbers: [`docs/dit_gate_report_2026-08-21.md`](docs/dit_gate_report_2026-08-21.md) (G0-G2)
and [`docs/corrector_training_report_2026-08-22.md`](docs/corrector_training_report_2026-08-22.md) (G3,
training and deploying C).

**G3 headline.** C is learnable here -- per-block val `rel_mse` 0.0537, versus 0.846 for the SD1.5
arm's surrogate. But deploying it per block destroys the image (SSIM 0.678 -> 0.075), because 28
injection points per step turn the cache's *additive* error accumulation into a *multiplicative*
one. Coarsening to one cached segment per step (K=1) is the only granularity that helps, and it
does: **+0.0176 SSIM over the correction-free Pareto frontier at 3.40x**, on 24 unseen-subject
prompts x 2 seeds (paired t = 6.40, 40/48 wins). Across K = 28/4/2/1 the residual accuracy and the
deployed quality move in *opposite* directions -- the most accurate corrector is the worst one to
deploy. A side effect worth taking for free: caching the 28 blocks as one segment is provably
identical in output to caching them individually (unit-tested) but 12% faster (3.73x vs 3.28x at
interval 5).

The short version: correcting a *subset* of cache slots is useless here -- the loss
is spread evenly across all 280 slots, and six blocks are actively made worse by
being corrected alone.

**Round 2 (2026-08-22, [`dagger_and_loss_report_2026-08-22.md`](docs/dagger_and_loss_report_2026-08-22.md)).**
DAgger was the report's top recommendation and it does not work here: the distribution
gap it targets is only 4.2%, and at equal data volume plain base data deploys *better*
than the base+DAgger union (+0.0363 vs +0.0305 SSIM over the frontier at i5). What did
work was the training objective. The loss divided the target by a single global scale,
so gradients were magnitude-weighted and horizon-4 slots (3.3x the rms of horizon 1)
carried ~11x the energy; making it per-slot scale-invariant (`--normalize-loss`) takes
the equal-FLOPs gain from **+0.0225 to +0.0522 +-0.0025 SSIM** (3 seeds, 3.97x FLOPs,
injection scale selected on val24) at **identical architecture and identical wall clock**.
The corrector is data-limited but less so than the objective was: fixing the loss is
worth more than doubling the data, and halves what doubling the data buys.

`rel_mse` is not merely a weak proxy -- on that comparison (same data, same architecture,
same FLOPs, only the loss differs) it prefers the *worse* model on all 9 seed pairs.
Use the per-slot relative-error median instead, and only to compare models trained on the
same data; see the report for where it fails.

**G3 result (2026-08-22, [`docs/corrector_report_2026-08-22.md`](docs/corrector_report_2026-08-22.md)).**
A corrector was trained. On this backbone the residual is easy: a per-block C
reaches relative residual MSE **0.0537** (SD1.5's trained surrogate: 0.846). But
deploying it destroys the image, and the reason is structural, not a bug --
verified by an open-loop on-policy probe showing C is accurate (0.066) on exactly
the trajectory it fails on. With fixed cache the per-block error is
state-independent, so deviations add along depth; a *working* corrector restores
state dependence and turns the recursion into `delta_{b+1} = (I + J_b) delta_b + eps_b`,
which multiplies over 28 blocks. Sweeping the injection granularity K makes the
tradeoff explicit and inverted:

| K (segments) | blocks/segment | val rel_mse | deployed vs fixed_i5 |
|---:|---:|---:|---:|
| 28 | 1 | **0.0537** | **-0.39 SSIM** (0/48 wins) |
| 4 | 7 | 0.2021 | -0.017 |
| 2 | 14 | 0.2724 | +0.012 (n.s.) |
| 1 | 28 | 0.3166 | **+0.038** (t=6.4, 40/48) |

So the residual-MSE gate is not the right gate: the least accurate corrector is
the only one that helps. Caching the whole stack as one unit with one correction
per step beats the correction-free frontier by **+0.02 to +0.03 SSIM and
-0.03 to -0.05 LPIPS at matched speed** on held-out prompts, in the 3.0-3.3x
region, at a cost of 25 ms per image.

## Commands

```bash
pip install -e .
python -m pytest tests -q                       # runtime semantics, no GPU needed

python scripts/profile_model.py --steps 20                                  # G0
python scripts/smoke_test.py                                                # sanity
python scripts/evaluate_variants.py --config configs/pixart_sigma_512.toml \
  --prompt-file data/prompts/design4.txt --seeds 8201 8202 \
  --variants-file configs/cache_grid.json --output-dir runs/cache_grid       # G1
python scripts/make_oracle_variants.py --mode per-block --anchor-steps ... \
  --output configs/oracle_per_block.json                                     # G2
```

## Rules carried over from the SD1.5 arm

- Speed numbers come from measured wall clock, never from FLOP counts, and never
  from eager-mode surrogate timing (kernel-launch overhead dominated there and
  produced a 0.887x artifact that fusion later corrected to 1.240x).
- Residual-space accuracy is a diagnostic, never a selection metric.
- Every variant is compared against the same-seed exact image, and against the
  correction-free cache baseline -- not only against exact.
- Artifacts live under `/workspace` (the previous pod's `/dev/shm` features and
  `/root` checkpoints were lost on restart).
