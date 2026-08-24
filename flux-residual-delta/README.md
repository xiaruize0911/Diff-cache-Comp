# FLUX Residual-Delta Surrogate

Single-A40 research scaffold for replacing a contiguous segment of
FLUX.1-dev single-stream transformer blocks with a current-conditioned
residual-delta surrogate.

The first implementation targets a segment inside the 38 single-stream blocks:

```text
19 dual-stream blocks -> real single blocks [0:start)
                      -> surrogate for [start:start+length)
                      -> real single blocks [start+length:38)
                      -> output projection
```

For an anchor denoising step `a` and a target step `t`, the teacher records
segment input `h`, segment residual `R = G(h) - h`, and trains:

```text
C(delta_h, R_anchor, timestep, horizon, segment_id) -> delta_R
delta_h = h_t - h_a
delta_R = R_t - R_a
```

## Status

- Phase 0: environment check and model-access check
- Phase 1: feature capture, sharded pair dataset, surrogate training, offline metrics
- Phase 2: rollout integration and image-level evaluation (next implementation gate)
- Phase 3: quantization after the BF16 gate passes

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python scripts/check_env.py --config configs/pilot.toml
python scripts/smoke_test.py
pytest -q
```

Model weights are gated. Set `HF_TOKEN` or log in with the Hugging Face CLI
before running capture:

```bash
huggingface-cli login
python scripts/collect_features.py --config configs/pilot.toml --prompts data/prompts/pilot.txt
python scripts/train_surrogate.py --config configs/pilot.toml
python scripts/evaluate_offline.py --config configs/pilot.toml --checkpoint runs/pilot/best.pt
```

Image-level exact, fixed-cache, and learned-surrogate runs share one entrypoint:

```bash
python scripts/generate_runtime.py --config configs/pilot.toml --mode exact \
  --prompt "a red fox in snow"
python scripts/generate_runtime.py --config configs/pilot.toml --mode fixed \
  --cache-interval 4 --prompt "a red fox in snow"
python scripts/generate_runtime.py --config configs/pilot.toml --mode learned \
  --cache-interval 4 --checkpoint runs/pilot/best.pt --prompt "a red fox in snow"
```

The default pilot deliberately records only four anchor/target pairs per
prompt. Recording all 30 steps for every prompt is unnecessary and creates a
large dataset.

The dependency versions are pinned to the server's CUDA 12.4 / PyTorch 2.4.1
runtime. Newer Transformers 5.x and Diffusers 0.37.x require a newer PyTorch
custom-op API and are intentionally not used.

## Reproducibility rules

- The FLUX backbone is frozen and runs under inference mode.
- Text embeddings are cached per prompt during capture.
- Features are stored as BF16 CPU tensors in bounded shards.
- Train/validation/test prompt splits are disjoint.
- Fake-quantized experiments are never reported as real latency gains.
- Every run writes its resolved config, git revision, package versions, and GPU
  information into the run directory.

See `docs/experiment_plan.md` for the full experiment matrix and gates.
