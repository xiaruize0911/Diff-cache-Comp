#!/usr/bin/env bash
# Trajectory-objective training for every rung, not just K=1.
#
# The one-interval truncation is used because it won the image-level comparison
# (+0.0551 vs +0.0512 for full horizon at K=1); full horizon was better on the
# latent objective but not on SSIM, which localised the bottleneck to the reward.
#
# Training sigma is each rung's val-selected optimum from the 7-point sweep
# (K=1 0.75, K=2 0.3, K=4 0.3, K=7 0.2). K=14 and K=28 were never swept, so they
# take 0.2 on the trend that the optimum falls with K -- sigma is re-selected on
# val at evaluation time regardless, so this only sets the training operating point.
set -euo pipefail
cd /workspace/dit-residual-delta
for spec in 2:0.3 4:0.3 7:0.2 14:0.2 28:0.2; do
  K=${spec%%:*}; SG=${spec##*:}
  OUT=runs/traj_k$K
  [ -f $OUT/train_report.json ] && { echo "=== K=$K 已完成，跳过 ==="; continue; }
  echo "=== traj K=$K sigma=$SG ==="
  SEGARG="--num-segments $K"
  [ "$K" = "28" ] && SEGARG="--num-segments 28"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 scripts/train_latent_objective.py \
    --config configs/pixart_sigma_512.toml \
    --prompt-file data/prompts/train112.txt --embeddings data/embeddings/train112.pt \
    --init-checkpoint runs/m_k$K/best.pt --output-dir $OUT \
    --cache-interval 5 $SEGARG --surrogate-scale $SG \
    --objective latent --truncate-intervals 1 --checkpoint-corrector \
    --steps 400 --learning-rate 1e-5 --train-images 48 --val-images 12 --validate-every 25
  echo "=== traj K=$K done ==="
done
echo TRAJ_ALL_K_DONE
