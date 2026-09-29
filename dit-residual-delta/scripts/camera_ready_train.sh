#!/usr/bin/env bash
# Camera-ready retraining of every corrector the ICDM paper uses, for one K.
#
# The weights behind the submitted paper were lost with the pod that trained them, so
# the rerun retrains from scratch with the recipes recorded in runs/*/train_report.json:
#   m_k$K     residual corrector at matched data (50,176 slots), trained at the budget
#             the original doubling ladder selected (matched_ladder.sh, 0.5% rule).
#             K=2 now uses 128k, the rule's winner, rather than the 64k fallback the
#             submission deployed.
#   ctrl_k$K  (K<=7) the original recipe: one seed, 16k steps -- the "before" arm of
#             the matched-data retraining comparison.
#   bc_k1     Block Caching scale-shift, fitted on the K=1 matched bank.
#   traj_k$K  Trajector: 400 steps of pathwise training initialised from m_k$K.
# Banks are kept in host RAM (--bank-store cpu) so several K can train at once; the
# report records this as bit-identical to the GPU store.
#
# Usage: camera_ready_train.sh K   (idempotent: finished stages are skipped)
set -euo pipefail
K=$1
cd /workspace/dit-residual-delta
source /root/venv/bin/activate
BANKS=/workspace/scratch_banks
CFG=configs/pixart_sigma_512.toml
COMMON="--width 512 --depth 4 --mix-tokens --linear-rank 256 \
  --batch-slots 48 --learning-rate 3e-4 --weight-decay 0.01 --warmup 200 --seed 2027 \
  --bank-store cpu"
declare -A BUDGET=([1]=32000 [2]=128000 [4]=512000 [7]=512000 [14]=256000 [28]=256000)
declare -A TSIG=([1]=0.75 [2]=0.3 [4]=0.3 [7]=0.2 [14]=0.2 [28]=0.2)

collect () {  # collect TAG_DIR NSEED
  local DIR=$1 NSEED=$2 SEEDS
  SEEDS=$(python -c "print(' '.join(str(4001+1000*j) for j in range($NSEED)))")
  for SPEC in train112:112:train val24:24:val; do
    local NAME=${SPEC%%:*} REST=${SPEC#*:}; local LIM=${REST%%:*} TAG=${REST#*:}
    # the manifest is written only after the last shard, so it marks a complete bank
    # (shard counts do not: with >= 8 seeds every prompt fills its own shard)
    if [ ! -f "$DIR/$TAG/manifest.json" ]; then
      rm -rf "$DIR/$TAG"
      python scripts/collect_features.py --config $CFG \
        --prompt-file "data/prompts/$NAME.txt" --embeddings "data/embeddings/$NAME.pt" \
        --anchor-steps 0 5 10 15 --num-segments "$K" --limit "$LIM" \
        --seeds $SEEDS --slot-seed 7717 --output-dir "$DIR/$TAG"
    fi
  done
}

# ---- residual corrector at matched data -------------------------------------------
if [ ! -f runs/m_k$K/best.pt ]; then
  collect "$BANKS/m$K" $(( 28 / K ))
  B=${BUDGET[$K]}
  echo "=== m_k$K budget $B ==="
  python scripts/train_surrogate.py --train-dir "$BANKS/m$K/train" --val-dir "$BANKS/m$K/val" \
    --output-dir runs/m_k$K --num-blocks "$K" $COMMON --steps "$B" --validate-every $(( B / 40 ))
fi
if [ "$K" = 1 ] && [ ! -f runs/bc_k1/best.pt ]; then
  collect "$BANKS/m$K" 28
  echo "=== bc_k1 ==="
  python scripts/train_scale_shift.py --train-dir "$BANKS/m1/train" --val-dir "$BANKS/m1/val" \
    --num-blocks 1 --output-dir runs/bc_k1
fi
rm -rf "$BANKS/m$K"

# ---- original-recipe corrector (before matched-data retraining) --------------------
if [ "$K" -le 7 ] && [ ! -f runs/ctrl_k$K/best.pt ]; then
  collect "$BANKS/c$K" 1
  echo "=== ctrl_k$K ==="
  python scripts/train_surrogate.py --train-dir "$BANKS/c$K/train" --val-dir "$BANKS/c$K/val" \
    --output-dir runs/ctrl_k$K --num-blocks "$K" $COMMON --steps 16000 --validate-every 2000
  rm -rf "$BANKS/c$K"
fi

# ---- Trajector ----------------------------------------------------------------------
if [ ! -f runs/traj_k$K/best.pt ]; then
  CK=""; [ "$K" -ge 14 ] && CK="--checkpoint-corrector"
  echo "=== traj_k$K sigma ${TSIG[$K]} ==="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/train_latent_objective.py --config $CFG \
    --prompt-file data/prompts/train112.txt --embeddings data/embeddings/train112.pt \
    --init-checkpoint runs/m_k$K/best.pt --output-dir runs/traj_k$K \
    --cache-interval 5 --num-segments "$K" --surrogate-scale "${TSIG[$K]}" \
    --objective latent --truncate-intervals 1 $CK \
    --steps 400 --learning-rate 1e-5 --train-images 48 --val-images 12 --validate-every 25
fi
echo "=== K=$K TRAIN_DONE ==="
