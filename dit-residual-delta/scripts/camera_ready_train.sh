#!/usr/bin/env bash
# Camera-ready retraining of every corrector the ICDM paper uses.
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
#
# The pod's network volume holds ~50 GB and a matched bank is 24.5 GB, so matched
# banks are strictly one at a time: `lane main` collects bank K, fits m_k$K with the
# bank on the GPU (as the original ladder did), deletes it, and moves on. `lane side`
# runs the small jobs around it: the one-seed ctrl banks (<= 5 GB) and their
# correctors, and Trajector as each m_k$K lands. Every stage is idempotent.
#
# Usage: camera_ready_train.sh lane main|side
set -euo pipefail
cd /workspace/dit-residual-delta
source /root/venv/bin/activate
BANKS=/workspace/scratch_banks
CFG=configs/pixart_sigma_512.toml
COMMON="--width 512 --depth 4 --mix-tokens --linear-rank 256 \
  --batch-slots 48 --learning-rate 3e-4 --weight-decay 0.01 --warmup 200 --seed 2027"
declare -A BUDGET=([1]=32000 [2]=128000 [4]=512000 [7]=512000 [14]=256000 [28]=256000)
declare -A TSIG=([1]=0.75 [2]=0.3 [4]=0.3 [7]=0.2 [14]=0.2 [28]=0.2)
M_ORDER="7 4 1 2 14 28"
C_ORDER="1 2 4 7"

collect () {  # collect NAME K NSEED   -> $BANKS/NAME/{train,val}/manifest.json
  local NAME=$1 K=$2 NSEED=$3 SEEDS
  SEEDS=$(python -c "print(' '.join(str(4001+1000*j) for j in range($NSEED)))")
  for SPEC in train112:112:train val24:24:val; do
    local PF=${SPEC%%:*} REST=${SPEC#*:}; local LIM=${REST%%:*} TAG=${REST#*:}
    # the manifest is written only after the last shard, so it marks a complete bank
    if [ ! -f "$BANKS/$NAME/$TAG/manifest.json" ]; then
      rm -rf "$BANKS/$NAME/$TAG"
      echo "=== collect $NAME/$TAG ($NSEED seeds) ==="
      python scripts/collect_features.py --config $CFG \
        --prompt-file "data/prompts/$PF.txt" --embeddings "data/embeddings/$PF.pt" \
        --anchor-steps 0 5 10 15 --num-segments "$K" --limit "$LIM" \
        --seeds $SEEDS --slot-seed 7717 --output-dir "$BANKS/$NAME/$TAG"
    fi
  done
  touch "$BANKS/$NAME/READY"
}

wait_for () { until [ -e "$1" ]; do sleep 60; done; }

# A run is finished only when ITS OWN train_report.json records the full budget:
# best.pt appears at the first validation, and the reports cloned from git belong to
# the submitted runs (different output_dir / train_dir), so neither can be trusted.
done_run () {  # done_run RUN_DIR STEPS BANK_SUBSTRING
  python - "$1" "$2" "$3" <<'PY'
import json, sys
from pathlib import Path
run, steps, bank = sys.argv[1], int(sys.argv[2]), sys.argv[3]
try:
    d = json.loads((Path(run) / "train_report.json").read_text())
except Exception:
    sys.exit(1)
a = d.get("args", {})
ok = (a.get("output_dir") == run and a.get("steps") == steps and bank in str(a.get("train_dir", ""))
      and d.get("history") and d["history"][-1]["step"] == steps and (Path(run) / "best.pt").exists())
sys.exit(0 if ok else 1)
PY
}

train_m () {
  local K=$1 B=${BUDGET[$1]}
  done_run runs/m_k$K $B scratch_banks/m$K && return 0
  wait_for "$BANKS/m$K/READY"
  echo "=== m_k$K budget $B ==="
  python scripts/train_surrogate.py --train-dir "$BANKS/m$K/train" --val-dir "$BANKS/m$K/val" \
    --output-dir runs/m_k$K --num-blocks "$K" $COMMON --steps "$B" --validate-every $(( B / 40 ))
}

train_bc () {
  [ -f runs/bc_k1/DONE ] && return 0
  wait_for "$BANKS/m1/READY"
  echo "=== bc_k1 ==="
  python scripts/train_scale_shift.py --train-dir "$BANKS/m1/train" --val-dir "$BANKS/m1/val" \
    --num-blocks 1 --output-dir runs/bc_k1
  touch runs/bc_k1/DONE
}

train_ctrl () {
  local K=$1
  done_run runs/ctrl_k$K 16000 scratch_banks/c$K && return 0
  wait_for "$BANKS/c$K/READY"
  echo "=== ctrl_k$K ==="
  python scripts/train_surrogate.py --train-dir "$BANKS/c$K/train" --val-dir "$BANKS/c$K/val" \
    --output-dir runs/ctrl_k$K --num-blocks "$K" $COMMON --steps 16000 --validate-every 2000
}

train_traj () {
  local K=$1 CK=""
  [ -f runs/traj_k$K/DONE ] && return 0
  until done_run runs/m_k$K ${BUDGET[$K]} scratch_banks/m$K; do sleep 120; done
  # recomputation only (same gradients); without it K=7 needs ~39 GB and cannot share
  # the GPU with the residual lane's 24 GB bank. The submission checkpointed K>=14.
  [ "$K" -ge 4 ] && CK="--checkpoint-corrector"
  echo "=== traj_k$K sigma ${TSIG[$K]} ==="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/train_latent_objective.py --config $CFG \
    --prompt-file data/prompts/train112.txt --embeddings data/embeddings/train112.pt \
    --init-checkpoint runs/m_k$K/best.pt --output-dir runs/traj_k$K \
    --cache-interval 5 --num-segments "$K" --surrogate-scale "${TSIG[$K]}" \
    --objective latent --truncate-intervals 1 $CK \
    --steps 400 --learning-rate 1e-5 --train-images 48 --val-images 12 --validate-every 25
  touch runs/traj_k$K/DONE
}

case "$1 $2" in
  "lane main")
    for K in $M_ORDER; do
      if ! done_run runs/m_k$K ${BUDGET[$K]} scratch_banks/m$K || { [ "$K" = 1 ] && [ ! -f runs/bc_k1/DONE ]; }; then
        collect m$K $K $(( 28 / K ))
        train_m $K
        [ "$K" = 1 ] && train_bc
      fi
      rm -rf "$BANKS/m$K"
    done
    echo LANE_MAIN_DONE ;;
  "lane side")
    for K in $C_ORDER; do
      done_run runs/ctrl_k$K 16000 scratch_banks/c$K || { collect c$K $K 1; train_ctrl $K; }
      rm -rf "$BANKS/c$K"
    done
    for K in $M_ORDER; do train_traj $K; done
    echo LANE_SIDE_DONE ;;
  *) echo "usage: $0 lane main|side" >&2; exit 2 ;;
esac
