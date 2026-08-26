#!/usr/bin/env bash
# Redo the granularity ladder with a defensible training regime.
#
# What was wrong with the original ladder (audited in report section 4.12):
#   * one fixed 16k-step budget for every K, while slots scale with K, so
#     effective epochs ran 429 (K=1) down to 54 (K=28). K=1/2/4 overfit before
#     the FIRST validation; K=14/28 were still improving at the cap. The
#     asymmetry lay along the granularity claim's own axis.
#   * validation every 2000 steps, too coarse to locate any optimum.
#   * unmatched image counts across K: 112/112/112/64/64/32.
#
# Regime here: the same 112 train / 24 val images for every K, and the step
# budget is SELECTED PER K on validation over fully-annealed runs.
#
# Why a budget sweep rather than probe-then-retrain: the LR schedule is cosine
# over --steps, so a probe run's optimum is the optimum under a near-constant
# high LR and does not transfer. Measured on K=1 -- the probe put its optimum at
# step 1250 (rel-MSE 0.3840), and retraining with --steps 1250 gave 0.3915,
# worse, because compressing the cosine into 1250 steps undertrains. Budget has
# to be searched over annealed runs.
#
# Both the budget and the checkpoint within a run are chosen on val, never test.
set -euo pipefail
cd /workspace/dit-residual-delta
BANKS=/workspace/scratch_banks
BUDGETS="2000 4000 8000 16000"
COMMON="--width 512 --depth 4 --mix-tokens --linear-rank 256 \
  --batch-slots 48 --learning-rate 3e-4 --weight-decay 0.01 --warmup 200 --seed 2027"

for K in 1 2 4 7 14 28; do
  echo "=== K=$K collect ==="
  for SPEC in train112:112:train val24:24:val; do
    NAME=${SPEC%%:*}; REST=${SPEC#*:}; LIM=${REST%%:*}; TAG=${REST#*:}
    NSHARD=$(( (LIM + 7) / 8 ))          # collector writes 8 prompts per shard
    if [ "$(ls "$BANKS/k$K/$TAG"/shard_*.pt 2>/dev/null | wc -l)" != "$NSHARD" ]; then
      rm -rf "$BANKS/k$K/$TAG"           # never reuse a partial bank
      python3 scripts/collect_features.py --config configs/pixart_sigma_512.toml \
        --prompt-file "data/prompts/$NAME.txt" --embeddings "data/embeddings/$NAME.pt" \
        --anchor-steps 0 5 10 15 --num-segments "$K" --limit "$LIM" \
        --output-dir "$BANKS/k$K/$TAG"
    fi
  done

  for B in $BUDGETS; do
    echo "=== K=$K budget $B ==="
    python3 scripts/train_surrogate.py --train-dir "$BANKS/k$K/train" \
      --val-dir "$BANKS/k$K/val" --output-dir "runs/redo_k${K}_b${B}" \
      --num-blocks "$K" $COMMON --steps "$B" --validate-every $(( B / 40 ))
  done

  python3 - "$K" $BUDGETS <<'PY'
import json, shutil, sys
from pathlib import Path
K, budgets = sys.argv[1], [int(x) for x in sys.argv[2:]]
rows = []
for b in budgets:
    d = json.loads(Path(f"runs/redo_k{K}_b{b}/train_report.json").read_text())
    rows.append({"budget": b, "best_val_rel_mse": d["best"]["rel_mse"],
                 "best_step": d["best"]["step"], "epochs": d["epochs_at_stop"],
                 "best_is_last_step": d["best"]["step"] == d["history"][-1]["step"]})
win = min(rows, key=lambda r: r["best_val_rel_mse"])
out = Path(f"runs/redo_k{K}"); out.mkdir(parents=True, exist_ok=True)
shutil.copy(f"runs/redo_k{K}_b{win['budget']}/best.pt", out / "best.pt")
shutil.copy(f"runs/redo_k{K}_b{win['budget']}/train_report.json", out / "train_report.json")
(out / "budget_selection.json").write_text(json.dumps(
    {"K": int(K), "sweep": rows, "selected": win,
     "budget_at_ladder_edge": win["budget"] in (budgets[0], budgets[-1])}, indent=2))
print(json.dumps({"K": int(K), "selected_budget": win["budget"],
                  "val_rel_mse": win["best_val_rel_mse"],
                  "at_edge": win["budget"] in (budgets[0], budgets[-1])}))
PY

  rm -rf "$BANKS/k$K"        # peak disk stays at one bank (K=28 is 22 GB)
  echo "=== K=$K done ==="
done
echo LADDER_REDO_DONE
