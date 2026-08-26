#!/usr/bin/env bash
# Train each high-K rung to convergence instead of to a fixed budget ladder.
#
# Why: in the first pass K=7 selected the sweep's top budget (16000) and was
# still improving, so its number was budget-limited, not converged. The optimum
# step moves out monotonically with K (1400, 2200, 3800, 10000) because slots
# scale with K, so every rung needs its own budget found rather than assumed.
#
# Protocol: double the fully-annealed budget until the val rel-MSE actually
# REGRESSES. No threshold -- the earlier 1%-improvement cutoff would have stopped
# these rungs while they were still gaining 24% per doubling. The criterion is
# clean here because every budget for one K shares one bank, one seed, and
# deterministic training, so a regression is signal rather than resampling noise.
#
# This is expensive by construction. K=28 has 50,176 slots, so at batch 48 one
# epoch is 1,045 steps and the old 16k budget bought 15 epochs against K=1\'s 429.
# Matching K=1\'s epochs needs ~448k steps, and a doubling sequence costs about
# twice its final budget. The whole curve is reported so convergence can be
# checked rather than trusted.
#
# One bank per K, collected once and kept until that K finishes every budget, so
# every budget for a given K sees identical data and the comparison carries no
# collection noise.
set -euo pipefail
cd /workspace/dit-residual-delta
BANKS=/workspace/scratch_banks
CAP=2048000
COMMON="--width 512 --depth 4 --mix-tokens --linear-rank 256 \
  --batch-slots 48 --learning-rate 3e-4 --weight-decay 0.01 --warmup 200 --seed 2027"

for K in 7 14 28; do
  echo "=== converge K=$K collect ==="
  for SPEC in train112:112:train val24:24:val; do
    NAME=${SPEC%%:*}; REST=${SPEC#*:}; LIM=${REST%%:*}; TAG=${REST#*:}
    NSHARD=$(( (LIM + 7) / 8 ))
    if [ "$(ls "$BANKS/k$K/$TAG"/shard_*.pt 2>/dev/null | wc -l)" != "$NSHARD" ]; then
      rm -rf "$BANKS/k$K/$TAG"
      python3 scripts/collect_features.py --config configs/pixart_sigma_512.toml \
        --prompt-file "data/prompts/$NAME.txt" --embeddings "data/embeddings/$NAME.pt" \
        --anchor-steps 0 5 10 15 --num-segments "$K" --limit "$LIM" \
        --output-dir "$BANKS/k$K/$TAG"
    fi
  done

  B=16000; PREV=""
  while :; do
    OUT="runs/conv_k${K}_b${B}"
    if [ ! -f "$OUT/train_report.json" ]; then
      echo "=== converge K=$K budget $B ==="
      python3 scripts/train_surrogate.py --train-dir "$BANKS/k$K/train" \
        --val-dir "$BANKS/k$K/val" --output-dir "$OUT" \
        --num-blocks "$K" $COMMON --steps "$B" --validate-every $(( B / 40 ))
    fi
    CUR=$(python3 -c "import json;print(json.load(open('$OUT/train_report.json'))['best']['rel_mse'])")
    if [ -n "$PREV" ]; then
      GAIN=$(python3 -c "print(round(100*($PREV-$CUR)/$PREV,2))")
      echo "{\"K\": $K, \"budget\": $B, \"rel_mse\": $CUR, \"gain_vs_half_pct\": $GAIN}"
      STOP=$(python3 -c "print(1 if $CUR >= $PREV else 0)")
      [ "$STOP" = "1" ] && break
    fi
    PREV=$CUR
    B=$(( B * 2 ))
    if [ "$B" -gt "$CAP" ]; then echo "{\"K\": $K, \"hit_cap\": $CAP}"; break; fi
  done

  python3 - "$K" <<'PY'
import json, shutil, sys, glob, re
from pathlib import Path
K = sys.argv[1]
rows = []
for p in sorted(glob.glob(f"runs/conv_k{K}_b*/train_report.json"),
                key=lambda x: int(re.search(r"_b(\d+)/", x).group(1))):
    b = int(re.search(r"_b(\d+)/", p).group(1))
    d = json.loads(Path(p).read_text())
    rows.append({"budget": b, "best_val_rel_mse": d["best"]["rel_mse"],
                 "best_step": d["best"]["step"], "epochs": d["epochs_at_stop"],
                 "best_is_last_step": d["best"]["step"] == d["history"][-1]["step"]})
for i in range(1, len(rows)):
    prev = rows[i-1]["best_val_rel_mse"]
    rows[i]["gain_vs_half_pct"] = round(100 * (prev - rows[i]["best_val_rel_mse"]) / prev, 2)
win = min(rows, key=lambda r: r["best_val_rel_mse"])
out = Path(f"runs/conv_k{K}"); out.mkdir(parents=True, exist_ok=True)
shutil.copy(f"runs/conv_k{K}_b{win['budget']}/best.pt", out / "best.pt")
shutil.copy(f"runs/conv_k{K}_b{win['budget']}/train_report.json", out / "train_report.json")
last = rows[-1].get("gain_vs_half_pct")
(out / "budget_selection.json").write_text(json.dumps(
    {"K": int(K), "sweep": rows, "selected": win,
     "converged": last is not None and last <= 0.0,
     "final_gain_per_doubling_pct": last,
     "one_bank_for_all_budgets": True}, indent=2))
print(json.dumps({"K": int(K), "selected_budget": win["budget"],
                  "val_rel_mse": win["best_val_rel_mse"],
                  "converged": last is not None and last <= 0.0,
                  "final_gain_pct": last}))
PY
  rm -rf "$BANKS/k$K"
  echo "=== converge K=$K done ==="
done
echo LADDER_CONVERGE_DONE
