#!/usr/bin/env bash
# Controlled granularity ladder at MATCHED data volume, trained to convergence.
#
# The previous ladder gave K=1 only 1,792 slots for an 11.9M-parameter corrector,
# so it overfit by step 1400 -- that is a data problem, and more steps cannot fix
# it. It also mattered for the claim: if low K is data-starved, part of its
# residual disadvantage is an artefact, and the inversion's residual gap should
# narrow once the starvation is removed.
#
# Seed multiplier 28/K makes the slot count identical across every rung:
#   slots = 112 images * (28/K) seeds * 16 (4 anchors * 4 reuse steps) * K = 50,176
# which is an integer multiplier exactly for K in {1,2,4,7,14,28}. Low K gains 28x
# data; every rung sees the same number of gradient samples and the same prompts.
#
# Seeds are spaced by 1000 because the collector seeds its latent generator with
# (seed + prompt_index): consecutive seeds would hand different prompts the same
# latent. Token subsampling is seeded separately via --slot-seed (banks used to be
# irreproducible; see the comment in collect_features.py).
#
# Budget: doubled until the val rel-MSE actually regresses, no threshold. With 28x
# the data the optimum moves out a long way, so K=1 is run first on its own to
# calibrate the scale before committing the other five rungs.
set -euo pipefail
cd /workspace/dit-residual-delta
BANKS=/workspace/scratch_banks
CAP=2048000
COMMON="--width 512 --depth 4 --mix-tokens --linear-rank 256 \
  --batch-slots 48 --learning-rate 3e-4 --weight-decay 0.01 --warmup 200 --seed 2027"
KS="${*:-1 2 4 7 14 28}"

for K in $KS; do
  NSEED=$(( 28 / K ))
  SEEDS=$(python3 -c "print(' '.join(str(4001+1000*j) for j in range($NSEED)))")
  echo "=== matched K=$K : $NSEED seeds, expect 50176 train slots ==="
  for SPEC in train112:112:train val24:24:val; do
    NAME=${SPEC%%:*}; REST=${SPEC#*:}; LIM=${REST%%:*}; TAG=${REST#*:}
    # the seed loop runs INSIDE a shard (a shard covers 8 prompts at every seed),
    # so the shard count does not scale with NSEED -- each shard just gets NSEED
    # times bigger. Getting this wrong only cost a needless recollect on resume.
    NSHARD=$(python3 -c "print(-(-$LIM // 8))")
    # `|| true` matters: this is an ASSIGNMENT, not a condition, so under
    # `set -e -o pipefail` a failing glob (bank not collected yet) would kill the
    # script outright -- which it silently did twice before this was caught.
    HAVE=$(ls "$BANKS/m$K/$TAG"/shard_*.pt 2>/dev/null | wc -l || true)
    if [ "$HAVE" != "$NSHARD" ]; then
      rm -rf "$BANKS/m$K/$TAG"
      python3 scripts/collect_features.py --config configs/pixart_sigma_512.toml \
        --prompt-file "data/prompts/$NAME.txt" --embeddings "data/embeddings/$NAME.pt" \
        --anchor-steps 0 5 10 15 --num-segments "$K" --limit "$LIM" \
        --seeds $SEEDS --slot-seed 7717 --output-dir "$BANKS/m$K/$TAG"
    fi
  done

  B=16000; PREV=""
  while :; do
    OUT="runs/m_k${K}_b${B}"
    if [ ! -f "$OUT/train_report.json" ]; then
      echo "=== matched K=$K budget $B ==="
      python3 scripts/train_surrogate.py --train-dir "$BANKS/m$K/train" \
        --val-dir "$BANKS/m$K/val" --output-dir "$OUT" \
        --num-blocks "$K" $COMMON --steps "$B" --validate-every $(( B / 40 ))
    fi
    CUR=$(python3 -c "import json;print(json.load(open('$OUT/train_report.json'))['best']['rel_mse'])")
    if [ -n "$PREV" ]; then
      echo "{\"K\": $K, \"budget\": $B, \"rel_mse\": $CUR, \"gain_vs_half_pct\": $(python3 -c "print(round(100*($PREV-$CUR)/$PREV,2))")}"
      [ "$(python3 -c "print(1 if $CUR >= $PREV else 0)")" = "1" ] && break
    fi
    PREV=$CUR
    B=$(( B * 2 ))
    [ "$B" -gt "$CAP" ] && { echo "{\"K\": $K, \"hit_cap\": $CAP}"; break; }
  done

  python3 - "$K" <<'PY'
import json, shutil, sys, glob, re
from pathlib import Path
K = sys.argv[1]
rows = []
for p in sorted(glob.glob(f"runs/m_k{K}_b*/train_report.json"),
                key=lambda x: int(re.search(r"_b(\d+)/", x).group(1))):
    d = json.loads(Path(p).read_text())
    rows.append({"budget": int(re.search(r"_b(\d+)/", p).group(1)),
                 "best_val_rel_mse": d["best"]["rel_mse"], "best_step": d["best"]["step"],
                 "train_slots": d["train_slots"], "epochs": round(d["epochs_at_stop"], 1),
                 "best_is_last_step": d["best"]["step"] == d["history"][-1]["step"]})
for i in range(1, len(rows)):
    prev = rows[i-1]["best_val_rel_mse"]
    rows[i]["gain_vs_half_pct"] = round(100 * (prev - rows[i]["best_val_rel_mse"]) / prev, 2)
win = min(rows, key=lambda r: r["best_val_rel_mse"])
out = Path(f"runs/m_k{K}"); out.mkdir(parents=True, exist_ok=True)
shutil.copy(f"runs/m_k{K}_b{win['budget']}/best.pt", out / "best.pt")
shutil.copy(f"runs/m_k{K}_b{win['budget']}/train_report.json", out / "train_report.json")
last = rows[-1].get("gain_vs_half_pct")
(out / "budget_selection.json").write_text(json.dumps(
    {"K": int(K), "sweep": rows, "selected": win,
     "converged": last is not None and last <= 0.0,
     "final_gain_per_doubling_pct": last, "matched_slots": True}, indent=2))
print(json.dumps({"K": int(K), "slots": win["train_slots"], "budget": win["budget"],
                  "val_rel_mse": win["best_val_rel_mse"],
                  "converged": last is not None and last <= 0.0}))
PY
  rm -rf "$BANKS/m$K"
  echo "=== matched K=$K done ==="
done
echo MATCHED_LADDER_DONE
