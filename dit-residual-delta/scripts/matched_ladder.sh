#!/usr/bin/env bash
# THRESHOLD IS 0.5% BY EXPLICIT USER DECISION. It was found changed to 1.5% mid-run
# by a process outside this session, which made K=7 stop at 256k on a 1.10% gain and
# broke comparability with K=1/2/4. Do not retune it without the user saying so.
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
# Budget: doubled until the gain over the previous doubling falls below 1.5%.
#
# This started as "double until it actually regresses", which turned out to be
# pathological: K=2's gains ran 3.11%, 0.77%, 0.33%, 0.41%, 0.12% -- decaying but
# never negative, so the loop would have run to the 2M-step cap. Measured cost of
# that tail: 16x the compute from 32k to 512k bought 1.63%, against the 26% that
# matching data volume bought and the 5.45x granularity ratio under study. K=2 was
# stopped by hand and a 0.5% threshold adopted at the user's direction.
#
# A process outside this session later raised it to 1.5% and re-selected K=1/2/4 and
# K=7 under that rule (K=1 unchanged, K=2 +0.34%, K=4 +2.02%, K=7 stopped at 256k on
# a 1.10% gain). The reasoning was defensible -- 0.5% is nearly as expensive because
# gains keep landing at 0.6-1.1% -- but 1.5% is precisely the option the user was
# offered and declined, so it has been restored to 0.5% and every rung re-selected.
# Do not retune it without the user saying so.
#
# The SELECTION obeys the same rule, not just the loop: K=2 has measured rungs out
# to 512k, but selecting its best over all of them while later rungs stop at the
# threshold would give K=2 a deeper search than the rest and make the ladder
# uncomparable. Rungs past the stopping point are kept in the record as evidence of
# what a deeper search buys, and excluded from selection.
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
      [ "$(python3 -c "print(1 if ($PREV-$CUR)/$PREV < 0.005 else 0)")" = "1" ] && break
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
# apply the stopping rule to selection too, so every rung searched equally deep
considered = [rows[0]]
for r in rows[1:]:
    considered.append(r)
    if r["gain_vs_half_pct"] < 0.5:
        break
for r in rows:
    r["counted_in_selection"] = r in considered
win = min(considered, key=lambda r: r["best_val_rel_mse"])
out = Path(f"runs/m_k{K}"); out.mkdir(parents=True, exist_ok=True)
shutil.copy(f"runs/m_k{K}_b{win['budget']}/best.pt", out / "best.pt")
shutil.copy(f"runs/m_k{K}_b{win['budget']}/train_report.json", out / "train_report.json")
last = considered[-1].get("gain_vs_half_pct")
(out / "budget_selection.json").write_text(json.dumps(
    {"K": int(K), "sweep": rows, "selected": win,
     "converged": last is not None and last < 0.5,
     "stop_rule": "gain over previous doubling < 0.5%",
     "final_gain_per_doubling_pct": last, "matched_slots": True}, indent=2))
print(json.dumps({"K": int(K), "slots": win["train_slots"], "budget": win["budget"],
                  "val_rel_mse": win["best_val_rel_mse"],
                  "converged": last is not None and last < 0.5}))
PY
  rm -rf "$BANKS/m$K"
  echo "=== matched K=$K done ==="
done
echo MATCHED_LADDER_DONE
