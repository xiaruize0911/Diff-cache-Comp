#!/usr/bin/env bash
# Camera-ready evaluation. Selection and reporting are on disjoint prompts:
#   select: val24  x seeds 5101 7717 9401, one sigma grid for every corrector
#   report: test24 x seeds 5101 7717 9401, run once at the selected sigma
# Wall-clock speedups come only from the test run of the Table 1 arms, which runs
# alone on the GPU. Idempotent: finished stages are skipped.
set -euo pipefail
cd /workspace/dit-residual-delta
source /root/venv/bin/activate
CFG=configs/pixart_sigma_512.toml
SEEDS="5101 7717 9401"
R=runs/cr
mkdir -p $R configs/cr
ev () {  # ev OUT PROMPTS DTYPE VARIANTS [extra]
  local OUT=$1 P=$2 DT=$3 V=$4; shift 4
  [ -f "$OUT/results.json" ] && return 0
  python scripts/evaluate_variants.py --config $CFG --prompt-file data/prompts/$P.txt \
    --embeddings data/embeddings/$P.pt --seeds $SEEDS --variants-file "$V" \
    --output-dir "$OUT" --surrogate-dtype "$DT" "$@"
}

# ---- 1. sigma sweep on val (no timing is read from these) --------------------------
# grouped by when the correctors become available, so early groups can run while the
# last residual correctors are still training (`camera_ready_eval.sh sweeps-early`)
python scripts/cr_tools.py sweep --with-cache --out configs/cr/sweep_a.json ctrl_k1 m_k1 traj_k1
python scripts/cr_tools.py sweep --out configs/cr/sweep_b.json m_k4 m_k7 traj_k4 traj_k7
python scripts/cr_tools.py sweep --out configs/cr/sweep_c.json ctrl_k2 ctrl_k4 ctrl_k7
python scripts/cr_tools.py sweep --out configs/cr/sweep_d.json m_k2 traj_k2 m_k14 traj_k14 m_k28 traj_k28
ev $R/sweep_a val24 fp16 configs/cr/sweep_a.json > $R/sweep_a.log 2>&1 &
ev $R/sweep_b val24 fp32 configs/cr/sweep_b.json > $R/sweep_b.log 2>&1 &
ev $R/sweep_c val24 fp32 configs/cr/sweep_c.json > $R/sweep_c.log 2>&1 &
if [ "${1:-}" = "sweeps-early" ]; then wait; echo SWEEPS_EARLY_DONE; exit 0; fi
ev $R/sweep_d val24 fp32 configs/cr/sweep_d.json > $R/sweep_d.log 2>&1 &
wait
for g in a b c d; do [ -f $R/sweep_$g/results.json ] || { echo "sweep_$g failed" >&2; exit 1; }; done
python scripts/cr_tools.py select --out $R/sigma.json $R/sweep_{a,b,c,d}/results.json

# ---- 2. test, once, at the selected sigma --------------------------------------------
python scripts/cr_tools.py test --sigma $R/sigma.json --group table1 --out configs/cr/test_table1.json
python scripts/cr_tools.py test --sigma $R/sigma.json --group fp16k1 --out configs/cr/test_ctrl1.json
python scripts/cr_tools.py test --sigma $R/sigma.json --group perk --out configs/cr/test_perk.json
ev $R/test_table1 test24 fp16 configs/cr/test_table1.json --save-images   # alone: timed
ev $R/test_ctrl1 test24 fp16 configs/cr/test_ctrl1.json
ev $R/test_perk test24 fp32 configs/cr/test_perk.json

# ---- 3. ImageReward on the Table 1 arms (test) ----------------------------------------
[ -f $R/test_ir/results.json ] || python scripts/eval_imagereward.py --config $CFG \
  --prompt-file data/prompts/test24.txt --embeddings data/embeddings/test24.pt --seeds $SEEDS \
  --variants-file configs/cr/test_table1.json --output-dir $R/test_ir

# ---- 4. trajectory error and oracle transfer curve (test, 24 prompts, one seed) ------
python - <<'PY'
import json, math
sig = json.load(open("runs/cr/sigma.json"))
rel = {n: json.load(open(f"runs/{n}/train_report.json"))["best"]["rel_mse"] for n in ("m_k1", "ctrl_k1")}
eq = {n: round(1 - math.sqrt(v), 3) for n, v in rel.items()}
blocks = list(range(28))
betas = sorted({0.0, 0.25, 0.6, 0.8, 1.0, *eq.values()})
specs = [{"name": f"oracle_b{b:.3f}", "cache_interval": 5, "oracle_block_ids": blocks,
          "oracle_blend": b} for b in betas]
for n in ("m_k1", "ctrl_k1", "traj_k1"):
    specs.append({"name": f"{n} (real)", "cache_interval": 5, "num_segments": 1,
                  "surrogate_checkpoint": f"runs/{n}/best.pt", "surrogate_scale": sig[n]["sigma"]})
json.dump(specs, open("configs/cr/oracle_test.json", "w"), indent=1)
json.dump({"rel_mse": rel, "equivalent_beta": eq}, open("runs/cr/equivalent_beta.json", "w"), indent=1)
PY
[ -f $R/oracle_test/step_error.json ] || python scripts/plot_step_error.py --config $CFG \
  --prompt-file data/prompts/test24.txt --embeddings data/embeddings/test24.pt \
  --variants-file configs/cr/oracle_test.json --output-dir $R/oracle_test --limit 24 --seed 5101

# ---- 5. non-uniform, corrector-independent schedule (test) ---------------------------
SM=$(python -c "import json;print(json.load(open('$R/sigma.json'))['m_k1']['sigma'])")
ST=$(python -c "import json;print(json.load(open('$R/sigma.json'))['traj_k1']['sigma'])")
[ -f $R/frozen_test/results.json ] || python scripts/eval_frozen_schedule.py --config $CFG \
  --prompt-file data/prompts/test24.txt --embeddings data/embeddings/test24.pt --seeds $SEEDS \
  --threshold 0.33 --arm resid=runs/m_k1/best.pt:$SM --arm traj=runs/traj_k1/best.pt:$ST \
  --output-dir $R/frozen_test
echo EVAL_DONE
