#!/usr/bin/env python
"""Every number the camera-ready paper quotes, recomputed from runs/cr/.

All image metrics are on test24 x 3 seeds (n=72), paired by case id. Generation is
deterministic per (prompt, seed, variant), so arms from different test runs pair
case-for-case. Writes runs/cr/report.json and prints a readable summary.
"""
from __future__ import annotations

import json, math, statistics as st
from pathlib import Path

R = Path("runs/cr")
KS = [1, 2, 4, 7, 14, 28]


def load_cases(*names):
    """case id -> variant name -> metrics, merged over several evaluate_variants runs."""
    out: dict[str, dict] = {}
    for n in names:
        d = json.loads((R / n / "results.json").read_text())
        for c in d["cases"]:
            out.setdefault(c["case"], {}).update(c["variants"])
    return out


def paired(xs, ys):
    d = [x - y for x, y in zip(xs, ys)]
    mu, sd = st.mean(d), st.stdev(d)
    return {"mean": mu, "t": mu / (sd / math.sqrt(len(d))) if sd else float("inf"),
            "wins": sum(v > 0 for v in d), "n": len(d)}


def col(cases, arm, metric):
    return [cases[c][arm][metric] for c in sorted(cases)]


SS, LP, PS, SP = "ssim_gaussian_vs_exact", "lpips_alex_vs_exact", "psnr_vs_exact", "speedup_vs_exact"


def main():
    rep: dict = {}
    sig = json.loads((R / "sigma.json").read_text())
    rep["sigma"] = {k: v["sigma"] for k, v in sig.items()}
    rep["sigma_at_grid_edge"] = [k for k, v in sig.items() if v["at_grid_edge"]]
    cases = load_cases("test_table1", "test_ctrl1", "test_perk")
    rep["n"] = len(cases)

    # ---- Table 1 ------------------------------------------------------------------
    ir = json.loads((R / "test_ir" / "results.json").read_text())
    ir_cases = {c["case"]: c for c in ir["cases"]}   # own id format; paired within IR only
    t1_arms = ["cache_i5", "taylor1", "taylor2", "blockcache", "m_k1", "traj_k1"]
    t1 = {}
    for a in t1_arms:
        row = {"speedup": st.mean(col(cases, a, SP)), "ssim": st.mean(col(cases, a, SS)),
               "psnr": st.mean(col(cases, a, PS)), "lpips": st.mean(col(cases, a, LP)),
               "ir": ir["aggregate"][a]["reward"]["mean"]}
        if a != "cache_i5":
            row["dssim_vs_cache"] = paired(col(cases, a, SS), col(cases, "cache_i5", SS))
        t1[a] = row
    t1["exact"] = {"ir": ir["aggregate"]["exact_reward"]["mean"]}
    rep["table1"] = t1
    ours = "traj_k1"
    vs = {}
    for a in ["taylor1", "taylor2", "blockcache", "m_k1", "cache_i5"]:
        vs[a] = {"ssim": paired(col(cases, ours, SS), col(cases, a, SS)),
                 "psnr": paired(col(cases, ours, PS), col(cases, a, PS)),
                 "lpips_improvement": paired(col(cases, a, LP), col(cases, ours, LP)),
                 "speed_ratio": t1[ours]["speedup"] / t1[a]["speedup"]}
    rep["traj_vs"] = vs

    def ir_col(arm):
        return [ir_cases[c]["variants"][arm]["reward"] for c in sorted(ir_cases)]
    rep["ir_paired"] = {f"{a}_minus_cache": paired(ir_col(a), ir_col("cache_i5"))
                        for a in t1_arms if a != "cache_i5"}
    rep["ir_paired"].update({f"traj_minus_{a}": paired(ir_col(ours), ir_col(a))
                             for a in ["taylor1", "taylor2", "blockcache", "m_k1"]})

    # ---- Table 2: per granularity ----------------------------------------------------
    t2 = {}
    base = col(cases, "cache_i5", SS)
    for K in KS:
        m, t = f"m_k{K}", f"traj_k{K}"
        t2[K] = {"sigma_resid": sig[m]["sigma"], "sigma_traj": sig[t]["sigma"],
                 "resid_gain": paired(col(cases, m, SS), base),
                 "traj_gain": paired(col(cases, t, SS), base),
                 "diff": paired(col(cases, t, SS), col(cases, m, SS)),
                 "lpips_diff_improvement": paired(col(cases, m, LP), col(cases, t, LP))}
    rep["table2"] = t2
    g_r = [t2[K]["resid_gain"]["mean"] for K in KS]
    g_t = [t2[K]["traj_gain"]["mean"] for K in KS]
    rep["falloff"] = {"resid_K1_over_highK": [g_r[0] / g for g in g_r[3:]],
                      "traj_K1_over_highK": [g_t[0] / g for g in g_t[3:]]}

    # ---- Sec 3.2: matched-data retraining vs original recipe -------------------------
    s32 = {}
    for K in [1, 2, 4, 7]:
        rm = json.loads(Path(f"runs/m_k{K}/train_report.json").read_text())["best"]["rel_mse"]
        rc = json.loads(Path(f"runs/ctrl_k{K}/train_report.json").read_text())["best"]["rel_mse"]
        s32[K] = {"rel_mse_ctrl": rc, "rel_mse_matched": rm, "residual_error_drop": 1 - rm / rc,
                  "ssim_gain": paired(col(cases, f"m_k{K}", SS), col(cases, f"ctrl_k{K}", SS)),
                  "correction_gain_ctrl": paired(col(cases, f"ctrl_k{K}", SS), base)}
    for K in [14, 28]:
        s32[K] = {"rel_mse_matched":
                  json.loads(Path(f"runs/m_k{K}/train_report.json").read_text())["best"]["rel_mse"]}
    rep["sec32"] = s32

    # ---- Sec 3.1 / 3.3: trajectory error and oracle ------------------------------------
    se = json.loads((R / "oracle_test" / "step_error.json").read_text())
    mean_curve = {k: [st.mean(c[i] for c in v) for i in range(len(v[0]))] for k, v in se.items()}
    cache = mean_curve["oracle_b0.000"]
    rep["sec31"] = {"curve": cache, "step1": cache[1], "final": cache[-1],
                    "dips_after": [i for i in range(1, len(cache)) if cache[i] < cache[i - 1]]}
    eqb = json.loads((R / "equivalent_beta.json").read_text())
    rep["sec33"] = {"final_by_beta": {k: v[-1] for k, v in mean_curve.items() if k.startswith("oracle")},
                    "final_real": {k: v[-1] for k, v in mean_curve.items() if "real" in k},
                    "equivalent_beta": eqb["equivalent_beta"]}
    for n, b in eqb["equivalent_beta"].items():
        o = mean_curve[f"oracle_b{b:.3f}"][-1]
        real = mean_curve[f"{n} (real)"][-1]
        rep["sec33"][f"{n}_reduction_ratio_vs_oracle"] = (cache[-1] - real) / (cache[-1] - o)

    # ---- Sec 5.3: frozen non-uniform schedule ------------------------------------------
    rep["frozen"] = {k: v for k, v in json.loads((R / "frozen_test" / "results.json")
                                                  .read_text()).items() if k != "cases"}
    (R / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1)[:20000])


if __name__ == "__main__":
    main()
