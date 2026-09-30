"""Fig. 2 (icdm_main.pdf) and Fig. 3 (icdm_qual.pdf) of the ICDM paper.

Reads the camera-ready rerun in dit-residual-delta/runs/cr/: report.json (test24,
n=72) for the per-granularity gains, the val24 sweeps for the sigma curves, and the
PNGs that test_table1 saved for the crops. Fonts are embedded as TrueType (fonttype
42), which IEEE PDF eXpress accepts; matplotlib's default Type 3 fonts it may not.

    cd paper/figs && python make_icdm_figs.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

RUNS = Path(__file__).resolve().parents[2] / "dit-residual-delta" / "runs" / "cr"
KS = [1, 2, 4, 7, 14, 28]
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 14, "axes.titlesize": 14.5,
    # drawn 7.2 in wide and placed at column width (~0.48x), so ~7 pt in print
})

rep = json.loads((RUNS / "report.json").read_text())
sig = json.loads((RUNS / "sigma.json").read_text())

# ---- Fig. 2 ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
x = range(len(KS))
t2 = rep["table2"]
res = [t2[str(k)]["resid_gain"]["mean"] for k in KS]
tra = [t2[str(k)]["traj_gain"]["mean"] for k in KS]
ax[0].plot(x, res, "o--", color="0.45", ms=4, lw=1.1, label="residual objective")
ax[0].plot(x, tra, "s-", color="#c0392b", ms=4.5, lw=1.5, label="Trajector")
ax[0].axhline(0, color="0.6", lw=0.6)
ax[0].set_xticks(list(x)); ax[0].set_xticklabels([str(k) for k in KS])
ax[0].set_xlabel("injection sites per reuse step $K$")
ax[0].set_ylabel(r"$\Delta$SSIM vs. verbatim cache")
ax[0].set_title(f"test gain ($n{{=}}{rep['n']}$)")
ax[0].legend(fontsize=12, frameon=True); ax[0].grid(alpha=0.25, lw=0.4)

colors = {4: "#1f77b4", 7: "#c0392b", 14: "#8e44ad", 28: "#117a65"}
val_cache = json.loads((RUNS / "sweep_a" / "results.json").read_text())["aggregate"]["cache_i5"][
    "ssim_gaussian_vs_exact"]["mean"]
for k in [4, 7, 14, 28]:
    for obj, ls, mk in (("traj", "-", "o"), ("m", ":", None)):
        curve = sig[f"{obj}_k{k}"]["curve"]
        s = sorted(float(v) for v in curve if 0.05 <= float(v) <= 0.3)
        base_val = sig[f"{obj}_k{k}"]["curve"]
        ax[1].plot(s, [base_val[f"{v:g}"] - val_cache for v in s], ls=ls, marker=mk, ms=3,
                   color=colors[k], lw=1.1 if obj == "traj" else 0.9,
                   label=f"$K{{=}}{k}$" if obj == "traj" else None)
ax[1].set_ylim(-0.08, 0.065)   # the collapse past sigma=0.3 would flatten the peaks
ax[1].set_xlabel(r"injection strength $\sigma$")
ax[1].axhline(0, color="0.6", lw=0.6)
ax[1].set_ylabel(r"validation $\Delta$SSIM")
ax[1].set_title("validation sweep")
ax[1].legend(fontsize=12, ncol=2); ax[1].grid(alpha=0.25, lw=0.4)
fig.tight_layout(pad=0.4)
fig.savefig("icdm_main.pdf", bbox_inches="tight")

# ---- Fig. 3 ---------------------------------------------------------------------------
QUAL = json.loads((RUNS / "qual_cases.json").read_text())["cases"]   # chosen by the rule stored alongside
cols = [("exact", "exact,\n20 steps"), ("cache_i5", "verbatim\ncache"),
        ("taylor1", "TaylorSeer\nmech."), ("blockcache", "Block Caching\nmech."),
        ("traj_k1", "Trajector\n")]
cases = {c["case"]: c for c in json.loads((RUNS / "test_table1" / "results.json").read_text())["cases"]}
fig, ax = plt.subplots(len(QUAL), len(cols), figsize=(7.2, 1.62 * len(QUAL) + 0.3))
for r, q in enumerate(QUAL):
    x0, y0 = q["box"]
    for c, (name, title) in enumerate(cols):
        img = Image.open(RUNS / "test_table1" / f"{q['case']}-{name}.png").crop((x0, y0, x0 + 250, y0 + 250))
        a = ax[r, c]; a.imshow(img); a.set_xticks([]); a.set_yticks([])
        if r == 0:
            a.set_title(title, fontsize=14)
        if name != "exact":
            a.set_xlabel(f"{cases[q['case']]['variants'][name]['ssim_gaussian_vs_exact']:.2f}",
                         fontsize=14, labelpad=1.5)
        for s in a.spines.values():
            s.set_edgecolor("#c0392b" if name == "traj_k1" else "0.6")
            s.set_linewidth(1.2 if name == "traj_k1" else 0.6)
fig.tight_layout(pad=0.2, w_pad=0.3, h_pad=0.5)
fig.savefig("icdm_qual.pdf", bbox_inches="tight", dpi=200)
print("wrote icdm_main.pdf, icdm_qual.pdf")
