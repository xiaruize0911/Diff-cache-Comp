import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.4,
                     "axes.spines.top": False, "axes.spines.right": False})
R = "/workspace/dit-residual-delta/runs"
d = json.load(open(f"{R}/family_grid/results.json"))
a = d["aggregate"]
fl = json.load(open(f"{R}/family_grid_flops.json"))
def g(k, m="ssim_gaussian_vs_exact"):
    v = a[k][m]; return v["mean"] if isinstance(v, dict) else v
def sp(k): return fl[k]["speedup"]

CACHE, TAY, OURS = "0.45", "#1f77b4", "#d62728"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9))

# ---- (a) three families, each sweeping the anchor interval ----
fams = [
    ([f"A_cache_i{i}" for i in (2, 3, 4, 5)], CACHE, "o-", "A  same-model baseline (verbatim reuse)"),
    ([f"B_taylor1_i{i}" for i in (2, 3, 4, 5)], TAY, "s-", "B  TaylorSeer mechanism (order 1)"),
    (["C_ours_i2_s025", "C_ours_i3_s050", "C_ours_i4_s050", "C_ours_i5_s050"], OURS, "D-",
     "C  learned corrector (ours, one model)"),
]
for keys, c, m, lab in fams:
    ax1.plot([sp(k) for k in keys], [g(k) for k in keys], m, color=c, lw=1.3, ms=4.5, label=lab)
for i, k in zip((2, 3, 4, 5), [f"A_cache_i{i}" for i in (2, 3, 4, 5)]):
    ax1.annotate(f"$i$={i}", (sp(k), g(k)), textcoords="offset points",
                 xytext=(3, -9), fontsize=6.3, color="0.35")
ax1.plot([sp("B_taylor2_i4"), sp("B_taylor2_i5")],
         [g("B_taylor2_i4"), g("B_taylor2_i5")], "^:", color=TAY, lw=1.0, ms=4.5,
         mfc="white", label="B  order 2")
ax1.set_xlabel("FLOPs speedup vs. exact ($\\times$)")
ax1.set_ylabel("SSIM vs. same-seed exact")
ax1.set_title("(a) each family sweeps its own cost axis", fontsize=8)
ax1.legend(frameon=False, fontsize=6.0, loc="upper right")

# ---- (b) family C's own configuration axis, at i=5 ----
sig = [(0.25, "C_sigma_i5_s025"), (0.50, "C_ours_i5_s050"),
       (0.75, "C_sigma_i5_s075"), (1.00, "C_sigma_i5_s100")]
ax2.plot([s for s, _ in sig], [g(k) for _, k in sig], "D-", color=OURS, lw=1.3, ms=4.5,
         label="ours, $\\sigma$ sweep at $i$=5")
ax2.axhline(g("A_cache_i5"), color=CACHE, lw=1.2, ls="-", label="A  baseline, $i$=5")
ax2.axhline(g("B_taylor1_i5"), color=TAY, lw=1.2, ls="--", label="B  Taylor order 1, $i$=5")
ax2.plot([0.50], [g("C_ours_i5_s050")], "o", mfc="none", mec="black", ms=11, mew=1.0)
ax2.annotate("selected on val", (0.50, g("C_ours_i5_s050")), textcoords="offset points",
             xytext=(6, 8), fontsize=6.3)
ax2.set_xlabel("injection strength $\\sigma$")
ax2.set_ylabel("SSIM vs. same-seed exact")
ax2.set_title("(b) family C's configuration axis", fontsize=8)
ax2.legend(frameon=False, fontsize=6.2, loc="lower left")
fig.tight_layout()
fig.savefig("fig1_families.png", dpi=200, bbox_inches="tight")
print("fig1_families.png written")
