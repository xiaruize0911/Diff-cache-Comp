import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.4,
                     "axes.spines.top": False, "axes.spines.right": False})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.55))

# ---- (a) speed/quality front ----
front_x = [1.906, 2.618, 3.485, 4.177]          # FLOPs speedup
front_y = [0.6909, 0.5825, 0.4940, 0.4626]      # SSIM, correction-free, n=192
ax1.plot(front_x, front_y, "o-", color="0.45", lw=1.2, ms=3.5,
         label="no corrector (uniform caching)")
for x, y, lab in zip(front_x, front_y, ["$i$=2", "3", "4", "5"]):
    ax1.annotate(lab, (x, y), textcoords="offset points", xytext=(3, 4),
                 fontsize=6.5, color="0.35")
std = {3.351: (0.5416, 0.0), 3.973: (0.5070, 0.0)}     # n=192, single seed
inv = {3.351: (0.5493, 0.0), 3.973: (0.5186, 0.0)}
for d, c, m, lab in ((std, "#1f77b4", "s", "corrector, standard loss"),
                     (inv, "#d62728", "D", "corrector, scale-invariant loss")):
    xs = sorted(d); ys = [d[x][0] for x in xs]; es = [d[x][1] for x in xs]
    ax1.errorbar(xs, ys, yerr=es, fmt=m + "-", color=c, lw=1.2, ms=4,
                 capsize=2, label=lab)
ax1.set_xlabel("FLOPs speedup vs.\\ exact  ($\\times$)")
ax1.set_ylabel("SSIM vs.\\ same-seed exact")
ax1.legend(frameon=False, fontsize=6.4, loc="upper right")
ax1.set_title("(a) equal-cost quality", fontsize=8)

# ---- (b) the inversion ----
K = [(0.0537, -0.3116, "$K$=28"), (0.2021, -0.2509, "$K$=4"),
     (0.2724, -0.2352, "$K$=2"), (0.3166, +0.0516, "$K$=1")]
ax2.plot([k[0] for k in K], [k[1] for k in K], "^--", color="#2ca02c", lw=1.1,
         ms=4.5, label="granularity, $\\sigma$=1 ($n$=8)")
for x, y, lab in K:
    ax2.annotate(lab, (x, y), textcoords="offset points", xytext=(4, -1), fontsize=6.5)
ax2.axhline(0, color="0.6", lw=0.7, ls=":")
# loss contrast at fixed K=1, 3 seeds each, delta vs correction-free front
ax2.errorbar([0.2929], [0.0363], xerr=[0.0003], fmt="s",
             color="#1f77b4", ms=5, capsize=2, label="standard loss")
ax2.errorbar([0.3069], [0.0478], xerr=[0.0005], fmt="D",
             color="#d62728", ms=5, capsize=2, label="scale-invariant loss")
ax2.annotate("", xy=(0.3069, 0.0478), xytext=(0.2929, 0.0363),
             arrowprops=dict(arrowstyle="->", color="0.3", lw=0.8))
ax2.text(0.298, 0.030, "worse rel-MSE,\nbetter images", fontsize=6.2, color="0.25")
ax2.set_xlabel("residual error  $\\mathrm{rel\\text{-}MSE}$  (lower = more accurate)")
ax2.set_ylabel("$\\Delta$SSIM (deployed)")
ax2.legend(frameon=False, fontsize=6.4, loc="lower left")
ax2.set_title("(b) accuracy vs.\\ deployed quality", fontsize=8)

fig.tight_layout(pad=0.4)
fig.savefig("front.pdf", bbox_inches="tight")
print("wrote figs/front.pdf")
