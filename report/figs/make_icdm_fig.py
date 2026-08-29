"""Single figure for the 5-page ICDM Teen-track paper: gain per rung under the two
objectives, with the sigma sensitivity that sets each rung's operating point.

One figure has to carry both the headline (the trajectory objective wins at every
granularity) and the caveat that makes the numbers reproducible (the optimum sigma
moves with K and the peak narrows), so it is a two-panel figure.
"""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "font.family": "serif"})

rows = json.load(open("/tmp/claude-0/-workspace/75597288-e5d0-45e3-bb4f-df6865532b2c/scratchpad/paper_numbers.json"))
K = [r["K"] for r in rows]
resid = [r["resid"] for r in rows]
traj = [r["traj"] for r in rows]

SIG = [0.10, 0.15, 0.18, 0.22, 0.25]
CURVES = {4: [0.0189, 0.0311, 0.0390, 0.0450, 0.0426],
          7: [0.0245, 0.0450, 0.0515, 0.0429, 0.0158],
          14: [0.0199, 0.0400, 0.0414, 0.0076, -0.0500],
          28: [0.0148, 0.0310, 0.0380, 0.0163, -0.0422]}

fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
x = range(len(K))
ax[0].plot(x, resid, "o--", color="#888888", lw=1.2, ms=4, label="residual objective")
ax[0].plot(x, traj, "s-", color="#c0392b", lw=1.6, ms=4.5, label="trajectory objective")
ax[0].axhline(0, color="0.6", lw=0.6)
ax[0].set_xticks(list(x)); ax[0].set_xticklabels([str(k) for k in K])
ax[0].set_xlabel("injection sites per reuse step $K$")
ax[0].set_ylabel("$\\Delta$SSIM vs. verbatim cache")
ax[0].set_title("gain at every granularity ($n$=72)", fontsize=8.5)
ax[0].legend(fontsize=6.8, loc="upper right"); ax[0].grid(alpha=0.25, lw=0.4)

for k, c in zip((4, 7, 14, 28), ("#2471a3", "#c0392b", "#884ea0", "#117864")):
    ax[1].plot(SIG, CURVES[k], "o-", ms=3.2, lw=1.2, color=c, label="$K$=%d" % k)
ax[1].axhline(0, color="0.6", lw=0.6)
ax[1].set_xlabel("injection strength $\\sigma$")
ax[1].set_ylabel("$\\Delta$SSIM")
ax[1].set_title("the peak moves and narrows with $K$", fontsize=8.5)
ax[1].legend(fontsize=6.8); ax[1].grid(alpha=0.25, lw=0.4)
fig.tight_layout(pad=0.4)
fig.savefig("/workspace/paper/figs/icdm_main.pdf", bbox_inches="tight")
print("written /workspace/paper/figs/icdm_main.pdf")
