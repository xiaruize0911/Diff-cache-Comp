import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
d = json.load(open("/workspace/dit-residual-delta/runs/ik_sweep/results.json"))
a = d["aggregate"]
def g(k, m="ssim_gaussian_vs_exact"):
    v = a[k][m]; return v["mean"] if isinstance(v, dict) else v
IS, KS = [3, 5, 7, 10], [1, 2, 4, 7, 14, 28]
fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
for ax, sig, lab in ((axes[0], 50, "$\\sigma$=0.5"), (axes[1], 20, "$\\sigma$=0.2")):
    M = np.array([[g(f"i{i}_k{K}_s{sig:02d}") - g(f"base_i{i}") for K in KS] for i in IS])
    im = ax.imshow(M, cmap="RdBu", vmin=-0.32, vmax=0.32, aspect="auto")
    ax.set_xticks(range(len(KS))); ax.set_xticklabels(KS)
    ax.set_yticks(range(len(IS)))
    ax.set_yticklabels([f"{i}" + (" ✓" if i == 5 else "") for i in IS])
    ax.set_xlabel("$K$  (injection sites per reuse step)")
    ax.set_ylabel("$i$  (anchor interval)")
    ax.set_title(f"{lab}", fontsize=8.5)
    for r in range(len(IS)):
        for c in range(len(KS)):
            ax.text(c, r, f"{M[r, c]:+.3f}", ha="center", va="center", fontsize=5.9,
                    color="white" if abs(M[r, c]) > 0.17 else "black")
    ax.grid(False)
fig.colorbar(im, ax=axes, fraction=0.022, pad=0.02,
             label="$\\Delta$SSIM vs. verbatim cache at the same $i$")
fig.suptitle("Only $K$=1 ever gains; the $K$>1 collapse deepens with $i$  ($n$=24, val)",
             fontsize=8.5, y=1.04)
fig.savefig("fig10_ik_surface.png", dpi=200, bbox_inches="tight")
print("written")
