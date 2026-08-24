import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({"font.size": 7, "font.family": "serif"})
R = "/workspace/dit-residual-delta/runs/qual_grid"
res = json.load(open(f"{R}/results.json"))
cases = {c["case"]: c for c in res["cases"]}
cols = [("exact", "exact (50 steps)"), ("cache", "verbatim reuse"),
        ("taylor1", "Taylor order 1"), ("ours", "corrector (ours)"),
        ("combined", "Taylor $+$ corrector")]
rows = sorted(cases)
fig, axes = plt.subplots(len(rows), len(cols), figsize=(len(cols)*1.75, len(rows)*1.82))
for r, case in enumerate(rows):
    for c, (key, title) in enumerate(cols):
        ax = axes[r][c]
        ax.imshow(Image.open(f"{R}/{case}-{key}.png"))
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_linewidth(0.4); s.set_color("0.7")
        if r == 0: ax.set_title(title, fontsize=7.2)
        if key != "exact":
            v = cases[case]["variants"][key]["ssim_gaussian_vs_exact"]
            ax.set_xlabel(f"SSIM {v:.3f}", fontsize=6.4, labelpad=1.5)
        else:
            ax.set_xlabel("reference", fontsize=6.4, color="0.45", labelpad=1.5)
fig.tight_layout(pad=0.28)
fig.savefig("fig5_qualitative.png", dpi=170, bbox_inches="tight")

# native-resolution crop, one prompt, to avoid judging on thumbnails
case = rows[1]
fig, axes = plt.subplots(1, len(cols), figsize=(len(cols)*2.0, 2.35))
box = (150, 150, 400, 400)
for c, (key, title) in enumerate(cols):
    ax = axes[c]
    ax.imshow(Image.open(f"{R}/{case}-{key}.png").crop(box))
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(0.4); s.set_color("0.7")
    ax.set_title(title, fontsize=7.2)
fig.suptitle("native-resolution 250$\\times$250 crop, same seed", fontsize=7.5, y=1.02)
fig.tight_layout(pad=0.3)
fig.savefig("fig6_crop.png", dpi=190, bbox_inches="tight")
print("qualitative figures written")
print("prompts:")
for r in rows: print("  ", r, "->", cases[r]["prompt"][:70])
