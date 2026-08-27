"""Three-level comparison grid: caching hurts, correction helps, better training
does not show, and simply taking fewer exact steps beats all of it more cheaply.

Columns are ordered by the argument rather than by cost, so the pair that matters
-- old vs matched-data corrector, 25.4% apart in residual accuracy -- sits adjacent.
"""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({"font.size": 7, "font.family": "serif"})
R = "/workspace/dit-residual-delta/runs/compare_samples"
meta = json.load(open(f"{R}/meta.json"))
cols = [("reference", "exact 20 steps\n(reference)"),
        ("cache_i5", "cache $i$=5\nno correction"),
        ("old_k1_s50", "old corrector\n$\\sigma$=0.5, rel-MSE 0.381"),
        ("new_k1_s75", "matched-data corrector\n$\\sigma$=0.75, rel-MSE 0.285"),
        ("exact_10", "exact 10 steps\n(cheaper than all three)")]
rows = sorted(meta)
fig, ax = plt.subplots(len(rows), len(cols), figsize=(len(cols)*1.75, len(rows)*1.85))
for r, key in enumerate(rows):
    for c, (name, title) in enumerate(cols):
        A = ax[r][c]
        A.imshow(Image.open(f"{R}/{key}_{name}.png"))
        A.set_xticks([]); A.set_yticks([])
        for s in A.spines.values():
            s.set_linewidth(1.1 if name in ("old_k1_s50", "new_k1_s75") else 0.4)
            s.set_color("#884ea0" if name in ("old_k1_s50", "new_k1_s75") else "0.7")
        if r == 0: A.set_title(title, fontsize=6.5)
        m = meta[key][name]
        lab = ("IR %.2f" % m["reward"]) if name == "reference" else \
              ("SSIM %.2f · IR %.2f" % (m["ssim"], m["reward"]))
        A.set_xlabel(lab, fontsize=5.8, labelpad=1.4,
                     color=("0.45" if name == "reference" else "black"))
fig.suptitle("The two boxed columns differ by 25.4% in residual accuracy and by "
             "+0.0004 SSIM ($t$=0.0) on the full val split", fontsize=8, y=1.006)
fig.tight_layout(pad=0.26)
fig.savefig("fig13_three_level_compare.png", dpi=170, bbox_inches="tight")

# native-resolution crop of the decisive pair, since thumbnails hide fine structure
key = rows[3]; box = (150, 150, 410, 410)
pair = [("cache_i5", "no correction"), ("old_k1_s50", "old corrector"),
        ("new_k1_s75", "matched-data corrector"), ("reference", "exact 20 steps")]
fig, ax = plt.subplots(1, len(pair), figsize=(len(pair)*2.1, 2.5))
for c, (name, title) in enumerate(pair):
    ax[c].imshow(Image.open(f"{R}/{key}_{name}.png").crop(box))
    ax[c].set_xticks([]); ax[c].set_yticks([])
    for s in ax[c].spines.values(): s.set_linewidth(0.4); s.set_color("0.7")
    ax[c].set_title(title, fontsize=7)
fig.suptitle("native-resolution 260$\\times$260 crop, same seed", fontsize=7.6, y=1.02)
fig.tight_layout(pad=0.26)
fig.savefig("fig14_compare_crop.png", dpi=190, bbox_inches="tight")
print("written")
