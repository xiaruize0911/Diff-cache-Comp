import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({"font.size": 7, "font.family": "serif"})
R = "/workspace/dit-residual-delta/runs/samples"
meta = json.load(open(f"{R}/meta.json"))
cols = [("reference", "exact 20 steps\n(full compute)"), ("exact_10", "exact 10 steps"),
        ("exact_5", "exact 5 steps"), ("cache_i5", "verbatim cache $i$=5"),
        ("taylor_i5", "TaylorSeer mech."), ("blockcache_i5", "Block Caching mech."),
        ("ours_i5", "ours $i$=5"), ("ours_i4", "ours $i$=4")]
rows = sorted(meta)
fig, ax = plt.subplots(len(rows), len(cols), figsize=(len(cols)*1.6, len(rows)*1.78))
for r, key in enumerate(rows):
    for c, (name, title) in enumerate(cols):
        A = ax[r][c]
        A.imshow(Image.open(f"{R}/{key}_{name}.png"))
        A.set_xticks([]); A.set_yticks([])
        for s in A.spines.values(): s.set_linewidth(0.4); s.set_color("0.7")
        if r == 0: A.set_title(title, fontsize=6.6)
        m = meta[key][name]
        lab = ("IR %.2f" % m["reward"]) if name == "reference" else \
              ("SSIM %.2f · IR %.2f" % (m["ssim"], m["reward"]))
        A.set_xlabel(lab, fontsize=5.6, labelpad=1.4,
                     color=("0.45" if name == "reference" else "black"))
fig.tight_layout(pad=0.24)
fig.savefig("fig8_samples.png", dpi=165, bbox_inches="tight")

# native-resolution crop for one prompt, since thumbnails hide the artefacts
key = rows[1]; box = (140, 140, 400, 400)
fig, ax = plt.subplots(1, len(cols), figsize=(len(cols)*1.9, 2.3))
for c, (name, title) in enumerate(cols):
    ax[c].imshow(Image.open(f"{R}/{key}_{name}.png").crop(box))
    ax[c].set_xticks([]); ax[c].set_yticks([])
    for s in ax[c].spines.values(): s.set_linewidth(0.4); s.set_color("0.7")
    ax[c].set_title(title, fontsize=6.6)
fig.suptitle("native-resolution 260$\\times$260 crop, same seed", fontsize=7.4, y=1.03)
fig.tight_layout(pad=0.26)
fig.savefig("fig9_samples_crop.png", dpi=185, bbox_inches="tight")
print("written")
