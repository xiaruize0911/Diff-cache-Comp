"""Qualitative panel for the ICDM paper: one row per prompt at native-resolution crop.

Thumbnails hide the texture differences these methods actually differ in -- an
earlier version of this project's reporting was misled that way -- so the panel is a
crop at full resolution rather than downscaled whole images.
"""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({"font.size": 7, "font.family": "serif"})
R = "/workspace/dit-residual-delta/runs/icdm_qual"
meta = json.load(open(f"{R}/meta.json"))
cols = [("reference", "exact, 20 steps"), ("cache", "verbatim cache"),
        ("taylor", "TaylorSeer mech."), ("blockcache", "Block Caching mech."),
        ("ours", "trajectory objective")]
rows = sorted(meta)
BOX = (150, 150, 400, 400)
fig, ax = plt.subplots(len(rows), len(cols), figsize=(len(cols) * 1.32, len(rows) * 1.44))
for r, key in enumerate(rows):
    for c, (name, title) in enumerate(cols):
        A = ax[r][c]
        A.imshow(Image.open(f"{R}/{key}_{name}.png").crop(BOX))
        A.set_xticks([]); A.set_yticks([])
        for sp in A.spines.values():
            sp.set_linewidth(1.0 if name == "ours" else 0.4)
            sp.set_color("#c0392b" if name == "ours" else "0.7")
        if r == 0:
            A.set_title(title, fontsize=6.2)
        if name != "reference":
            A.set_xlabel("%.2f" % meta[key][name]["ssim"], fontsize=5.8, labelpad=1.0)
fig.tight_layout(pad=0.16)
fig.savefig("/workspace/paper/figs/icdm_qual.pdf", bbox_inches="tight")
print("written")
