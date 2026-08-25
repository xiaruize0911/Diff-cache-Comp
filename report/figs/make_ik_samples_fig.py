import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({"font.size": 7, "font.family": "serif"})
R = "/workspace/dit-residual-delta/runs/ik_samples"
meta = json.load(open(f"{R}/meta.json"))
cols = [("reference", "exact 20 steps"),
        ("base_i5", "cache $i$=5\nno correction"),
        ("i5_k1_s50", "$i$=5 $K$=1 $\\sigma$=0.5\nbest in-dist."),
        ("i3_k1_s50", "$i$=3 $K$=1 $\\sigma$=0.5"),
        ("base_i10", "cache $i$=10\nno correction"),
        ("i10_k1_s20", "$i$=10 $K$=1 $\\sigma$=0.2"),
        ("i10_k7_s50", "$i$=10 $K$=7 $\\sigma$=0.5\nworst cell"),
        ("i10_k28_s20", "$i$=10 $K$=28 $\\sigma$=0.2")]
rows = sorted(meta)
fig, ax = plt.subplots(len(rows), len(cols), figsize=(len(cols)*1.6, len(rows)*1.8))
for r, key in enumerate(rows):
    for c, (name, title) in enumerate(cols):
        A = ax[r][c]
        A.imshow(Image.open(f"{R}/{key}_{name}.png"))
        A.set_xticks([]); A.set_yticks([])
        for s in A.spines.values(): s.set_linewidth(0.4); s.set_color("0.7")
        if r == 0: A.set_title(title, fontsize=6.4)
        m = meta[key][name]
        lab = ("IR %.2f" % m["reward"]) if name == "reference" else \
              ("SSIM %.2f · IR %.2f" % (m["ssim"], m["reward"]))
        A.set_xlabel(lab, fontsize=5.6, labelpad=1.4,
                     color=("0.45" if name == "reference" else "black"))
fig.suptitle("Samples from the $(i,K)$ surface, same seed 9401", fontsize=8, y=1.005)
fig.tight_layout(pad=0.24)
fig.savefig("fig11_ik_samples.png", dpi=165, bbox_inches="tight")
print("written")
