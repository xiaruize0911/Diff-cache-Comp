import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.4,
                     "axes.spines.top": False, "axes.spines.right": False})
R = "/workspace/dit-residual-delta/runs"
PS = 54.062 / 20
rows = {}
for f in ("fixed_ref_probe", "fixed_ref_frontier", "fixed_ref_exact"):
    d = json.load(open(f"{R}/{f}/results.json"))
    specs = {s["name"]: s for s in d["variants"]}
    for k, v in d["aggregate"].items():
        sp = specs[k]; st = sp["steps"]
        c = st * PS if sp.get("exact") else len(range(0, st, sp["cache_interval"])) * PS
        if sp.get("surrogate_checkpoint"): c *= 13.606 / 12.944
        s = v["ssim_gaussian_vs_reference"]
        rows[k] = (c, s["mean"] if isinstance(s, dict) else s)

def fam(pred):
    pts = sorted((c, s) for k, (c, s) in rows.items() if pred(k))
    return [p[0] for p in pts], [p[1] for p in pts]

fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.plot(*fam(lambda k: k.startswith("exact")), "o-", color="#2ca02c", lw=1.4, ms=4.5,
        label="exact sampling, fewer steps (no caching)")
ax.plot(*fam(lambda k: k.startswith("cache")), "s--", color="0.45", lw=1.2, ms=4,
        label="verbatim caching, various (steps, $i$)")
ax.plot(*fam(lambda k: k.startswith("ours")), "D-", color="#d62728", lw=1.4, ms=5,
        label="learned corrector (ours)")
ax.plot(*fam(lambda k: k.startswith("taylor")), "v", color="#1f77b4", ms=5,
        label="Taylor forecast")
ax.axvline(20.5, color="0.3", lw=0.8, ls=":")
ax.annotate("crossover $\\approx$ 20 TFLOP\n($\\approx$ 8 exact solver steps)", (20.5, 0.60),
            textcoords="offset points", xytext=(6, 0), fontsize=6.4, color="0.3")
ax.annotate("caching wins", (11, 0.90), fontsize=7, color="0.35")
ax.annotate("fewer exact steps win", (34, 0.66), fontsize=7, color="#2ca02c")
ax.set_xlabel("TFLOP per image (block compute)")
ax.set_ylabel("SSIM vs. a fixed 50-step exact reference")
ax.set_title("Cost/quality against ONE reference, across step counts", fontsize=8.5)
ax.legend(frameon=False, fontsize=6.4, loc="lower right")
fig.tight_layout()
fig.savefig("fig7_frontier.png", dpi=200, bbox_inches="tight")
print("fig7_frontier.png written")
