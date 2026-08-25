"""Audit of every training run's validation curve, from runs/*/train_report.json.

Left: all curves, normalised by their own first validation point, so the two
failure modes separate visually -- red runs whose best checkpoint IS the first
validation (overfit before anything was measured), blue runs whose best is the
last step (budget exhausted while still improving).

Right: why the controlled ladder's "matched recipe" was not matched optimisation.
Slots scale with K, so a fixed step budget buys 8x fewer epochs at K=28 than at
K=1, and the best-checkpoint step tracks epochs monotonically.
"""
import json, glob, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
R = "/workspace/dit-residual-delta/runs"
fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.0))
n_over = n_under = 0
for f in sorted(glob.glob(f"{R}/*/train_report.json")):
    d = json.load(open(f)); h = d.get("history") or []
    if not h or len(h) < 2: continue
    if os.path.basename(os.path.dirname(f)).startswith("redo_"): continue
    xs = [x["step"] / h[-1]["step"] for x in h]
    ys = [x["val_rel_mse"] / h[0]["val_rel_mse"] for x in h]
    if d["best"]["step"] == h[0]["step"]:
        col, n_over = "#c0392b", n_over + 1
    elif d["best"]["step"] == h[-1]["step"]:
        col, n_under = "#2471a3", n_under + 1
    else:
        col = "0.75"
    ax[0].plot(xs, ys, color=col, lw=1.0, alpha=0.85 if col != "0.75" else 0.45)
ax[0].axhline(1.0, color="0.3", lw=0.6, ls=":")
ax[0].set_xlabel("fraction of the step budget")
ax[0].set_ylabel("val rel-MSE, relative to its first eval")
ax[0].set_title(f"46 runs. red ({n_over}): best IS the first eval.\n"
                f"blue ({n_under}): best is the last step.", fontsize=7.6)

ladder = [1, 2, 4, 7, 14, 28]
imgs = {1: 112, 2: 112, 4: 112, 7: 64, 14: 64, 28: 32}
ep = [16000 * 48 / (imgs[K] * 16 * K) for K in ladder]
bs = [json.load(open(f"{R}/ctrl_k{K}/train_report.json"))["best"]["step"] for K in ladder]
ax[1].plot(ep, bs, "o-", color="#884ea0")
for K, e, b in zip(ladder, ep, bs):
    ax[1].annotate(f"$K$={K}", (e, b), textcoords="offset points", xytext=(5, 4), fontsize=7)
ax[1].set_xscale("log"); ax[1].set_xlabel("epochs over the feature bank (log)")
ax[1].set_ylabel("best checkpoint step")
ax[1].set_title("the ladder's fixed 16k budget bought 8$\\times$ fewer\n"
                "epochs at $K$=28; best step tracks it monotonically", fontsize=7.6)
ax[1].grid(alpha=0.25, lw=0.4)
fig.tight_layout(pad=0.5)
fig.savefig("fig12_training_audit.png", dpi=200, bbox_inches="tight")
print("overfit:", n_over, "truncated:", n_under)
