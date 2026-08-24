import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.4,
                     "axes.spines.top": False, "axes.spines.right": False})
R = "/workspace/dit-residual-delta/runs"
def load(p): return json.load(open(f"{R}/{p}"))
def g(agg, k, m="ssim_gaussian_vs_exact"):
    v = agg[k][m]; return v["mean"] if isinstance(v, dict) else v

CACHE, TAY, OURS, COMB = "0.45", "#1f77b4", "#d62728", "#2ca02c"

# ---------------- Figure 1: Pareto at both step counts ----------------
a20 = load("baseline_compare/results.json")["aggregate"]
a50 = load("s50_final/results.json")["aggregate"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7))

# 20 steps: plain-cache front from the granularity/scaleup runs
front = [("fixed_i4", "$i$=4"), ("fixed_i5", "$i$=5")]
ax1.plot([g(a20, k, "speedup_vs_exact") for k, _ in front],
         [g(a20, k) for k, _ in front], "o-", color=CACHE, lw=1.3, ms=4.5,
         label="no corrector (uniform caching)")
for k, lab in front:
    ax1.annotate(lab, (g(a20, k, "speedup_vs_exact"), g(a20, k)),
                 textcoords="offset points", xytext=(4, -8), fontsize=6.5, color="0.35")
for keys, c, m, lab in (
        (["taylor1_i4", "taylor1_i5"], TAY, "s", "Taylor forecast, order 1"),
        (["ours_i4_s075", "ours_i5_s050"], OURS, "D", "learned corrector (ours)")):
    ax1.plot([g(a20, k, "speedup_vs_exact") for k in keys], [g(a20, k) for k in keys],
             m + "-", color=c, lw=1.3, ms=4.5, label=lab)
ax1.plot([g(a20, "taylor2_i5", "speedup_vs_exact")], [g(a20, "taylor2_i5")],
         "^", color=TAY, ms=4.5, mfc="white", label="Taylor order 2")
ax1.set_xlabel("wall-clock speedup vs. exact ($\\times$)")
ax1.set_ylabel("SSIM vs. same-seed exact")
ax1.set_title("(a) 20 sampling steps  ($n$=192)", fontsize=8)
ax1.legend(frameon=False, fontsize=6.2, loc="lower left")

order50 = [("fixed_i5", CACHE, "o", "no corrector"),
           ("taylor1_i5", TAY, "s", "Taylor order 1"),
           ("ours50_inv_s075", OURS, "D", "learned corrector (ours)"),
           ("taylor1_plus_ours50_s025", COMB, "*", "Taylor $+$ corrector")]
for k, c, m, lab in order50:
    ax2.plot([g(a50, k, "speedup_vs_exact")], [g(a50, k)], m, color=c,
             ms=8 if m == "*" else 5.5, label=lab)
ax2.set_xlabel("wall-clock speedup vs. exact ($\\times$)")
ax2.set_ylabel("SSIM vs. same-seed exact")
ax2.set_title("(b) 50 sampling steps  ($n$=48)", fontsize=8)
ax2.legend(frameon=False, fontsize=6.2, loc="lower left")
fig.tight_layout()
fig.savefig("fig1_pareto.png", dpi=200, bbox_inches="tight")

# ---------------- Figure 2: order sweep, residual vs image ----------------
diag = load("taylor_diag_i5.json")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
orders = sorted(int(k) for k in diag["orders"])
rel = [diag["orders"][str(o)]["rel_mse"] for o in orders]
ax1.axhline(1.0, color="0.6", lw=0.8, ls=":")
ax1.plot(orders, rel, "o-", color=TAY, lw=1.4, ms=5)
ax1.annotate("verbatim reuse", (2.55, 1.005), fontsize=6.3, color="0.4")
ax1.annotate("worse than\nnot correcting", (1.9, 1.45), fontsize=6.3, color="0.3")
ax1.set_xticks(orders); ax1.set_xlabel("Taylor order")
ax1.set_ylabel("residual rel-MSE  (1 $=$ verbatim reuse)")
ax1.set_title("(a) residual space, 20 steps", fontsize=8)

d20 = {o: g(a20, f"taylor{o}_i5") - g(a20, "fixed_i5") for o in (1, 2)}
a50b = load("baseline_compare_s50/results.json")["aggregate"]
d50 = {o: g(a50b, f"taylor{o}_i5") - g(a50b, "fixed_i5") for o in (1, 2, 3)}
ax2.axhline(0, color="0.6", lw=0.8, ls=":")
ax2.plot(list(d20), list(d20.values()), "s-", color="#8c564b", lw=1.4, ms=5,
         label="20 steps ($n$=192)")
ax2.plot(list(d50), list(d50.values()), "o-", color=TAY, lw=1.4, ms=5,
         label="50 steps ($n$=48)")
ax2.set_xticks([1, 2, 3]); ax2.set_xlabel("Taylor order")
ax2.set_ylabel("$\\Delta$SSIM vs. verbatim reuse")
ax2.set_title("(b) deployed quality", fontsize=8)
ax2.legend(frameon=False, fontsize=6.5, loc="lower left")
fig.tight_layout()
fig.savefig("fig2_order.png", dpi=200, bbox_inches="tight")

# ---------------- Figure 3: granularity -- the mechanism ----------------
fig, ax = plt.subplots(figsize=(3.6, 2.7))
# learned corrector, sigma=1, from the paper's granularity table (n=8 design split)
Kl = [1, 2, 4, 28]; dl = [+0.0516, -0.2352, -0.2509, -0.3116]
ax.plot(Kl, dl, "D--", color=OURS, lw=1.4, ms=5, label="learned corrector, $\\sigma$=1")
tay = [(1, g(a20, "taylor1_i5") - g(a20, "fixed_i5")),
       (28, g(a20, "taylor1_K28_i5") - g(a20, "fixed_i5"))]
ax.plot([k for k, _ in tay], [v for _, v in tay], "s-", color=TAY, lw=1.4, ms=5,
        label="Taylor order 1 (state-independent)")
ax.axhline(0, color="0.6", lw=0.8, ls=":")
ax.set_xscale("log"); ax.set_xticks(Kl); ax.set_xticklabels([str(k) for k in Kl])
ax.set_xlabel("$K$  (injection sites per reuse step)")
ax.set_ylabel("$\\Delta$SSIM vs. verbatim reuse")
ax.set_title("Granularity collapse needs state dependence", fontsize=8)
ax.legend(frameon=False, fontsize=6.3, loc="lower left")
fig.tight_layout()
fig.savefig("fig3_granularity.png", dpi=200, bbox_inches="tight")

# ---------------- Figure 4: per-horizon energy and error ----------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.7))
h = sorted(int(k) for k in diag["target_rms_by_horizon"])
rms = [diag["target_rms_by_horizon"][str(k)] for k in h]
ax1.bar([str(k) for k in h], rms, color=CACHE, width=0.6)
for x, v in zip(range(len(h)), rms):
    ax1.text(x, v + 0.05, f"{v:.2f}", ha="center", fontsize=6.5, color="0.3")
ax1.set_xlabel("$\\tau$  (steps since anchor)")
ax1.set_ylabel("target $\\Delta R$ RMS  (normalised)")
ax1.set_title("(a) target energy grows steeply with $\\tau$", fontsize=8)
ax1.grid(axis="x", visible=False)

tr = json.load(open("/workspace/dit-residual-delta/runs/sm_inv_3037/train_report.json"))
ph_i = tr["best"]["metrics"]["per_horizon"]
ph_s = json.load(open("/workspace/dit-residual-delta/runs/sm_std_3037/train_report.json"))["best"]["metrics"]["per_horizon"]
ax2.plot(h, [ph_s[str(k)] for k in h], "s-", color=TAY, lw=1.4, ms=5, label="standard loss")
ax2.plot(h, [ph_i[str(k)] for k in h], "D-", color=OURS, lw=1.4, ms=5, label="scale-invariant loss")
ax2.set_xticks(h); ax2.set_xlabel("$\\tau$  (steps since anchor)")
ax2.set_ylabel("rel-MSE at this $\\tau$")
ax2.set_title("(b) the corrector is worst where energy is lowest", fontsize=8)
ax2.legend(frameon=False, fontsize=6.5)
fig.tight_layout()
fig.savefig("fig4_energy.png", dpi=200, bbox_inches="tight")
print("figures written")
