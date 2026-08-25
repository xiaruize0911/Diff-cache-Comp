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

a20 = load("baseline_compare/results.json")["aggregate"]
a50 = load("s50_final/results.json")["aggregate"]

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
bc = load("blockcache_test/results.json")["aggregate"]
def gbc(k, m="ssim_gaussian_vs_exact"):
    v = bc[k][m]; return v["mean"] if isinstance(v, dict) else v
bc_base = gbc("A_cache_i5")
BC = "#9467bd"
fig, ax = plt.subplots(figsize=(4.4, 2.9))
# learned corrector at sigma=1, CONTROLLED ladder: identical recipe at every K
# (w512 d4 rank256 mix_tokens, 16k steps, batch 48, seed 2027), n=192
ctrl = load("ctrl_ladder_test/results.json")["aggregate"]
def gc(k, m="ssim_gaussian_vs_exact"):
    v = ctrl[k][m]; return v["mean"] if isinstance(v, dict) else v
ctrl_base = gc("A_cache_i5")
Kl = [1, 2, 4, 28]
dl = [gc(f"ctrl_K{k}_s100") - ctrl_base for k in Kl]
ax.plot(Kl, dl, "D--", color=OURS, lw=1.4, ms=5,
        label="learned corrector, $\\sigma$=1 (matched recipe)")
tay = [(1, g(a20, "taylor1_i5") - g(a20, "fixed_i5")),
       (28, g(a20, "taylor1_K28_i5") - g(a20, "fixed_i5"))]
ax.plot([k for k, _ in tay], [v for _, v in tay], "s-", color=TAY, lw=1.4, ms=5,
        label="Taylor order 1 (state-independent)")
ax.plot([1, 28], [gbc("D_bc_K1_s100") - bc_base, gbc("D_bc_K28_s100") - bc_base],
        "v-", color=BC, lw=1.4, ms=5.5,
        label="Block Caching scale-shift (state-independent)")

ax.axhline(0, color="0.6", lw=0.8, ls=":")
ax.set_xscale("log"); ax.set_xticks(Kl); ax.set_xticklabels([str(k) for k in Kl])
ax.set_xlabel("$K$  (injection sites per reuse step)")
ax.set_ylabel("$\\Delta$SSIM vs. verbatim reuse")
ax.set_title("Granularity collapse needs state dependence", fontsize=8)
ax.legend(frameon=False, fontsize=5.9, loc="lower left")
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
