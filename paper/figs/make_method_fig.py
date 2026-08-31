"""Two-panel method figure: residual regression vs Trajector.

Only the training-graph contrast. Architecture, the i=5 schedule, and
K-site internals are already in the text and are omitted here.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update(
    {
        "font.size": 8.5,
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

EXACT = "#1f4e79"
OURS = "#c0392b"
MUTED = "#5d6d7e"
INK = "#1b1b1b"


def roundbox(ax, x, y, w, h, text, fc, tc="white", fs=8.2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fc,
            edgecolor=fc,
            linewidth=0,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs)


def arr(ax, p, q, color, lw=1.15):
    ax.add_patch(
        FancyArrowPatch(
            p,
            q,
            arrowstyle="-|>",
            mutation_scale=11,
            lw=lw,
            color=color,
            shrinkA=1,
            shrinkB=1,
        )
    )


fig, axes = plt.subplots(1, 2, figsize=(7.16, 1.72))
fig.subplots_adjust(left=0.03, right=0.99, top=0.86, bottom=0.08, wspace=0.18)

# ---------- (a) residual ----------
ax = axes[0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.2)
ax.axis("off")
ax.set_title("(a)  Residual regression", loc="left", fontsize=9.2, pad=4, color=INK)

xs = [1.6, 5.0, 8.4]
ax.annotate(
    "",
    xy=(9.3, 4.55),
    xytext=(0.7, 4.55),
    arrowprops=dict(arrowstyle="-|>", color=EXACT, lw=1.4),
)
for x in xs:
    ax.plot(x, 4.55, "o", color=EXACT, ms=7, zorder=3)
ax.text(5.0, 5.35, r"exact path $x^\star$", ha="center", fontsize=8.2, color=EXACT)
ax.text(1.6, 5.05, r"$t$", ha="center", fontsize=7.4, color=EXACT)
ax.text(5.0, 5.05, r"$t{+}1$", ha="center", fontsize=7.4, color=EXACT)
ax.text(8.4, 5.05, r"$t{+}2$", ha="center", fontsize=7.4, color=EXACT)

arr(ax, (5.0, 4.35), (5.0, 2.55), MUTED, lw=1.2)
ax.text(5.45, 3.45, r"$\Delta R^\star$", fontsize=8.0, color=MUTED)
roundbox(ax, 3.35, 0.55, 3.3, 1.7, r"$C_\theta$  on $h^\star$", MUTED, fs=8.4)
ax.text(5.0, 0.18, "open-loop  ·  no sampler gradient", ha="center", fontsize=7.0, color=MUTED)

# ---------- (b) Trajector ----------
ax = axes[1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.2)
ax.axis("off")
ax.set_title("(b)  Trajector", loc="left", fontsize=9.2, pad=4, color=INK)

ax.annotate(
    "",
    xy=(9.3, 5.05),
    xytext=(0.7, 5.05),
    arrowprops=dict(arrowstyle="-|>", color=OURS, lw=1.4),
)
ax.plot([0.9, 9.1], [2.95, 2.95], ls="--", color=EXACT, lw=1.15)
for x in xs:
    ax.plot(x, 5.05, "s", color=OURS, ms=6.4, zorder=3)
    ax.plot(x, 2.95, "o", color=EXACT, ms=6.2, zorder=3)

ax.text(5.0, 5.72, r"cached path $x$", ha="center", fontsize=8.2, color=OURS)
ax.text(9.45, 2.95, r"$x^\star$", va="center", fontsize=8.2, color=EXACT)

# error bar at t+1
ax.annotate(
    "",
    xy=(5.0, 3.15),
    xytext=(5.0, 4.85),
    arrowprops=dict(arrowstyle="<|-|>", color="#1a5276", lw=1.15, mutation_scale=9),
)
ax.text(5.45, 4.05, r"$\|x-x^\star\|$", fontsize=8.0, color="#1a5276")

arr(ax, (4.55, 3.55), (3.15, 2.15), "#1a5276", lw=1.25)
roundbox(ax, 1.35, 0.55, 3.5, 1.7, r"$C_\theta$  on $h_t$", OURS, fs=8.4)
ax.text(6.7, 1.4, "adjoint through\nthe cached rollout", ha="center", fontsize=7.2, color="#1a5276")
ax.text(5.0, 0.18, "closed-loop  ·  cut at anchors", ha="center", fontsize=7.0, color=MUTED)

fig.savefig("/workspace/paper/figs/icdm_method.pdf", bbox_inches="tight", pad_inches=0.03)
print("written /workspace/paper/figs/icdm_method.pdf")
