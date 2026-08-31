"""Method schematic for the ICDM Teen paper: caching schedule, residual
regression on the unperturbed path, and Trajector on the cached rollout.
"""
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.size": 7.2,
        "font.family": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

EXACT = "#1f4e79"
CACHE = "#7f8c8d"
OURS = "#c0392b"
ADJ = "#1a5276"
RES = "#6c7a89"
INK = "#1c1c1c"


def box(ax, x, y, w, h, text, fc, ec=None, tc="white", fs=6.6, lw=0.7):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.04",
        facecolor=fc,
        edgecolor=ec or fc,
        linewidth=lw,
        mutation_aspect=0.6,
    )
    ax.add_patch(p)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=tc,
        fontsize=fs,
        linespacing=1.15,
    )


def arrow(ax, x1, y1, x2, y2, color=INK, style="-|>", lw=0.9):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            mutation_scale=8,
            lw=lw,
            color=color,
            shrinkA=0,
            shrinkB=0,
        )
    )


fig = plt.figure(figsize=(7.16, 1.92))
ax = fig.add_axes([0.01, 0.04, 0.98, 0.88])
ax.set_xlim(0, 100)
ax.set_ylim(0, 36)
ax.axis("off")

# --- panel labels ---
ax.text(1.2, 34.6, "(a)  Step cache", fontsize=8, fontweight="bold", color=INK)
ax.text(34.2, 34.6, "(b)  Residual regression  (open-loop)", fontsize=8, fontweight="bold", color=INK)
ax.text(67.8, 34.6, "(c)  Trajector  (pathwise adjoint)", fontsize=8, fontweight="bold", color=INK)

# ========== (a) timeline + one reuse step ==========
# timeline
ax.text(1.4, 30.6, "solver steps  ($i{=}5$)", fontsize=6.4, color=CACHE)
xs = [3.5, 8.5, 13.5, 18.5, 23.5, 28.5]
labs = ["0", "1", "2", "3", "4", "5"]
kinds = ["A", "R", "R", "R", "R", "A"]
for x, lab, k in zip(xs, labs, kinds):
    fc = EXACT if k == "A" else "#d5dbdb"
    tc = "white" if k == "A" else INK
    label = "anchor\n$t{=}%s$" % lab if k == "A" else "reuse\n$t{=}%s$" % lab
    box(ax, x - 1.55, 25.2, 3.1, 3.4, label, fc, tc=tc, fs=5.7)
for a, b in zip(xs[:-1], xs[1:]):
    arrow(ax, a + 1.55, 26.9, b - 1.55, 26.9, color=CACHE, lw=0.7)

ax.text(1.4, 23.4, "one reuse step, $K$ sites", fontsize=6.4, color=CACHE)
# stack of K sites
site_y = 16.4
box(ax, 2.0, site_y, 5.4, 5.6, "$h_t$\nlatent", EXACT, fs=6.4)
arrow(ax, 7.5, site_y + 2.8, 9.3, site_y + 2.8)
box(ax, 9.4, site_y, 8.6, 5.6, r"site $k$" + "\n" + r"$h{+}R_a{+}\sigma\widehat{\Delta R}$", OURS, fs=6.2)
arrow(ax, 18.1, site_y + 2.8, 19.9, site_y + 2.8)
box(ax, 20.0, site_y, 5.0, 5.6, r"$\cdots$", "#d5dbdb", tc=INK, fs=9)
arrow(ax, 25.1, site_y + 2.8, 26.8, site_y + 2.8)
box(ax, 27.0, site_y, 5.4, 5.6, "$x_{t+1}$\nlatent", EXACT, fs=6.4)

ax.text(
    16.5,
    14.4,
    r"verbatim: $\widehat{\Delta R}{=}0$    residual: $C$ on exact path    Trajector: $C$ on cached path",
    ha="center",
    fontsize=5.9,
    color=CACHE,
)

# architecture strip
ax.text(1.4, 12.4, "corrector $C_\\theta$  (shared across sites)", fontsize=6.4, color=CACHE)
box(ax, 2.0, 4.0, 7.2, 7.4, "inputs\n$\\Delta h,\\,R_a,\\,h_a$\nRMS-scaled", "#2e4053", fs=6.0)
arrow(ax, 9.3, 7.7, 10.7, 7.7)
box(ax, 10.8, 4.0, 8.8, 7.4, "AdaLN MLP\n$+$ token mix\n$+$ rank-$r$ path", ADJ, fs=6.0)
arrow(ax, 19.7, 7.7, 21.1, 7.7)
box(ax, 21.2, 4.0, 11.2, 7.4, r"$\widehat{\Delta R}=s_{\Delta R}\,C_\theta(\cdot)$" + "\nzero-init head", OURS, fs=6.0)

# ========== (b) residual regression ==========
# exact path dashed
ax.annotate(
    "",
    xy=(63.5, 27.2),
    xytext=(36.5, 27.2),
    arrowprops=dict(arrowstyle="-|>", color=EXACT, lw=1.15),
)
ax.text(50.0, 28.6, r"exact trajectory $x^\star$  (teacher)", ha="center", fontsize=6.2, color=EXACT)
for x, lab in zip((40.0, 50.0, 60.0), (r"$t$", r"$t{+}1$", r"$t{+}2$")):
    ax.plot(x, 27.2, "o", color=EXACT, ms=4.2, zorder=3)
    ax.text(x, 25.3, lab, ha="center", fontsize=6.0, color=EXACT)

# residual supervision arrows (vertical, no chain)
box(ax, 37.2, 16.6, 8.0, 5.8, r"$C_\theta$ on $h^\star$" + "\n" + r"$\mathcal{L}_{\mathrm{res}}$", RES, fs=6.3)
arrow(ax, 41.2, 22.5, 40.0, 26.3, color=RES, lw=0.85)
ax.text(45.8, 22.8, r"$\Delta R^\star$", fontsize=6.2, color=RES)
ax.text(
    50.0,
    13.4,
    "pointwise  ·  open-loop\nno gradient through the sampler",
    ha="center",
    fontsize=6.0,
    color=CACHE,
)
# small note
box(ax, 47.2, 16.6, 15.4, 5.8, r"$\mathcal{L}_{\mathrm{res}}=\sum_{t,k}\frac{\|\Delta R-\widehat{\Delta R}\|^2}{\|\Delta R\|^2}$", "#eef1f4", ec=RES, tc=INK, fs=6.0)

# ========== (c) Trajector ==========
ax.annotate(
    "",
    xy=(96.6, 27.2),
    xytext=(70.2, 27.2),
    arrowprops=dict(arrowstyle="-|>", color=OURS, lw=1.15),
)
ax.text(83.4, 28.6, r"cached trajectory $x$  (deployed)", ha="center", fontsize=6.2, color=OURS)
for x, lab in zip((73.6, 83.4, 93.2), (r"$t$", r"$t{+}1$", r"$t{+}2$")):
    ax.plot(x, 27.2, "s", color=OURS, ms=4.0, zorder=3)
    ax.text(x, 25.3, lab, ha="center", fontsize=6.0, color=OURS)

# ghost exact
ax.plot([73.6, 93.2], [22.6, 22.6], "--", color=EXACT, lw=0.8, alpha=0.7)
ax.plot(73.6, 22.6, "o", color=EXACT, ms=3.2, alpha=0.7)
ax.plot(83.4, 22.6, "o", color=EXACT, ms=3.2, alpha=0.7)
ax.plot(93.2, 22.6, "o", color=EXACT, ms=3.2, alpha=0.7)
ax.text(96.8, 22.6, r"$x^\star$", va="center", fontsize=6.2, color=EXACT)

# vertical error
arrow(ax, 83.4, 26.5, 83.4, 23.3, color=ADJ, lw=0.9, style="<|-|>")
ax.text(84.6, 24.8, r"$\|x-x^\star\|$", fontsize=6.0, color=ADJ)

box(ax, 71.4, 13.2, 11.4, 6.6, r"$C_\theta$ on cached $h_t$" + "\n$K$ cheap forwards", OURS, fs=6.2)
# adjoint arrows backward
arrow(ax, 83.0, 22.4, 82.4, 19.9, color=ADJ, lw=1.05)
arrow(ax, 82.4, 19.9, 77.2, 16.6, color=ADJ, lw=1.05)
ax.text(86.2, 18.6, "discrete adjoint", fontsize=6.1, color=ADJ, style="italic")

box(ax, 84.0, 13.2, 13.6, 6.6, r"$\mathcal{L}=\sum_{t\in\mathrm{reuse}}\frac{\|x_t-x_t^\star\|^2}{\|x_t^\star\|^2}$", "#fdecea", ec=OURS, tc=INK, fs=6.0)

ax.text(
    84.5,
    10.6,
    "closed-loop  ·  gradients truncated at anchors",
    ha="center",
    fontsize=6.0,
    color=CACHE,
)

fig.savefig("/workspace/paper/figs/icdm_method.pdf", bbox_inches="tight", pad_inches=0.02)
print("written /workspace/paper/figs/icdm_method.pdf")
