"""Tables 1 and 2 of the ICDM paper, generated from the camera-ready rerun.

Reads dit-residual-delta/runs/cr/report.json and writes ../tab_main.tex and
../tab_perk.tex, which icdm_teen.tex \\input{}s, so no table cell is typed by hand.

    cd paper/figs && python make_tables.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
rep = json.loads((HERE.parents[1] / "dit-residual-delta" / "runs" / "cr" / "report.json").read_text())
t1 = rep["table1"]

rows = [("cache_i5", "Verbatim cache"), ("taylor1", "TaylorSeer 1st"), ("taylor2", "TaylorSeer 2nd"),
        ("blockcache", "Block Caching"), ("m_k1", "Residual obj."), ("traj_k1", r"\textbf{\ourmethod{}}")]
cols = {"speedup": max, "ssim": max, "psnr": max, "lpips": min, "ir": max}
best = {c: f(t1[a][c] for a, _ in rows) for c, f in cols.items()}
best_d = max(t1[a]["dssim_vs_cache"]["mean"] for a, _ in rows if a != "cache_i5")


def cell(a, c, fmt):
    v = t1[a][c]
    s = fmt.format(v)
    return rf"\textbf{{{s}}}" if v == best[c] else s


def sgn(v, d=2):   # used inside math mode
    return f"{v:+.{d}f}"


lines = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r"Method & Sp.\ & SSIM$\uparrow$ & PSNR$\uparrow$ & LPIPS$\downarrow$ & IR$\uparrow$ & $\Delta$SSIM \\",
         r"\midrule",
         rf"Exact, 20 steps & 1.00 & --- & --- & --- & ${sgn(t1['exact']['ir'])}$ & --- \\"]
for a, name in rows:
    ir = sgn(t1[a]["ir"])
    ir = rf"$\mathbf{{{ir}}}$" if t1[a]["ir"] == best["ir"] else f"${ir}$"
    if a == "cache_i5":
        d = "---"
    else:
        dv = t1[a]["dssim_vs_cache"]["mean"]
        d = f"${sgn(dv, 4)}$"
        if dv == best_d:
            d = rf"$\mathbf{{{sgn(dv, 4)}}}$"
    lines.append(f"{name} & {cell(a, 'speedup', '{:.2f}')} & {cell(a, 'ssim', '{:.4f}')} & "
                 f"{cell(a, 'psnr', '{:.2f}')} & {cell(a, 'lpips', '{:.4f}')} & {ir} & {d} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(HERE.parent / "tab_main.tex").write_text("\n".join(lines) + "\n")

t2 = rep["table2"]
lines = [r"\begin{tabular}{lccc}", r"\toprule",
         r"$K$ & Residual obj. & \ourmethod{} & Difference \\", r"\midrule"]
dmax = max(t2[k]["diff"]["mean"] for k in t2)
for k in ["1", "2", "4", "7", "14", "28"]:
    r = t2[k]
    diff = f"${sgn(r['diff']['mean'], 4)}$ ($t{{=}}{r['diff']['t']:.1f}$)"
    if r["diff"]["mean"] == dmax:
        diff = rf"$\mathbf{{{sgn(r['diff']['mean'], 4)}}}$ ($t{{=}}{r['diff']['t']:.1f}$)"
    lines.append(f"{k} & ${sgn(r['resid_gain']['mean'], 4)}$ ($\\sigma{{=}}{r['sigma_resid']:g}$) & "
                 f"${sgn(r['traj_gain']['mean'], 4)}$ ($\\sigma{{=}}{r['sigma_traj']:g}$) & {diff} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(HERE.parent / "tab_perk.tex").write_text("\n".join(lines) + "\n")
print("wrote tab_main.tex, tab_perk.tex")

# ---- Table 3: K=1 correctors under uniform vs frozen non-uniform anchors -------------
fs = rep["frozen"]["summary"]
lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{Uniform anchors} & \multicolumn{2}{c}{Frozen schedule} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Method & SSIM$\uparrow$ & LPIPS$\downarrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ \\", r"\midrule"]
for arm, name in (("plain", "Verbatim cache"), ("resid", "Residual obj."), ("traj", r"\ourmethod{}")):
    cells = []
    for sch in ("uniform", "frozen"):
        for m, f in (("ssim", max), ("lpips", min)):
            v = fs[sch][arm][m]
            s_ = f"{v:.4f}"
            cells.append(rf"\textbf{{{s_}}}" if v == f(fs[sch][a][m] for a in ("plain", "resid", "traj")) else s_)
    lines.append(f"{name} & " + " & ".join(cells) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(HERE.parent / "tab_sched.tex").write_text("\n".join(lines) + "\n")
print("wrote tab_sched.tex")
