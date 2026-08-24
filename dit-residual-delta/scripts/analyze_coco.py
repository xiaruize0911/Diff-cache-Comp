#!/usr/bin/env python
"""Turn the COCO run into the paper's primary evaluation table.

The statistical situation here is much better than on our templated split. Each of the
~2000 COCO captions describes a distinct image, so captions are independent units and a
paired test over them needs no clustering -- unlike the templated split, where 24
prompts collapsed into 6 subjects and left intervals resting on 6 clusters.
"""
from __future__ import annotations
import argparse, json, statistics as st
from pathlib import Path

import numpy as np


def paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = st.fmean(d)
    se = st.stdev(d) / n ** 0.5
    return m, m / se, sum(x > 0 for x in d), n, m - 1.96 * se, m + 1.96 * se


def split_half_baseline(npz_path):
    """FID/KID between two halves of the EXACT set: the same-distribution baseline.

    This is the honest calibration of small-sample bias, on the real feature space we
    report in. Any variant FID below this is indistinguishable from "same distribution".
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location("efc", "scripts/eval_fid_coco.py")
    src = open("scripts/eval_fid_coco.py").read()
    from scipy import linalg
    ns = {"np": np, "linalg": linalg}
    exec(src[src.index("def kid("): src.index("class _Feats")], ns)
    z = np.load(npz_path)
    e = z["exact"].astype(np.float64)
    rng = np.random.default_rng(0)
    idx = rng.permutation(e.shape[0])
    a, b = e[idx[: len(idx) // 2]], e[idx[len(idx) // 2:]]
    fid = ns["frechet"](a.mean(0), np.cov(a, rowvar=False), b.mean(0), np.cov(b, rowvar=False))
    km, ks = ns["kid"](a, b)
    return {"n_per_half": int(a.shape[0]), "fid_split_half": fid,
            "kid_split_half": km, "kid_subset_std": ks}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--features", default=None,
                   help="inception_features.npz; enables the split-half same-distribution baseline")
    p.add_argument("--per-image", default=None,
                   help="optional npz of per-image metrics for paired tests")
    args = p.parse_args()
    r = json.loads(Path(args.results).read_text())
    v = r["variants"]
    print(f"COCO val2017, n={r['n']} captions (each a distinct image), "
          f"exact CLIP {r['exact_clip']:.4f}\n")
    hdr = f"{'variant':12s} {'FID':>8s} {'KID x1e3':>12s} {'SSIM':>7s} {'LPIPS':>7s} {'PSNR':>6s} {'dCLIP':>8s}"
    print(hdr); print("-" * len(hdr))
    for name, m in v.items():
        print(f"{name:12s} {m['fid_vs_exact']:8.2f} "
              f"{m['kid_vs_exact']*1e3:7.3f}±{m['kid_subset_std']*1e3:<4.2f} "
              f"{m['ssim']:7.4f} {m['lpips']:7.4f} {m['psnr']:6.2f} {m['clip_vs_exact']:+8.4f}")
    if args.features:
        bl = split_half_baseline(args.features)
        print(f"\nsame-distribution baseline (two halves of the exact set, "
              f"n={bl['n_per_half']} each):")
        print(f"   FID {bl['fid_split_half']:.2f}   <- how much of any FID above is bias")
        print(f"   KID {bl['kid_split_half']*1e3:+.3f}e-3 +- {bl['kid_subset_std']*1e3:.3f}e-3")
    print("\nRelative to plain caching at the same anchor schedule (fixed_i5):")
    base = v.get("fixed_i5")
    if base:
        for name, m in v.items():
            if name == "fixed_i5":
                continue
            print(f"  {name:12s} FID {m['fid_vs_exact']-base['fid_vs_exact']:+7.2f}   "
                  f"KID {(m['kid_vs_exact']-base['kid_vs_exact'])*1e3:+7.3f}e-3   "
                  f"SSIM {m['ssim']-base['ssim']:+.4f}   CLIP {m['clip_vs_exact']-base['clip_vs_exact']:+.4f}")
    if "std_i5" in v and "inv_i5" in v:
        s, i = v["std_i5"], v["inv_i5"]
        print("\nThe loss contrast (scale-invariant - standard) on COCO:")
        print(f"  FID  {i['fid_vs_exact']:.2f} vs {s['fid_vs_exact']:.2f}  -> {i['fid_vs_exact']-s['fid_vs_exact']:+.2f}")
        print(f"  KID  {i['kid_vs_exact']*1e3:.3f} vs {s['kid_vs_exact']*1e3:.3f} e-3 -> "
              f"{(i['kid_vs_exact']-s['kid_vs_exact'])*1e3:+.3f}e-3 "
              f"(subset sd {max(i['kid_subset_std'],s['kid_subset_std'])*1e3:.3f}e-3)")
        for k in ("ssim", "lpips", "psnr", "clip_vs_exact"):
            print(f"  {k:14s} {i[k]-s[k]:+.4f}")


if __name__ == "__main__":
    main()
