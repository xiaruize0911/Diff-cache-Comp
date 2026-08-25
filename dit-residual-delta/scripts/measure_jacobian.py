#!/usr/bin/env python
"""Spectral radius of the per-injection-site Jacobian, measured not argued.

The deployed map at one injection site is out = h + R_a + C(h - h_a, R_a, h_a),
so its Jacobian with respect to the current state is J = I + dC/d(delta_h). The
paper's recursion delta_{b+1} ~= (I + J_b) delta_b predicts that what matters is
the product of these over the K sites a reuse step passes through. Three of the
four families have J = I exactly -- plain caching, Taylor forecasting and Block
Caching's scale-shift all read only cached quantities, so there is nothing to
measure and rho = 1 by construction. This measures the one family that does read
the current state.

Two things the measurement must not get wrong:

  * The deployed operator includes the training-time input/output scaling, a
    factor of scale_delta_residual / scale_delta_h that is order 30 at K=1.
    Differentiating the bare model understates the Jacobian by that factor, so we
    differentiate the SurrogateBank, which is what actually runs.
  * fp16 power iteration does not converge. The bank is loaded in fp32 here.

Power iteration runs on real captured states, so this is the Jacobian where the
corrector is actually evaluated, not at some arbitrary point. Tokens are the 48
subsampled per slot; the map is token-wise apart from one attention layer, so the
operator is the true one restricted to those tokens.
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path

import torch

from dit_residual_delta.dataset import SlotDataset
from dit_residual_delta.surrogate import load_surrogate_bank


def spectral_radius(bank, batch, iterations: int, seed: int = 0, scale: float = 1.0) -> float:
    """Largest |eigenvalue| of I + dC/d(delta_h) by power iteration.

    Iterates with the TRANSPOSE (reverse-mode VJP) rather than forward-mode JVP:
    a matrix and its transpose have identical eigenvalues, so the spectral radius
    is the same, and forward AD is unavailable here -- the corrector's attention
    layer lowers to a fused SDPA kernel with no forward-mode rule.
    """
    delta_h = batch["delta_h"].float().requires_grad_(True)
    args = (batch["anchor_residual"].float(), batch["anchor_hidden"].float(),
            batch["timestep"], batch["horizon"], batch["block_id"])

    def correction(x):
        return bank(x, *args) * scale

    output = correction(delta_h)
    generator = torch.Generator(device=delta_h.device).manual_seed(seed)
    v = torch.randn(delta_h.shape, device=delta_h.device, generator=generator)
    v /= v.norm()
    value = float("nan")
    for _ in range(iterations):
        (jtv,) = torch.autograd.grad(output, delta_h, grad_outputs=v,
                                     retain_graph=True, allow_unused=True)
        if jtv is None:
            # delta_h never entered the graph: the corrector cannot depend on the
            # current state, so J = I exactly and rho = 1 analytically. This is a
            # stronger result than a numerical 1.0 -- autograd is reporting that
            # the dependency does not exist, not that it is small.
            return 1.0
        w = v + jtv                      # (I + dC/ddelta_h)^T v
        value = float(w.norm())
        v = w / max(value, 1e-30)
    return value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--slots-dir", required=True)
    ap.add_argument("--num-blocks", type=int, required=True)
    ap.add_argument("--slots-per-site", type=int, default=8)
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="deployed injection strength sigma; J = I + sigma dC/dh")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    bank = load_surrogate_bank(args.checkpoint, dtype=torch.float32)
    ds = SlotDataset(args.slots_dir)
    block_id = ds.data["block_id"].long()

    per_site = {}
    for b in range(args.num_blocks):
        index = (block_id == b).nonzero(as_tuple=True)[0][: args.slots_per_site]
        if not len(index):
            continue
        values = []
        for i in index.tolist():
            batch = ds.batch(torch.tensor([i], device=ds.device))
            values.append(spectral_radius(bank, batch, args.iterations, scale=args.scale))
        values.sort()
        per_site[str(b)] = {"median": values[len(values) // 2],
                            "min": values[0], "max": values[-1],
                            "n_states": len(values)}
        print(json.dumps({"site": b, "rho_median": round(per_site[str(b)]["median"], 4)}),
              flush=True)

    medians = [v["median"] for v in per_site.values()]
    log_product = sum(math.log(m) for m in medians)
    out = {"checkpoint": args.checkpoint, "num_blocks": args.num_blocks,
           "scale": args.scale,
           "per_site": per_site,
           "rho_median_over_sites": sorted(medians)[len(medians) // 2],
           "rho_max_over_sites": max(medians),
           "log_product_over_sites": log_product,
           "amplification_per_reuse_step": math.exp(log_product)}
    Path(args.output).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in out.items() if k != "per_site"}, indent=1))


if __name__ == "__main__":
    main()
