#!/usr/bin/env python
"""Fine-tune the corrector against the latent trajectory instead of the residual.

This project's central finding is that residual accuracy does not predict deployed
quality. The direct response is to stop training on the proxy: the exact trajectory
is free -- it is produced by the same forward pass that produces the input -- so the
corrector can be optimised against the thing actually cared about.

Two facts make this tractable, both checked rather than assumed:

  * A cached block never runs the real block. runtime.py returns
    `hidden + cached_base + delta_residual`, so the graph over one reuse step is 28
    corrector forwards, not 28 transformer blocks. Memory is modest.
  * PixArtSigmaPipeline.__call__ carries @torch.no_grad(), and enabling grad inside
    the hook would not help: the surrounding adds and every following block stay in
    no_grad, so the chain breaks. Hence the denoising loop is reimplemented here,
    which also gives per-step control over where the graph is cut.

Backprop is truncated at anchor steps. The cache is refreshed with exact values
there, so a corrector call can only influence later intervals through the latent;
detaching at that boundary bounds memory to a single anchor interval and matches
where the corrector's influence is actually renewed.

Training starts from a residual-trained checkpoint rather than from scratch: that
checkpoint also carries the feature scale statistics the bank needs, and starting
from the residual solution makes the contrast "what does switching objective buy"
rather than "can this train at all".
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path

import torch

from dit_residual_delta.config import load_config
from dit_residual_delta.pipeline import load_pixart_pipeline, load_prompt_embeddings
from dit_residual_delta.surrogate import load_surrogate_bank
from dit_residual_delta.variants import build_variant, runtime_for_variant


def decode(pipe, latents, checkpoint=True):
    """Differentiable VAE decode of one latent to [0,1] image space.

    Checkpointed by default: the decoder at 512x512 stores several GB of activations,
    which is what pushed this over the card while a ladder rung was training. Trading
    one recomputation for that memory is the right side of the deal here, since the
    decode happens once per training step against a 20-step rollout.
    """
    x = latents / pipe.vae.config.scaling_factor
    def _dec(z):
        return pipe.vae.decode(z, return_dict=False)[0]
    img = (torch.utils.checkpoint.checkpoint(_dec, x, use_reentrant=False)
           if checkpoint and torch.is_grad_enabled() else _dec(x))
    return (img / 2 + 0.5).clamp(0, 1)


def rollout(pipe, embed, steps, cfg, generator, *, variant=None, bank=None,
            anchor_every=None, grad=False, reference=None, trunc_intervals=1):
    """One denoising trajectory. Returns the per-step latents.

    With `reference` given, accumulates a relative squared error against it at every
    step where the corrector was active, and returns (latents, loss).
    """
    sched = pipe.scheduler
    sched.set_timesteps(steps, device="cuda")          # also resets solver history
    timesteps = sched.timesteps
    pos, neg = embed["prompt_embeds"], embed["negative_prompt_embeds"]
    pos_m, neg_m = embed["prompt_attention_mask"], embed["negative_prompt_attention_mask"]
    prompt_embeds = torch.cat([neg, pos]).to("cuda")
    prompt_mask = torch.cat([neg_m, pos_m]).to("cuda")
    lat_ch = pipe.transformer.config.in_channels
    shape = (1, lat_ch, cfg["height"] // pipe.vae_scale_factor,
             cfg["width"] // pipe.vae_scale_factor)
    latents = torch.randn(shape, generator=generator, device="cuda",
                          dtype=prompt_embeds.dtype) * sched.init_noise_sigma
    added = {"resolution": None, "aspect_ratio": None}
    guidance = float(cfg["guidance_scale"])

    out, loss = [], latents.new_zeros((), dtype=torch.float32)
    anchors_seen = 0
    for i, t in enumerate(timesteps):
        is_anchor = anchor_every is None or (i % anchor_every == 0)
        # With trunc_intervals=1 an anchor step can stay out of the graph: nothing in
        # it depends on the corrector and the graph is cut there anyway. Spanning more
        # than one interval requires the opposite -- the latent flows THROUGH the
        # anchor step, so leaving it in no_grad would sever the chain and silently
        # give a one-interval gradient under a multi-interval label. Cost of keeping
        # it: one full 0.6B forward's activations per anchor.
        in_graph = grad and (trunc_intervals > 1 or not is_anchor)
        with torch.set_grad_enabled(in_graph):
            model_in = sched.scale_model_input(torch.cat([latents] * 2), t)
            ts = t.expand(model_in.shape[0]) if torch.is_tensor(t) else \
                torch.tensor([t], device="cuda").expand(model_in.shape[0])
            pred = pipe.transformer(model_in, encoder_hidden_states=prompt_embeds,
                                    encoder_attention_mask=prompt_mask, timestep=ts,
                                    added_cond_kwargs=added, return_dict=False)[0]
            un, tx = pred.chunk(2)
            pred = un + guidance * (tx - un)
            if pipe.transformer.config.out_channels // 2 == lat_ch:
                pred = pred.chunk(2, dim=1)[0]
            latents = sched.step(pred, t, latents, return_dict=False)[0]

        if reference is not None and not is_anchor:
            # fp32: an fp16 sum of squares over 4x64x64 overflows past 65504 even at
            # magnitude 3, which silently produced inf/inf = NaN on the second step
            ref = reference[i].float()
            cur = latents.float()
            loss = loss + ((cur - ref) ** 2).sum() / (ref ** 2).sum()
        out.append(latents if reference is None else latents.detach())
        if is_anchor:
            anchors_seen += 1
            if anchors_seen % trunc_intervals == 0:
                latents = latents.detach()  # cut the graph every trunc_intervals anchors
    return (out, loss) if reference is not None else out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--init-checkpoint", required=True,
                    help="residual-trained corrector; also supplies the feature scales")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--cache-interval", type=int, default=5)
    ap.add_argument("--num-segments", type=int, default=1)
    ap.add_argument("--surrogate-scale", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--learning-rate", type=float, default=1e-5)
    ap.add_argument("--train-images", type=int, default=48)
    ap.add_argument("--val-images", type=int, default=12)
    ap.add_argument("--validate-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=4001)
    ap.add_argument("--objective", choices=["latent", "image"], default="latent",
                    help="what the loss is computed on. 'latent' matches the trajectory; "
                         "'image' decodes the final latent through the VAE and scores "
                         "LPIPS against the exact image, which is what deployment is "
                         "measured with. Extending the horizon improved the latent "
                         "objective (0.833 vs 0.871) but moved the image metric by "
                         "-0.0039 (t=-0.8), i.e. the proxy became the bottleneck.")
    ap.add_argument("--adaptive-clip", type=float, default=0.0,
                    help="clip at this multiple of the running median gradient norm "
                         "(0 = fixed clip at 1.0, the previous behaviour)")
    ap.add_argument("--truncate-intervals", type=int, default=1,
                    help="how many anchor intervals the backprop spans. >1 pulls the "
                         "anchor steps into the graph (a full 0.6B forward each) and "
                         "makes the gradient traverse the (I+J) product measured at "
                         "rho~10.8 over 20 steps, so watch grad_norm and peak memory")
    a = ap.parse_args()

    model_cfg = load_config(a.config)["model"]
    n_steps = int(model_cfg["num_inference_steps"])
    pipe = load_pixart_pipeline(with_text_encoder=False).to("cuda")
    pipe.transformer.requires_grad_(False)
    bank = load_surrogate_bank(a.init_checkpoint).float()   # fp32 master weights
    bank.requires_grad_(True)
    trainable = [p for p in bank.parameters() if p.requires_grad]
    print(json.dumps({"trainable_params": sum(p.numel() for p in trainable)}), flush=True)

    prompts = [l.strip() for l in Path(a.prompt_file).read_text().splitlines() if l.strip()]
    emb = load_prompt_embeddings(a.embeddings)
    train_p = prompts[: a.train_images]
    val_p = prompts[a.train_images : a.train_images + a.val_images]
    if not val_p:
        raise SystemExit("no validation prompts left after --train-images")

    spec = {"name": f"latent_i{a.cache_interval}_k{a.num_segments}",
            "cache_interval": a.cache_interval, "surrogate_scale": a.surrogate_scale}
    if a.num_segments != 28:
        spec["num_segments"] = a.num_segments
    n_blocks = len(pipe.transformer.transformer_blocks)
    variant = build_variant(spec, n_steps, n_blocks)

    opt = torch.optim.AdamW(trainable, lr=a.learning_rate, weight_decay=0.0)
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)

    perceptual = None
    if a.objective == "image":
        import lpips as lpips_pkg
        # called directly, not through metrics.lpips_alex: that wrapper is inside
        # torch.no_grad() and returns a float, so it cannot carry a gradient
        perceptual = lpips_pkg.LPIPS(net="alex").to("cuda").eval()
        perceptual.requires_grad_(False)
        if a.truncate_intervals < 5:
            raise SystemExit(
                "--objective image needs --truncate-intervals covering the whole "
                "rollout: the graph is cut at every anchor otherwise, so only the "
                "last interval's corrector calls would receive gradient")

    def reference_for(prompt, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        with torch.no_grad():
            return rollout(pipe, emb[prompt], n_steps, model_cfg, g)

    def image_loss(prompt, seed, ref_latents, grad):
        """LPIPS between the decoded final latent and the exact image."""
        with torch.no_grad():
            ref_img = decode(pipe, ref_latents[-1])
        g = torch.Generator(device="cuda").manual_seed(seed)
        with torch.set_grad_enabled(grad):
            got = rollout(pipe, emb[prompt], n_steps, model_cfg, g,
                          anchor_every=a.cache_interval, grad=grad,
                          trunc_intervals=a.truncate_intervals)
            cand = decode(pipe, got[-1])
            return perceptual(cand * 2 - 1, ref_img * 2 - 1).mean()

    def evaluate():
        tot = 0.0
        for j, p in enumerate(val_p):
            sd = a.seed + 90000 + j
            ref = reference_for(p, sd)
            with runtime_for_variant(pipe.transformer, variant, bank):
                if a.objective == "image":
                    with torch.no_grad():
                        tot += float(image_loss(p, sd, ref, grad=False))
                    continue
                g = torch.Generator(device="cuda").manual_seed(sd)
                with torch.no_grad():
                    _, l = rollout(pipe, emb[p], n_steps, model_cfg, g,
                                   anchor_every=a.cache_interval, grad=False, reference=ref)
            tot += float(l)
        return tot / len(val_p)

    history, best = [], {"objective": float("inf"), "step": -1}

    recent: list[float] = []
    start = time.perf_counter()
    for step in range(1, a.steps + 1):
        p = train_p[(step - 1) % len(train_p)]
        seed = a.seed + (step - 1) // len(train_p)
        ref = reference_for(p, seed)
        with runtime_for_variant(pipe.transformer, variant, bank):
            if a.objective == "image":
                loss = image_loss(p, seed, ref, grad=True)
            else:
                g = torch.Generator(device="cuda").manual_seed(seed)
                _, loss = rollout(pipe, emb[p], n_steps, model_cfg, g,
                                  anchor_every=a.cache_interval, grad=True, reference=ref,
                                  trunc_intervals=a.truncate_intervals)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Fixed clipping at 1.0 against a grad_norm that swings 6.8-124.6 at full
        # horizon means almost every step is scaled by 1/norm, i.e. the magnitude is
        # discarded and only the direction survives. An adaptive clip tracks the
        # running median instead, so ordinary steps pass through and only genuine
        # spikes are cut -- the cheap alternative to bootstrapping for what is really
        # a variance-control problem.
        if a.adaptive_clip:
            gn = torch.nn.utils.clip_grad_norm_(trainable, float("inf"))  # measure only
            recent.append(float(gn))
            if len(recent) > 50: recent.pop(0)
            thresh = sorted(recent)[len(recent) // 2] * a.adaptive_clip
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, max(thresh, 1e-8))
        else:
            gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if step == 1:
            # the point of the truncation is a bounded graph; report it rather than assume
            reached = sum(1 for p in trainable if p.grad is not None and float(p.grad.abs().sum()) > 0)
            print(json.dumps({"peak_alloc_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                              "params_with_nonzero_grad": reached,
                              "trainable_tensors": len(trainable),
                              "grad_norm_step1": round(float(gnorm), 5),
                              "loss_step1": round(float(loss), 5)}), flush=True)
        opt.step()
        if step % a.validate_every == 0 or step == a.steps:
            v = evaluate()
            history.append({"step": step, "train_loss": float(loss), "val_objective": v})
            print(json.dumps({"step": step, "train_loss": round(float(loss), 5),
                              "grad_norm": round(float(gnorm), 4),
                              "val_objective": round(v, 5),
                              "seconds": round(time.perf_counter() - start)}), flush=True)
            if v < best["objective"]:
                best = {"objective": v, "step": step}
                torch.save({"model": {k: v.half() for k, v in bank.model.state_dict().items()}, "config": bank.model.cfg.__dict__,
                            "scale": bank.scale, "step": step, "val_objective": v},
                           out / "best.pt")
    (out / "train_report.json").write_text(json.dumps(
        {"args": vars(a), "history": history, "best": best,
         "objective": a.objective}, indent=2))
    print(json.dumps({"best_val_objective": best["objective"],
                      "best_step": best["step"]}, indent=1))


if __name__ == "__main__":
    main()
