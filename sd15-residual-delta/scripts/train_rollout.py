#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from sd15_residual_delta.config import load_config
from sd15_residual_delta.factory import build_surrogate_bank
from sd15_residual_delta.pipeline import load_sd15_pipeline
from sd15_residual_delta.runtime import SD15AttentionRuntime
from sd15_residual_delta.train import seed_everything


def encode_prompt(pipeline, prompt: str, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        positive, negative = pipeline.encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=None,
        )
    return torch.cat([negative, positive], dim=0).detach()


def guided_noise(
    unet,
    scheduler,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    prompt_embeds: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    model_input = torch.cat([latents, latents], dim=0)
    model_input = scheduler.scale_model_input(model_input, timestep)
    prediction = unet(
        model_input,
        timestep,
        encoder_hidden_states=prompt_embeds,
        return_dict=True,
    ).sample
    unconditional, conditional = prediction.chunk(2)
    return unconditional + guidance_scale * (conditional - unconditional)


@torch.no_grad()
def exact_trajectory(
    unet,
    scheduler,
    prompt_embeds: torch.Tensor,
    seed: int,
    height: int,
    width: int,
    guidance_scale: float,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, unet.config.in_channels, height // 8, width // 8),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    latents = latents * scheduler.init_noise_sigma
    trajectory = [latents.detach().cpu()]
    for timestep in scheduler.timesteps:
        noise = guided_noise(
            unet, scheduler, latents, timestep, prompt_embeds, guidance_scale
        )
        latents = scheduler.step(noise, timestep, latents).prev_sample
        trajectory.append(latents.detach().cpu())
    return trajectory


def normalized_latent_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    error = (prediction.float() - target.float()).square().mean()
    scale = target.float().square().mean().clamp_min(1e-8)
    return error / scale


def aggregate_fragment_losses(
    fragment_losses: list[torch.Tensor], tail_fraction: float, lambda_tail: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not fragment_losses:
        raise ValueError("at least one rollout fragment is required")
    if not 0.0 < tail_fraction <= 1.0:
        raise ValueError("tail_fraction must be in (0, 1]")
    if lambda_tail < 0.0:
        raise ValueError("lambda_tail must be non-negative")
    stacked = torch.stack(fragment_losses)
    mean_loss = stacked.mean()
    tail_count = max(1, math.ceil(len(fragment_losses) * tail_fraction))
    tail_loss = torch.topk(stacked, k=tail_count, largest=True).values.mean()
    return mean_loss + lambda_tail * tail_loss, mean_loss, tail_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit-prompts", type=int)
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["rollout_train"]
    max_steps = args.max_steps or int(train_cfg["max_steps"])
    fragments_per_step = int(train_cfg.get("fragments_per_step", 1))
    tail_fraction = float(train_cfg.get("tail_fraction", 1.0))
    lambda_tail = float(train_cfg.get("lambda_tail", 0.0))
    checkpoint_steps = {
        int(value) for value in train_cfg.get("checkpoint_steps", [])
    }
    if fragments_per_step < 1:
        raise ValueError("fragments_per_step must be positive")
    seed_everything(int(train_cfg["seed"]))
    random_generator = random.Random(int(train_cfg["seed"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = [
        line.strip()
        for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit_prompts:
        prompts = prompts[: args.limit_prompts]
    if not prompts:
        raise ValueError("prompt file is empty")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    pipeline = load_sd15_pipeline(
        cfg["model"]["model_id"], torch_dtype=dtype
    )
    pipeline.scheduler.set_timesteps(
        int(cfg["model"]["num_inference_steps"]), device=device
    )
    pipeline.text_encoder.to(device=device, dtype=dtype)
    embeddings = {
        prompt: encode_prompt(pipeline, prompt, device).cpu() for prompt in prompts
    }
    pipeline.text_encoder.to("cpu")
    pipeline.vae.to("cpu")
    unet = pipeline.unet.to(device=device, dtype=dtype).eval()
    unet.requires_grad_(False)

    surrogate = build_surrogate_bank(cfg).to(device=device, dtype=dtype)
    initial = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    surrogate.load_state_dict(initial["model"])
    surrogate.train()
    optimizer = torch.optim.AdamW(
        surrogate.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    trajectories = {}
    trajectory_start = time.perf_counter()
    for prompt_index, prompt in enumerate(prompts):
        prompt_embeds = embeddings[prompt].to(device=device, dtype=dtype)
        for trajectory_seed in train_cfg["trajectory_seeds"]:
            key = (prompt_index, int(trajectory_seed))
            trajectories[key] = exact_trajectory(
                unet,
                pipeline.scheduler,
                prompt_embeds,
                int(trajectory_seed),
                int(cfg["model"]["height"]),
                int(cfg["model"]["width"]),
                float(cfg["model"]["guidance_scale"]),
                device,
                dtype,
            )
    trajectory_seconds = time.perf_counter() - trajectory_start

    allowed_anchors = [
        int(step)
        for step in train_cfg["anchor_steps"]
        if int(step) + 3 <= len(pipeline.scheduler.timesteps)
    ]
    if not allowed_anchors:
        raise ValueError("no anchor step has room for two reuse steps")

    torch.cuda.reset_peak_memory_stats()
    training_start = time.perf_counter()
    history = []
    for optimizer_step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        fragment_losses = []
        fragment_records = []
        for fragment_index in range(fragments_per_step):
            prompt_index = random_generator.randrange(len(prompts))
            prompt = prompts[prompt_index]
            trajectory_seed = int(
                random_generator.choice(train_cfg["trajectory_seeds"])
            )
            anchor_step = int(random_generator.choice(allowed_anchors))
            trajectory = trajectories[(prompt_index, trajectory_seed)]
            prompt_embeds = embeddings[prompt].to(device=device, dtype=dtype)
            latents = trajectory[anchor_step].to(device=device, dtype=dtype)
            step_losses = []

            with SD15AttentionRuntime(
                unet,
                cache_interval=int(cfg["reuse"]["cache_interval"]),
                surrogate_bank=surrogate,
            ):
                for offset in range(3):
                    step_index = anchor_step + offset
                    timestep = pipeline.scheduler.timesteps[step_index]
                    noise = guided_noise(
                        unet,
                        pipeline.scheduler,
                        latents,
                        timestep,
                        prompt_embeds,
                        float(cfg["model"]["guidance_scale"]),
                    )
                    latents = pipeline.scheduler.step(
                        noise, timestep, latents
                    ).prev_sample
                    if offset > 0:
                        target = trajectory[step_index + 1].to(
                            device=device, dtype=dtype
                        )
                        step_losses.append(normalized_latent_mse(latents, target))

            fragment_loss = (
                float(train_cfg["lambda_step_1"]) * step_losses[0]
                + float(train_cfg["lambda_step_2"]) * step_losses[1]
            )
            fragment_losses.append(fragment_loss)
            fragment_records.append(
                {
                    "fragment": fragment_index,
                    "prompt_index": prompt_index,
                    "trajectory_seed": trajectory_seed,
                    "anchor_step": anchor_step,
                    "loss": fragment_loss.item(),
                    "step_1_loss": step_losses[0].item(),
                    "step_2_loss": step_losses[1].item(),
                }
            )

        loss, mean_fragment_loss, tail_fragment_loss = aggregate_fragment_losses(
            fragment_losses, tail_fraction, lambda_tail
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            surrogate.parameters(), float(train_cfg["grad_clip"])
        )
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
            raise FloatingPointError(
                f"non-finite rollout optimization: loss={loss}, grad={gradient_norm}"
            )
        optimizer.step()
        record = {
            "step": optimizer_step,
            "loss": loss.item(),
            "mean_fragment_loss": mean_fragment_loss.item(),
            "tail_fragment_loss": tail_fragment_loss.item(),
            "gradient_norm": float(gradient_norm),
            "fragments": fragment_records,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if optimizer_step in checkpoint_steps:
            torch.save(
                {
                    "step": optimizer_step,
                    "model": surrogate.state_dict(),
                    "source_checkpoint": args.checkpoint,
                    "rollout_history": history,
                },
                output_dir / f"step_{optimizer_step}.pt",
            )

    training_seconds = time.perf_counter() - training_start
    checkpoint = {
        "step": max_steps,
        "model": surrogate.state_dict(),
        "source_checkpoint": args.checkpoint,
        "rollout_history": history,
    }
    torch.save(checkpoint, output_dir / "last.pt")
    report = {
        "prompts": prompts,
        "trajectory_seeds": [int(value) for value in train_cfg["trajectory_seeds"]],
        "trajectory_seconds": trajectory_seconds,
        "training_seconds": training_seconds,
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "max_steps": max_steps,
        "fragments_per_step": fragments_per_step,
        "tail_fraction": tail_fraction,
        "lambda_tail": lambda_tail,
        "checkpoint_steps": sorted(checkpoint_steps),
        "history": history,
        "gradient_smoke_passed": all(
            record["gradient_norm"] > 0 and record["loss"] > 0 for record in history
        ),
    }
    (output_dir / "rollout_train_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
