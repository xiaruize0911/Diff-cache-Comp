#!/usr/bin/env python3
"""Fine-tune one sparse correction gate against an exact latent trajectory."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from sd15_residual_delta.config import load_config, save_resolved_config
from sd15_residual_delta.factory import build_surrogate_bank
from sd15_residual_delta.pipeline import load_sd15_pipeline
from sd15_residual_delta.runtime import SD15AttentionRuntime
from sd15_residual_delta.train import seed_everything
from train_rollout import encode_prompt, exact_trajectory, guided_noise


@torch.no_grad()
def cached_trajectory(
    unet,
    scheduler,
    prompt_embeds: torch.Tensor,
    seed: int,
    height: int,
    width: int,
    guidance_scale: float,
    anchor_steps: set[int],
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
    with SD15AttentionRuntime(
        unet,
        cache_interval=999,
        anchor_steps=anchor_steps,
    ):
        for timestep in scheduler.timesteps:
            noise = guided_noise(
                unet, scheduler, latents, timestep, prompt_embeds, guidance_scale
            )
            latents = scheduler.step(noise, timestep, latents).prev_sample
            trajectory.append(latents.detach().cpu())
    return trajectory


def relative_to_fixed_mse(
    prediction: torch.Tensor,
    exact: torch.Tensor,
    fixed: torch.Tensor,
) -> torch.Tensor:
    error = (prediction.float() - exact.float()).square().mean()
    baseline = (fixed.float() - exact.float()).square().mean().clamp_min(1e-10)
    return error / baseline


def run_fragment(
    unet,
    scheduler,
    surrogate,
    prompt_embeds: torch.Tensor,
    initial_latents: torch.Tensor,
    guidance_scale: float,
    anchor_steps: set[int],
    start_step: int,
    correction_step: int,
    module_id: int,
    horizon: int,
) -> torch.Tensor:
    latents = initial_latents
    with SD15AttentionRuntime(
        unet,
        cache_interval=999,
        anchor_steps=anchor_steps,
        surrogate_bank=surrogate,
        surrogate_module_horizon_keys={(module_id, horizon)},
        surrogate_step_module_keys={(correction_step, module_id)},
        step_offset=start_step,
    ):
        for step_index in range(start_step, correction_step + 1):
            timestep = scheduler.timesteps[step_index]
            if step_index < correction_step:
                with torch.no_grad():
                    noise = guided_noise(
                        unet,
                        scheduler,
                        latents,
                        timestep,
                        prompt_embeds,
                        guidance_scale,
                    )
                    latents = scheduler.step(noise, timestep, latents).prev_sample
            else:
                noise = guided_noise(
                    unet,
                    scheduler,
                    latents,
                    timestep,
                    prompt_embeds,
                    guidance_scale,
                )
                latents = scheduler.step(noise, timestep, latents).prev_sample
    return latents


@torch.no_grad()
def validate(
    unet,
    scheduler,
    surrogate,
    embeddings,
    exact_trajectories,
    fixed_trajectories,
    keys,
    cfg,
    train_cfg,
    device,
    dtype,
) -> float:
    surrogate.eval()
    values = []
    for prompt_index, seed in keys:
        prompt = train_cfg["prompts"][prompt_index]
        prediction = run_fragment(
            unet,
            scheduler,
            surrogate,
            embeddings[prompt].to(device=device, dtype=dtype),
            fixed_trajectories[(prompt_index, seed)][int(train_cfg["start_step"])].to(
                device=device, dtype=dtype
            ),
            float(cfg["model"]["guidance_scale"]),
            {int(step) for step in train_cfg["anchor_steps"]},
            int(train_cfg["start_step"]),
            int(train_cfg["correction_step"]),
            int(train_cfg["module_id"]),
            int(train_cfg["horizon"]),
        )
        target_index = int(train_cfg["correction_step"]) + 1
        values.append(
            relative_to_fixed_mse(
                prediction,
                exact_trajectories[(prompt_index, seed)][target_index].to(
                    device=device, dtype=dtype
                ),
                fixed_trajectories[(prompt_index, seed)][target_index].to(
                    device=device, dtype=dtype
                ),
            ).item()
        )
    surrogate.train()
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-file", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg["targeted_rollout"]
    seed_everything(int(train_cfg["seed"]))
    rng = random.Random(int(train_cfg["seed"]))
    prompts = [
        line.strip()
        for line in Path(args.prompt_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_cfg = {**train_cfg, "prompts": prompts}
    output = Path(train_cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, output)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    pipeline = load_sd15_pipeline(cfg["model"]["model_id"], torch_dtype=dtype)
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

    trajectory_seeds = [int(seed) for seed in train_cfg["trajectory_seeds"]]
    anchor_steps = {int(step) for step in train_cfg["anchor_steps"]}
    exact_trajectories = {}
    fixed_trajectories = {}
    trajectory_cache_value = train_cfg.get("trajectory_cache")
    trajectory_cache = (
        Path(str(trajectory_cache_value)) if trajectory_cache_value else None
    )
    trajectory_start = time.perf_counter()
    if trajectory_cache is not None and trajectory_cache.exists():
        cached = torch.load(trajectory_cache, map_location="cpu", weights_only=False)
        if cached["prompts"] != prompts or cached["seeds"] != trajectory_seeds:
            raise ValueError("trajectory cache prompts or seeds do not match config")
        exact_trajectories = cached["exact"]
        fixed_trajectories = cached["fixed"]
        print(json.dumps({"trajectory_cache": str(trajectory_cache), "loaded": True}), flush=True)
    else:
        for prompt_index, prompt in enumerate(prompts):
            prompt_embeds = embeddings[prompt].to(device=device, dtype=dtype)
            for trajectory_seed in trajectory_seeds:
                key = (prompt_index, trajectory_seed)
                exact_trajectories[key] = exact_trajectory(
                    unet,
                    pipeline.scheduler,
                    prompt_embeds,
                    trajectory_seed,
                    int(cfg["model"]["height"]),
                    int(cfg["model"]["width"]),
                    float(cfg["model"]["guidance_scale"]),
                    device,
                    dtype,
                )
                fixed_trajectories[key] = cached_trajectory(
                    unet,
                    pipeline.scheduler,
                    prompt_embeds,
                    trajectory_seed,
                    int(cfg["model"]["height"]),
                    int(cfg["model"]["width"]),
                    float(cfg["model"]["guidance_scale"]),
                    anchor_steps,
                    device,
                    dtype,
                )
            print(json.dumps({"trajectories": prompt_index + 1, "total": len(prompts)}), flush=True)
        if trajectory_cache is not None:
            trajectory_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "prompts": prompts,
                    "seeds": trajectory_seeds,
                    "exact": exact_trajectories,
                    "fixed": fixed_trajectories,
                },
                trajectory_cache,
            )
    trajectory_seconds = time.perf_counter() - trajectory_start

    val_prompt_count = int(train_cfg["val_prompt_count"])
    split_index = len(prompts) - val_prompt_count
    train_keys = [
        (prompt_index, seed)
        for prompt_index in range(split_index)
        for seed in trajectory_seeds
    ]
    val_keys = [
        (prompt_index, trajectory_seeds[0])
        for prompt_index in range(split_index, len(prompts))
    ]
    surrogate = build_surrogate_bank(cfg).to(device=device, dtype=dtype)
    initial = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    surrogate.load_state_dict(initial["model"])
    trainable_patterns = [str(value) for value in train_cfg.get("trainable_patterns", [])]
    if trainable_patterns:
        for name, parameter in surrogate.named_parameters():
            parameter.requires_grad_(any(pattern in name for pattern in trainable_patterns))
    trainable_parameters = [
        parameter for parameter in surrogate.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("no surrogate parameters selected for targeted rollout training")
    surrogate.train()
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    initial_validation = validate(
        unet,
        pipeline.scheduler,
        surrogate,
        embeddings,
        exact_trajectories,
        fixed_trajectories,
        val_keys,
        cfg,
        train_cfg,
        device,
        dtype,
    )
    history = [{"step": 0, "val_relative_to_fixed_mse": initial_validation}]
    best = initial_validation
    torch.save(
        {
            "step": 0,
            "model": surrogate.state_dict(),
            "val_relative_to_fixed_mse": initial_validation,
            "source_checkpoint": args.checkpoint,
        },
        output / "best.pt",
    )
    print(json.dumps(history[0]), flush=True)
    training_start = time.perf_counter()
    fragments_per_step = int(train_cfg.get("fragments_per_step", 1))
    for optimizer_step in range(1, int(train_cfg["max_steps"]) + 1):
        optimizer.zero_grad(set_to_none=True)
        fragment_values = []
        for _ in range(fragments_per_step):
            prompt_index, trajectory_seed = rng.choice(train_keys)
            prompt = prompts[prompt_index]
            prediction = run_fragment(
                unet,
                pipeline.scheduler,
                surrogate,
                embeddings[prompt].to(device=device, dtype=dtype),
                fixed_trajectories[(prompt_index, trajectory_seed)][int(train_cfg["start_step"])].to(
                    device=device, dtype=dtype
                ),
                float(cfg["model"]["guidance_scale"]),
                anchor_steps,
                int(train_cfg["start_step"]),
                int(train_cfg["correction_step"]),
                int(train_cfg["module_id"]),
                int(train_cfg["horizon"]),
            )
            target_index = int(train_cfg["correction_step"]) + 1
            fragment_loss = relative_to_fixed_mse(
                prediction,
                exact_trajectories[(prompt_index, trajectory_seed)][target_index].to(
                    device=device, dtype=dtype
                ),
                fixed_trajectories[(prompt_index, trajectory_seed)][target_index].to(
                    device=device, dtype=dtype
                ),
            )
            (fragment_loss / fragments_per_step).backward()
            fragment_values.append(fragment_loss.detach())
        loss = torch.stack(fragment_values).mean()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            surrogate.parameters(), float(train_cfg["grad_clip"])
        )
        optimizer.step()
        record = {
            "step": optimizer_step,
            "train_relative_to_fixed_mse": loss.item(),
            "gradient_norm": float(gradient_norm),
        }
        if optimizer_step % int(train_cfg["validation_every"]) == 0:
            record["val_relative_to_fixed_mse"] = validate(
                unet,
                pipeline.scheduler,
                surrogate,
                embeddings,
                exact_trajectories,
                fixed_trajectories,
                val_keys,
                cfg,
                train_cfg,
                device,
                dtype,
            )
            state = {
                "step": optimizer_step,
                "model": surrogate.state_dict(),
                "val_relative_to_fixed_mse": record["val_relative_to_fixed_mse"],
                "source_checkpoint": args.checkpoint,
            }
            torch.save(state, output / "last.pt")
            if record["val_relative_to_fixed_mse"] < best:
                best = record["val_relative_to_fixed_mse"]
                torch.save(state, output / "best.pt")
        history.append(record)
        print(json.dumps(record), flush=True)

    report = {
        "trajectory_seconds": trajectory_seconds,
        "training_seconds": time.perf_counter() - training_start,
        "train_cases": len(train_keys),
        "validation_cases": len(val_keys),
        "fragments_per_step": fragments_per_step,
        "trainable_patterns": trainable_patterns,
        "best_val_relative_to_fixed_mse": best,
        "initial_val_relative_to_fixed_mse": initial_validation,
        "history": history,
    }
    (output / "targeted_rollout_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
