from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import relative_mse


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare(item: dict, device: torch.device) -> dict:
    delta_h = item["delta_h"].to(device)
    batch = delta_h.shape[0]
    return {
        "delta_h": delta_h,
        "anchor_residual": item["anchor_residual"].to(device),
        "target": item["target_delta_residual"].to(device),
        "timestep": torch.full((batch,), float(item["timestep"]), device=device),
        "horizon": torch.full((batch,), float(item["horizon"]), device=device),
        "module_id": torch.full(
            (batch,), int(item["module_id"]), device=device, dtype=torch.long
        ),
    }


@torch.no_grad()
def validate(model, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    values = []
    for item in loader:
        batch = prepare(item, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(
                batch["delta_h"],
                batch["anchor_residual"],
                batch["timestep"],
                batch["horizon"],
                batch["module_id"],
            )
        values.append(relative_mse(prediction, batch["target"]).cpu())
    model.train()
    return torch.cat(values).mean().item() if values else float("nan")


def fit(model, train_loader, val_loader, config: dict, output_dir: Path) -> None:
    device = torch.device("cuda")
    model.to(device=device, dtype=torch.bfloat16).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    accumulation = int(config["gradient_accumulation"])
    max_steps = int(config["max_steps"])
    warmup = int(config["warmup_steps"])
    best = math.inf
    optimizer_step = 0
    micro_step = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=max_steps, desc="optimizer steps")
    while optimizer_step < max_steps:
        for item in train_loader:
            micro_step += 1
            batch = prepare(item, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(
                    batch["delta_h"],
                    batch["anchor_residual"],
                    batch["timestep"],
                    batch["horizon"],
                    batch["module_id"],
                )
                loss = relative_mse(prediction, batch["target"]).mean() / accumulation
            loss.backward()
            if micro_step % accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["grad_clip"]))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            learning_rate = float(config["learning_rate"]) * min(
                1.0, optimizer_step / max(1, warmup)
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            progress.update(1)
            progress.set_postfix(loss=float(loss) * accumulation)
            if optimizer_step % int(config["validation_every"]) == 0:
                score = validate(model, val_loader, device)
                state = {
                    "step": optimizer_step,
                    "model": model.state_dict(),
                    "val_relative_mse": score,
                }
                torch.save(state, output_dir / "last.pt")
                if score < best:
                    best = score
                    torch.save(state, output_dir / "best.pt")
            if optimizer_step >= max_steps:
                break
    progress.close()
