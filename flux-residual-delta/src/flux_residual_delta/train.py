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


@torch.no_grad()
def validate(model, loader, device) -> float:
    model.eval()
    values = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        prediction = model(
            batch["delta_h"], batch["anchor_residual"], batch["timestep"], batch["horizon"]
        )
        values.append(relative_mse(prediction, batch["target_delta_residual"]).cpu())
    model.train()
    return torch.cat(values).mean().item() if values else float("nan")


def fit(model, train_loader: DataLoader, val_loader: DataLoader, config: dict, output_dir: Path):
    device = torch.device("cuda")
    model.to(device=device, dtype=torch.bfloat16).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    accumulation = int(config["gradient_accumulation"])
    max_steps = int(config["max_steps"])
    warmup = int(config["warmup_steps"])
    optimizer_steps = 0
    best = math.inf
    progress = tqdm(total=max_steps, desc="optimizer steps")
    optimizer.zero_grad(set_to_none=True)
    for _epoch in range(int(config["epochs"])):
        for micro_step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = model(
                    batch["delta_h"], batch["anchor_residual"], batch["timestep"], batch["horizon"]
                )
                loss = relative_mse(prediction, batch["target_delta_residual"]).mean() / accumulation
            loss.backward()
            if micro_step % accumulation:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            scale = min(1.0, optimizer_steps / max(1, warmup))
            for group in optimizer.param_groups:
                group["lr"] = config["learning_rate"] * scale
            progress.update(1)
            progress.set_postfix(loss=float(loss) * accumulation)
            if optimizer_steps % int(config["validation_every"]) == 0:
                score = validate(model, val_loader, device)
                state = {"step": optimizer_steps, "model": model.state_dict(), "val_relative_mse": score}
                torch.save(state, output_dir / "last.pt")
                if score < best:
                    best = score
                    torch.save(state, output_dir / "best.pt")
            if optimizer_steps >= max_steps:
                progress.close()
                return
    progress.close()
