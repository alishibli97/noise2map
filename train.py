"""
Noise2Map unified training script.

Usage
-----
    python train.py --config configs/whu_cd.yaml
    python train.py --config configs/spacenet7_sem.yaml training.batch_size=4

Any key in the YAML config can be overridden from the command line using
dot-notation, e.g. ``training.lr=5e-5`` or ``model.pretrained=satellite``.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import wandb
import yaml
from diffusers import DDIMScheduler
from loguru import logger
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from noise2map import Noise2Map
from noise2map.datasets import (
    WHUChangeDetectionDataset,
    WHUSegmentationDataset,
    XView2WildfireCDDataset,
    XView2WildfireSemDataset,
    SpaceNet7CDDataset,
    SN7MAPPING,
)


# ──────────────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_config(config_path: str, overrides: list[str]) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for override in overrides:
        key_path, _, value = override.partition("=")
        keys = key_path.strip().split(".")
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        # attempt numeric coercion
        try:
            value = float(value) if "." in value else int(value)
        except ValueError:
            pass
        if value == "true":
            value = True
        elif value == "false":
            value = False
        node[keys[-1]] = value
    return cfg


# ──────────────────────────────────────────────────────────────────────────────
# Dataset factory
# ──────────────────────────────────────────────────────────────────────────────

def build_dataset(cfg: dict, split: str):
    task = cfg["task"]        # "cd" or "sem"
    dataset = cfg["dataset"]  # "whu", "xview2", "spacenet7"
    data = cfg["data"]

    root = data["root_dir"]
    img_size = data.get("img_size", 256)

    if dataset == "whu" and task == "cd":
        return WHUChangeDetectionDataset(root_dir=root, split=split)
    if dataset == "whu" and task == "sem":
        return WHUSegmentationDataset(root_dir=root, split=split, img_size=img_size)
    if dataset == "xview2" and task == "cd":
        return XView2WildfireCDDataset(root_dir=root, split=split, img_size=img_size)
    if dataset == "xview2" and task == "sem":
        return XView2WildfireSemDataset(root_dir=root, split=split, img_size=img_size)
    if dataset == "spacenet7" and task == "cd":
        return SpaceNet7CDDataset(
            root_dir=root, split=split, tile_size=img_size,
            min_val=data.get("min_val", 0.0), max_val=data.get("max_val", 255.0),
        )
    if dataset == "spacenet7" and task == "sem":
        return SN7MAPPING(
            root_path=root, split=split, img_size=img_size,
            domain_shift=data.get("domain_shift", True),
        )
    raise ValueError(f"Unknown dataset/task combination: {dataset}/{task}")


# ──────────────────────────────────────────────────────────────────────────────
# Batch accessor — normalises key names across datasets
# ──────────────────────────────────────────────────────────────────────────────

def _get(batch: dict, *keys):
    for k in keys:
        if k in batch:
            return batch[k]
    raise KeyError(f"None of {keys} found in batch keys: {list(batch.keys())}")


def unpack_cd(batch: dict, device):
    pre = _get(batch, "pre_image").to(device)
    post = _get(batch, "post_image").to(device)
    label = _get(batch, "label", "change_mask").to(device)
    return pre, post, label


def unpack_sem(batch: dict, device):
    image = _get(batch, "image").to(device)
    mask = _get(batch, "mask").to(device)
    return image, mask


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

def weighted_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    change_weight: float,
    no_change_weight: float,
) -> torch.Tensor:
    target = target.squeeze(1).long()
    weights = torch.tensor(
        [no_change_weight, change_weight], dtype=logits.dtype, device=logits.device
    )
    return torch.nn.functional.cross_entropy(logits, target, weight=weights)


# ──────────────────────────────────────────────────────────────────────────────
# Training step
# ──────────────────────────────────────────────────────────────────────────────

def training_step(
    batch: dict,
    task: str,
    model: Noise2Map,
    scheduler: DDIMScheduler,
    cfg: dict,
    device: torch.device,
) -> torch.Tensor:
    t_cfg = cfg["training"]
    change_w = t_cfg.get("change_weight", 1.0)
    no_change_w = t_cfg.get("no_change_weight", 1.0)

    if task == "cd":
        pre, post, label = unpack_cd(batch, device)
        x = torch.cat([pre, post], dim=1)
        noise = torch.cat([post, pre], dim=1)  # reversed pair as structured noise
    else:
        image, label = unpack_sem(batch, device)
        x = image
        noise = image  # self as noise (SS)

    timesteps = torch.randint(
        0, scheduler.config.num_train_timesteps, (x.size(0),), device=device
    ).long()
    x_noisy = scheduler.add_noise(x, noise, timesteps)

    logits = model(x_noisy, timesteps)
    loss = weighted_ce_loss(logits, label, change_w, no_change_w)
    return loss


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model: Noise2Map,
    val_loader: DataLoader,
    scheduler: DDIMScheduler,
    task: str,
    cfg: dict,
    device: torch.device,
) -> float:
    model.eval()
    t_cfg = cfg["training"]
    change_w = t_cfg.get("change_weight", 1.0)
    no_change_w = t_cfg.get("no_change_weight", 1.0)
    losses = []

    for batch in tqdm(val_loader, desc="Validating", leave=False):
        if task == "cd":
            pre, post, label = unpack_cd(batch, device)
            x = torch.cat([pre, post], dim=1)
            noise = torch.cat([post, pre], dim=1)
        else:
            image, label = unpack_sem(batch, device)
            x = image
            noise = image

        # Fixed mid-timestep for validation
        t = scheduler.config.num_train_timesteps // 2
        timesteps = torch.full((x.size(0),), t, device=device, dtype=torch.long)
        x_noisy = scheduler.add_noise(x, noise, timesteps)
        logits = model(x_noisy, timesteps)
        losses.append(weighted_ce_loss(logits, label, change_w, no_change_w).item())

    model.train()
    return sum(losses) / len(losses)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config, overrides)

    task = cfg["task"]
    t_cfg = cfg["training"]
    m_cfg = cfg["model"]
    log_cfg = cfg.get("logging", {})
    out_cfg = cfg["output"]

    # ── Logging ──────────────────────────────────────────────────────────────
    if log_cfg.get("write_to_wandb", False):
        wandb.init(
            project=log_cfg.get("wandb_project", "noise2map"),
            config=cfg,
            name=log_cfg.get("run_name", None),
        )

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_count = torch.cuda.device_count()
    logger.info(f"Device: {device}  |  GPUs: {gpu_count}")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_ds = build_dataset(cfg, "train")
    val_split = "val" if cfg["dataset"] == "spacenet7" else "test"
    val_ds = build_dataset(cfg, val_split)

    train_loader = DataLoader(
        train_ds,
        batch_size=t_cfg["batch_size"],
        shuffle=True,
        num_workers=t_cfg.get("num_workers", 4),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=t_cfg["batch_size"],
        shuffle=False,
        num_workers=t_cfg.get("num_workers", 4),
    )
    logger.info(f"Train: {len(train_ds)}  |  Val: {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    in_channels = 6 if task == "cd" else 3
    model = Noise2Map(
        in_channels=in_channels,
        out_channels=m_cfg.get("out_channels", 2),
        img_scale=cfg["data"].get("img_size", 256),
        pretrained=m_cfg.get("pretrained", "aid_10k"),
        freeze_unet=m_cfg.get("freeze_unet", False),
    )
    if gpu_count > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.print_trainable_parameters() if hasattr(model, "print_trainable_parameters") else None
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimiser & scheduler ────────────────────────────────────────────────
    scheduler = DDIMScheduler()
    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg.get("lr", 1e-4))
    scaler = GradScaler()
    grad_accum = t_cfg.get("grad_accumulation_steps", 2)

    # ── Output dir ───────────────────────────────────────────────────────────
    ckpt_dir = out_cfg["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val_loss = float("inf")
    val_period = t_cfg.get("val_period", 5)
    num_epochs = t_cfg["num_epochs"]

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        step = 0
        epoch_losses = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
            loss = training_step(batch, task, model, scheduler, cfg, device)

            if (step + 1) % grad_accum == 0:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            epoch_losses.append(loss.item())
            step += 1

            if log_cfg.get("write_to_wandb", False):
                wandb.log({"train/loss": loss.item()})

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        logger.info(f"Epoch {epoch} | avg loss: {avg_loss:.4f}")

        # Validation
        if epoch % val_period == 0:
            val_loss = validate(model, val_loader, scheduler, task, cfg, device)
            logger.info(f"Epoch {epoch} | val loss: {val_loss:.4f}")
            if log_cfg.get("write_to_wandb", False):
                wandb.log({"val/loss": val_loss, "epoch": epoch})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(ckpt_dir, f"best_model_epoch_{epoch}.pth")
                torch.save(model.state_dict(), best_path)
                logger.info(f"New best saved → {best_path}")

        # Late-epoch checkpoint
        if epoch >= num_epochs - 5:
            path = os.path.join(ckpt_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), path)

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    if log_cfg.get("write_to_wandb", False):
        wandb.finish()


if __name__ == "__main__":
    main()
