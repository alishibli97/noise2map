"""
Noise2Map unified evaluation script.

Usage
-----
    python evaluate.py --config configs/whu_cd.yaml --checkpoint path/to/best_model.pth

Prints per-class and mean F1, IoU, Precision, Recall, and Accuracy.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusers import DDIMScheduler

from noise2map import Noise2Map
from train import build_dataset, unpack_cd, unpack_sem


# ──────────────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model: Noise2Map,
    loader: DataLoader,
    scheduler: DDIMScheduler,
    task: str,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model over the full loader and return (predictions, targets)."""
    model.eval()
    all_preds, all_targets = [], []

    # Use the final timestep (t = T-1) for inference — cleanest input
    fixed_t = scheduler.config.num_train_timesteps - 1

    for batch in tqdm(loader, desc="Evaluating"):
        if task == "cd":
            pre, post, label = unpack_cd(batch, device)
            x = torch.cat([pre, post], dim=1)
            noise = torch.cat([post, pre], dim=1)
        else:
            image, label = unpack_sem(batch, device)
            x = image
            noise = image

        timesteps = torch.full((x.size(0),), fixed_t, device=device, dtype=torch.long)
        x_noisy = scheduler.add_noise(x, noise, timesteps)

        logits = model(x_noisy, timesteps)                       # (B, C, H, W)
        preds = torch.argmax(logits, dim=1).cpu().numpy()        # (B, H, W)
        targets = label.squeeze(1).cpu().numpy()                 # (B, H, W)

        all_preds.append(preds)
        all_targets.append(targets)

    return (
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_targets, axis=0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> None:
    flat_p = preds.flatten().astype(np.int64)
    flat_t = targets.flatten().astype(np.int64)

    f1 = f1_score(flat_t, flat_p, average=None, zero_division=0)
    iou = jaccard_score(flat_t, flat_p, average=None, zero_division=0)
    prec = precision_score(flat_t, flat_p, average=None, zero_division=0)
    rec = recall_score(flat_t, flat_p, average=None, zero_division=0)
    acc = accuracy_score(flat_t, flat_p)

    for i in range(len(f1)):
        print(
            f"  Class {i}: F1 = {f1[i]:.4f} | IoU = {iou[i]:.4f} | "
            f"Prec = {prec[i]:.4f} | Rec = {rec[i]:.4f}"
        )
    print(
        f"\n  Mean F1:   {f1.mean():.4f}"
        f"\n  Mean IoU:  {iou.mean():.4f}"
        f"\n  Mean Prec: {prec.mean():.4f}"
        f"\n  Mean Rec:  {rec.mean():.4f}"
        f"\n  Accuracy:  {acc:.4f}"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    task = cfg["task"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Dataset
    val_split = "val" if cfg["dataset"] == "spacenet7" else "test"
    val_ds = build_dataset(cfg, val_split)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"].get("batch_size", 2),
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 4),
    )
    logger.info(f"Eval set: {len(val_ds)} samples")

    # Model
    in_channels = 6 if task == "cd" else 3
    model = Noise2Map(
        in_channels=in_channels,
        out_channels=cfg["model"].get("out_channels", 2),
        img_scale=cfg["data"].get("img_size", 256),
        pretrained=cfg["model"].get("pretrained", "aid_10k"),
        freeze_unet=False,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    # Strip DataParallel prefix if present
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model.load_state_dict(state_dict)
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    scheduler = DDIMScheduler()
    preds, targets = run_inference(model, val_loader, scheduler, task, device)

    print("\n── Results ──────────────────────────────")
    compute_metrics(preds, targets)


if __name__ == "__main__":
    main()
