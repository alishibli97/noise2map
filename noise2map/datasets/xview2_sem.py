"""xView2 Santa-Rosa Wildfire Semantic Segmentation dataset.

Same image source as the CD counterpart but returns only the pre-disaster
image paired with its fire/damage mask — no temporal component.

Returns batches with keys ``image``, ``mask``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class XView2WildfireSemDataset(Dataset):
    """xView2 Wildfire building-damage Semantic Segmentation dataset.

    Parameters
    ----------
    root_dir : str
        Dataset root (parent of ``train/`` and ``test/``).
    split : str
        ``"train"`` or ``"test"``.
    img_size : int
        Resize images and masks to this spatial size.
    """

    _MISSING: dict[str, set[str]] = {
        "train": {"00000334", "00000192", "00000125", "00000247", "00000327", "00000258", "00000320"},
        "test": {"00000087", "00000289"},
    }

    def __init__(self, root_dir: str, split: str = "train", img_size: int = 256) -> None:
        self.img_size = img_size
        missing = self._MISSING.get(split, set())

        base = Path(root_dir) / split / "xBD" / "santa-rosa-wildfire"
        images_dir = base / "images"
        masks_dir = base / "masks"

        self.image_files = sorted([
            f for f in images_dir.glob("*_pre_disaster.png")
            if f.stem.split("_")[0] not in missing
            and (masks_dir / f.name).exists()
        ])
        self.masks_dir = masks_dir

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / img_path.name

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size))
        img = (img / 255.0) * 2.0 - 1.0

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")
        if self.img_size:
            mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = np.where(mask / 255.0 > 0.5, 1.0, 0.0).astype(np.float32)

        return {
            "image": torch.from_numpy(img).permute(2, 0, 1).float(),
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
        }
