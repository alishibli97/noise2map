"""xView2 Santa-Rosa Wildfire Change Detection dataset.

Directory structure expected under ``root_dir``::

    <root_dir>/
        train/
            xBD/santa-rosa-wildfire/
                images/   # *_pre_disaster.png  and  *_post_disaster.png
                masks/    # binary change masks (same filename as pre image)
        test/
            ...

Returns batches with keys ``pre_image``, ``post_image``, ``label``.

"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class XView2WildfireCDDataset(Dataset):
    """xView2 Wildfire building damage Change Detection dataset.

    Parameters
    ----------
    root_dir : str
        Dataset root (parent of ``train/`` and ``test/``).
    split : str
        ``"train"`` or ``"test"``.
    img_size : int
        Resize images and masks to this spatial size.
    """

    # Image IDs with missing mask files — excluded automatically
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
        pre_path = self.image_files[idx]
        post_path = pre_path.with_name(
            pre_path.stem.replace("_pre_disaster", "_post_disaster") + ".png"
        )
        mask_path = self.masks_dir / pre_path.name

        pre = self._load_image(str(pre_path))
        post = self._load_image(str(post_path))
        mask = self._load_mask(str(mask_path))

        return {
            "pre_image": torch.from_numpy(pre).permute(2, 0, 1).float(),
            "post_image": torch.from_numpy(post).permute(2, 0, 1).float(),
            "label": torch.from_numpy(mask).unsqueeze(0).float(),
        }

    # ---------------------------------------------------------------------- #

    def _load_image(self, path: str) -> np.ndarray:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size))
        # Normalise to [-1, 1]
        return (img / 255.0) * 2.0 - 1.0

    def _load_mask(self, path: str) -> np.ndarray:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {path}")
        if self.img_size:
            mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = mask / 255.0
        return np.where(mask > 0.5, 1.0, 0.0).astype(np.float32)
