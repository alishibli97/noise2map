"""WHU Building Semantic Segmentation dataset.

Directory structure expected under ``root_dir``::

    <root_dir>/
        train/
            Image/
            Mask/
        test/
            Image/
            Mask/

Returns batches with keys ``image``, ``mask``.
"""

from __future__ import annotations

import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class WHUSegmentationDataset(Dataset):
    """WHU Building Segmentation dataset.

    Parameters
    ----------
    root_dir : str
        Path to the dataset root.
    split : str
        ``"train"`` or ``"test"``.
    image_transform : callable, optional
        Transform for RGB images.  Defaults to resize → ToTensor → Normalize.
    label_transform : callable, optional
        Transform for masks.  Defaults to resize → ToTensor.
    img_size : int
        Resize target.
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_transform=None,
        label_transform=None,
        img_size: int = 256,
    ) -> None:
        self.image_dir = os.path.join(root_dir, split, "Image")
        self.mask_dir = os.path.join(root_dir, split, "Mask")
        self.file_names = sorted(os.listdir(self.image_dir))

        self.image_transform = image_transform or T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.label_transform = label_transform or T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        name = self.file_names[idx]
        image = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name))

        return {
            "image": self.image_transform(image),
            "mask": self.label_transform(mask),
        }
