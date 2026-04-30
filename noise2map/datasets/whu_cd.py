"""WHU Building Change Detection dataset.

Directory structure expected under ``root_dir``::

    <root_dir>/
        train/
            A/          # pre-event images
            B/          # post-event images
            OUT/        # binary change masks
        test/
            A/
            B/
            OUT/

Image pairs where the change mask contains *only* background (no changed
pixels) are filtered out automatically.  The filtered file list is cached as a
JSON file inside ``root_dir`` to avoid re-scanning on subsequent runs.

Returns batches with keys ``pre_image``, ``post_image``, ``label``.
"""

from __future__ import annotations

import json
import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


class WHUChangeDetectionDataset(Dataset):
    """WHU Building CD dataset with automatic no-change filtering.

    Parameters
    ----------
    root_dir : str
        Path to the dataset root (parent of ``train/`` and ``test/``).
    split : str
        ``"train"`` or ``"test"``.
    image_transform : callable, optional
        Transform applied to each RGB image tensor.
        Defaults to ``ToTensor`` + ``Normalize([0.5]*3, [0.5]*3)``.
    label_transform : callable, optional
        Transform applied to each label image.
        Defaults to ``ToTensor`` (keeps values in {0, 1}).
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        image_transform=None,
        label_transform=None,
    ) -> None:
        self.split_dir = os.path.join(root_dir, split)

        self.image_transform = image_transform or T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        self.label_transform = label_transform or T.ToTensor()

        self.pre_dir = os.path.join(self.split_dir, "A")
        self.post_dir = os.path.join(self.split_dir, "B")
        self.label_dir = os.path.join(self.split_dir, "OUT")

        cache_path = os.path.join(root_dir, f"_cache_{split}_filtered.json")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                self.file_names = json.load(f)
        else:
            all_files = sorted(os.listdir(self.pre_dir))
            self.file_names = self._filter(all_files)
            with open(cache_path, "w") as f:
                json.dump(self.file_names, f)

    # ---------------------------------------------------------------------- #

    def _filter(self, file_names: list[str]) -> list[str]:
        """Keep only pairs whose mask contains at least one changed pixel."""
        kept = []
        for name in tqdm(file_names, desc="Filtering WHU-CD (no-change removal)"):
            label = T.ToTensor()(Image.open(os.path.join(self.label_dir, name)).convert("L"))
            unique = torch.unique(label)
            if len(unique) > 1:  # both 0 and 1 present
                kept.append(name)
        return kept

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        name = self.file_names[idx]

        pre = Image.open(os.path.join(self.pre_dir, name)).convert("RGB")
        post = Image.open(os.path.join(self.post_dir, name)).convert("RGB")
        label = Image.open(os.path.join(self.label_dir, name)).convert("L")

        return {
            "pre_image": self.image_transform(pre),
            "post_image": self.image_transform(post),
            "label": self.label_transform(label),
        }
