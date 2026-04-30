"""SpaceNet7 Building Semantic Segmentation dataset (SN7MAPPING).

Inherits the AOI-level train/val/test split from Hafner et al. 2023.
Each item is a tiled single-timestamp image paired with its building mask.

Returns batches with keys ``image``, ``mask``.
"""

from __future__ import annotations

import json
from abc import abstractmethod
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from .spacenet7_cd import SN7_TRAIN, SN7_VAL, SN7_TEST

_ALL_AOIS = list(SN7_TRAIN) + list(SN7_VAL) + list(SN7_TEST)
_SPLITS = {"train": SN7_TRAIN, "val": SN7_VAL, "test": SN7_TEST}


class SN7MAPPING(Dataset):
    """SpaceNet7 building mapping (semantic segmentation) dataset.

    Parameters
    ----------
    root_path : str
        Path to the SpaceNet7 dataset root.
    split : str
        ``"train"``, ``"val"``, or ``"test"``.
    img_size : int
        Tile size.  The source images are 1024×1024, so ``img_size`` must
        evenly divide 1024.
    domain_shift : bool
        If ``True`` (default), splits are defined by AOI id.  If ``False``,
        uses a spatial within-scene split (for domain-adaptation experiments).
    i_split, j_split : int
        Pixel boundaries for the within-scene split (only used when
        ``domain_shift=False``).
    data_min, data_max : float
        Pixel range for normalisation to [-1, 1].
    """

    SN7_IMG_SIZE = 1024  # native image size

    def __init__(
        self,
        root_path: str,
        split: str = "train",
        img_size: int = 256,
        domain_shift: bool = True,
        i_split: int = 768,
        j_split: int = 512,
        data_min: float = 0.0,
        data_max: float = 255.0,
    ) -> None:
        assert self.SN7_IMG_SIZE % img_size == 0, (
            f"img_size {img_size} must evenly divide {self.SN7_IMG_SIZE}"
        )

        self.root_path = Path(root_path)
        self.split = split
        self.img_size = img_size
        self.data_min = data_min
        self.data_max = data_max

        metadata_file = self.root_path / "metadata_train.json"
        with open(metadata_file) as f:
            self.metadata = json.load(f)

        self.items = self._build_items(split, domain_shift, i_split, j_split)

    # ---------------------------------------------------------------------- #

    def _build_items(
        self, split: str, domain_shift: bool, i_split: int, j_split: int
    ) -> list[dict]:
        items: list[dict] = []

        if domain_shift:
            aoi_ids = _SPLITS[split]
        else:
            aoi_ids = _ALL_AOIS

        for aoi_id in aoi_ids:
            for ts in self.metadata[aoi_id]:
                if ts["mask"] or not ts["label"]:
                    continue

                base = {"aoi_id": ts["aoi_id"], "year": ts["year"], "month": ts["month"]}

                if domain_shift:
                    i_range = range(0, self.SN7_IMG_SIZE, self.img_size)
                    j_range = range(0, self.SN7_IMG_SIZE, self.img_size)
                else:
                    assert i_split % self.img_size == 0 and j_split % self.img_size == 0
                    if split == "train":
                        i_range = range(0, i_split, self.img_size)
                        j_range = range(0, self.SN7_IMG_SIZE, self.img_size)
                    elif split == "val":
                        i_range = range(i_split, self.SN7_IMG_SIZE, self.img_size)
                        j_range = range(0, j_split, self.img_size)
                    else:  # test
                        i_range = range(i_split, self.SN7_IMG_SIZE, self.img_size)
                        j_range = range(j_split, self.SN7_IMG_SIZE, self.img_size)

                for i in i_range:
                    for j in j_range:
                        items.append({**base, "i": i, "j": j})

        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.items[idx]
        aoi_id, year, month = item["aoi_id"], int(item["year"]), int(item["month"])
        i, j, s = item["i"], item["j"], self.img_size

        image = self._load_image(aoi_id, year, month)[:, i: i + s, j: j + s]
        mask = self._load_label(aoi_id, year, month)[i: i + s, j: j + s]

        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(mask).long(),
        }

    # ---------------------------------------------------------------------- #

    def _load_image(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        path = (
            self.root_path / "train" / aoi_id / "images_masked"
            / f"global_monthly_{year}_{month:02d}_mosaic_{aoi_id}.tif"
        )
        with rasterio.open(str(path)) as src:
            img = src.read(
                out_shape=(self.SN7_IMG_SIZE, self.SN7_IMG_SIZE),
                resampling=rasterio.enums.Resampling.nearest,
            )[:-1]  # drop alpha channel
        img = img.astype(np.float32)
        # Normalise to [-1, 1]
        img = 2.0 * (img - self.data_min) / (self.data_max - self.data_min) - 1.0
        return img

    def _load_label(self, aoi_id: str, year: int, month: int) -> np.ndarray:
        path = (
            self.root_path / "train" / aoi_id / "labels_raster"
            / f"global_monthly_{year}_{month:02d}_mosaic_{aoi_id}_Buildings.tif"
        )
        with rasterio.open(str(path)) as src:
            lbl = src.read(
                out_shape=(self.SN7_IMG_SIZE, self.SN7_IMG_SIZE),
                resampling=rasterio.enums.Resampling.nearest,
            )
        return (lbl > 0).squeeze().astype(np.int64)
