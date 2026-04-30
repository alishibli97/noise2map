"""SpaceNet7 Change Detection dataset.

Uses the train/val/test AOI split from:
  Hafner et al., "Semi-supervised urban change detection …", Remote Sensing, 2023.
  https://doi.org/10.3390/rs15215135

Directory structure expected under ``root_dir``::

    <root_dir>/
        metadata_train.json
        train/
            <aoi_id>/
                images_masked/   # monthly PlanetScope GeoTIFF mosaics
                labels_raster/   # building footprint rasters

Returns batches with keys ``pre_image``, ``post_image``, ``label``.
The ``label`` is the pixel-wise XOR change mask between consecutive months.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# AOI split definitions (from Hafner et al. 2023)
# ---------------------------------------------------------------------------

SN7_TRAIN = [
    "L15-0331E-1257N_1327_3160_13", "L15-0358E-1220N_1433_3310_13",
    "L15-0457E-1135N_1831_3648_13", "L15-0487E-1246N_1950_3207_13",
    "L15-0577E-1243N_2309_3217_13", "L15-0586E-1127N_2345_3680_13",
    "L15-0595E-1278N_2383_3079_13", "L15-0632E-0892N_2528_4620_13",
    "L15-0683E-1006N_2732_4164_13", "L15-0924E-1108N_3699_3757_13",
    "L15-1015E-1062N_4061_3941_13", "L15-1138E-1216N_4553_3325_13",
    "L15-1203E-1203N_4815_3378_13", "L15-1204E-1202N_4816_3380_13",
    "L15-1209E-1113N_4838_3737_13", "L15-1210E-1025N_4840_4088_13",
    "L15-1276E-1107N_5105_3761_13", "L15-1298E-1322N_5193_2903_13",
    "L15-1389E-1284N_5557_3054_13", "L15-1438E-1134N_5753_3655_13",
    "L15-1439E-1134N_5759_3655_13", "L15-1481E-1119N_5927_3715_13",
    "L15-1538E-1163N_6154_3539_13", "L15-1615E-1206N_6460_3366_13",
    "L15-1669E-1153N_6678_3579_13", "L15-1669E-1160N_6679_3549_13",
    "L15-1672E-1207N_6691_3363_13", "L15-1703E-1219N_6813_3313_13",
    "L15-1709E-1112N_6838_3742_13", "L15-1716E-1211N_6864_3345_13",
]
SN7_VAL = [
    "L15-0357E-1223N_1429_3296_13", "L15-0361E-1300N_1446_2989_13",
    "L15-0368E-1245N_1474_3210_13", "L15-0566E-1185N_2265_3451_13",
    "L15-0614E-0946N_2459_4406_13", "L15-0760E-0887N_3041_4643_13",
    "L15-1014E-1375N_4056_2688_13", "L15-1049E-1370N_4196_2710_13",
    "L15-1185E-0935N_4742_4450_13", "L15-1289E-1169N_5156_3514_13",
    "L15-1296E-1198N_5184_3399_13", "L15-1615E-1205N_6460_3370_13",
    "L15-1617E-1207N_6468_3360_13", "L15-1669E-1160N_6678_3548_13",
    "L15-1748E-1247N_6993_3202_13",
]
SN7_TEST = [
    "L15-0387E-1276N_1549_3087_13", "L15-0434E-1218N_1736_3318_13",
    "L15-0506E-1204N_2027_3374_13", "L15-0544E-1228N_2176_3279_13",
    "L15-0977E-1187N_3911_3441_13", "L15-1025E-1366N_4102_2726_13",
    "L15-1172E-1306N_4688_2967_13", "L15-1200E-0847N_4802_4803_13",
    "L15-1204E-1204N_4819_3372_13", "L15-1335E-1166N_5342_3524_13",
    "L15-1479E-1101N_5916_3785_13", "L15-1690E-1211N_6763_3346_13",
    "L15-1691E-1211N_6764_3347_13", "L15-1848E-0793N_7394_5018_13",
]

_SPLITS = {"train": SN7_TRAIN, "val": SN7_VAL, "test": SN7_TEST}


class SpaceNet7CDDataset(Dataset):
    """SpaceNet7 Change Detection dataset.

    Parameters
    ----------
    root_dir : str
        Path to the SpaceNet7 dataset root.
    split : str
        ``"train"``, ``"val"``, or ``"test"``.
    metadata_file : str, optional
        Path to ``metadata_train.json``.  Defaults to ``<root_dir>/metadata_train.json``.
    tile_size : int
        Tile size for cropping the 1024×1024 images.
    min_time_gap : int
        Minimum days between consecutive image pairs (default 5).
    min_val : float, optional
        Minimum pixel value for normalisation (e.g. 0.0).
    max_val : float, optional
        Maximum pixel value for normalisation (e.g. 255.0).
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        metadata_file: str | None = None,
        tile_size: int = 256,
        min_time_gap: int = 5,
        min_val: float | None = 0.0,
        max_val: float | None = 255.0,
    ) -> None:
        self.root_dir = root_dir
        self.tile_size = tile_size
        self.min_time_gap = timedelta(days=min_time_gap)
        self.min_val = min_val
        self.max_val = max_val

        metadata_file = metadata_file or os.path.join(root_dir, "metadata_train.json")
        with open(metadata_file) as f:
            self.metadata = json.load(f)

        self.aoi_ids = _SPLITS[split]
        self.items = self._build_item_list()

    def _build_item_list(self) -> list[dict]:
        items = []
        for aoi_id in self.aoi_ids:
            images_dir = os.path.join(self.root_dir, "train", aoi_id, "images_masked")
            labels_dir = os.path.join(self.root_dir, "train", aoi_id, "labels_raster")

            image_files = sorted(os.listdir(images_dir))
            for i in range(len(image_files) - 1):
                pre, post = image_files[i], image_files[i + 1]
                pre_date = self._parse_date(pre)
                post_date = self._parse_date(post)

                if post_date - pre_date < self.min_time_gap:
                    continue

                pre_label = pre.replace(".tif", "_Buildings.tif")
                post_label = post.replace(".tif", "_Buildings.tif")

                if not (
                    os.path.exists(os.path.join(labels_dir, pre_label))
                    and os.path.exists(os.path.join(labels_dir, post_label))
                ):
                    continue

                with rasterio.open(os.path.join(images_dir, pre)) as src:
                    h, w = src.shape

                for row in range(0, h, self.tile_size):
                    for col in range(0, w, self.tile_size):
                        if row + self.tile_size <= h and col + self.tile_size <= w:
                            items.append({
                                "pre_image": os.path.join(images_dir, pre),
                                "post_image": os.path.join(images_dir, post),
                                "pre_label": os.path.join(labels_dir, pre_label),
                                "post_label": os.path.join(labels_dir, post_label),
                                "row": row,
                                "col": col,
                            })
        return items

    @staticmethod
    def _parse_date(filename: str) -> datetime:
        parts = filename.split("_")
        return datetime(int(parts[2]), int(parts[3]), 1)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.items[idx]
        r, c, ts = item["row"], item["col"], self.tile_size

        pre = self._read_image(item["pre_image"], r, c)
        post = self._read_image(item["post_image"], r, c)

        pre_lbl = self._read_label(item["pre_label"], r, c)
        post_lbl = self._read_label(item["post_label"], r, c)
        change = (np.abs(pre_lbl.astype(int) - post_lbl.astype(int)) > 0).astype(np.int64)

        return {
            "pre_image": torch.from_numpy(pre),
            "post_image": torch.from_numpy(post),
            "label": torch.from_numpy(change),
        }

    def _read_image(self, path: str, row: int, col: int) -> np.ndarray:
        ts = self.tile_size
        with rasterio.open(path) as src:
            img = src.read(out_dtype="float32")[:3]
            img = img[:, row: row + ts, col: col + ts]
        if self.min_val is not None and self.max_val is not None:
            img = 2.0 * (img - self.min_val) / (self.max_val - self.min_val) - 1.0
        return img

    def _read_label(self, path: str, row: int, col: int) -> np.ndarray:
        ts = self.tile_size
        with rasterio.open(path) as src:
            lbl = src.read(1, out_dtype="uint8")
            lbl = lbl[row: row + ts, col: col + ts]
        return (lbl > 0).astype(np.uint8)
