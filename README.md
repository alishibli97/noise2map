# Noise2Map

Official implementation of **Noise2Map: End-to-End Diffusion Model for Semantic Segmentation and Change Detection** (IEEE TGRS 2026).

> Ali Shibli, Andrea Nascetti, Yifang Ban  
> Division of Geoinformatics, KTH Royal Institute of Technology

[[Paper]](https://doi.org/10.1109/TGRS.2026.3687393) · [[Pretrained Weights]](https://huggingface.co/ali97/noise2map)

---

## Overview

Noise2Map repurposes the diffusion denoising trajectory for discriminative remote sensing tasks. Rather than iterative sampling, it predicts semantic or change maps in a **single forward pass** using task-specific structured noise schedules.

- **Semantic Segmentation (SS)**: single image → segmentation mask  
- **Change Detection (CD)**: bi-temporal image pair → change mask  

Key results (ranked 1st on both tasks across SpaceNet7, WHU, and xView2-Wildfire):

| Task | SpaceNet7 F1 | WHU F1 | xView2 F1 |
|------|-------------|--------|-----------|
| SS   | 72.02       | 95.69  | 86.90     |
| CD   | 71.43       | 95.27  | 86.91     |

---

## Installation

```bash
git clone https://github.com/alishibli97/noise2map.git
cd noise2map
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

---

## Pretrained Weights

The denoising UNet backbone is pretrained on 10k images from the [AID dataset](https://captain-whu.github.io/DiRS/) using a DDPM denoising objective. Weights are hosted on HuggingFace and loaded automatically at runtime.

| Config key | HuggingFace path | Description |
|---|---|---|
| `aid_10k` | `ali97/noise2map` / `aid-10k` | AID pretrained, min-max norm, Google DDPM init (**recommended**) |
| `satellite` | `ali97/noise2map` / `sat2gen` | Pretrained on MajorTOM satellite imagery |
| `imagenet` | `ali97/noise2map` / `imagenet2gen` | ImageNet pretrained |
| `google` | `ali97/noise2map` / `ddpm-church` | Google DDPM church checkpoint |
| `None` | — | Random initialisation (ablation) |

See [HuggingFace](https://huggingface.co/ali97/noise2map) for all pretrained weights.

---

## Datasets

Download datasets and place them under `data/`:

```
data/
├── whu-cd/          # WHU Building CD  — train/{A,B,OUT}/  test/{A,B,OUT}/
├── WHU/             # WHU Building SS  — train/{Image,Mask}/  test/{Image,Mask}/
├── xview2/          # xView2 Wildfire  — train/xBD/santa-rosa-wildfire/{images,masks}/
└── spacenet7/       # SpaceNet7        — train/<aoi_id>/{images_masked,labels_raster}/
```

Dataset-specific filtering (e.g. WHU-CD skips image pairs with no changes) is handled automatically and cached on first run.

---

## Training

Each experiment is configured via a YAML file in `configs/`. To train:

```bash
python train.py --config configs/whu_cd.yaml
python train.py --config configs/whu_sem.yaml
python train.py --config configs/xview2_cd.yaml
python train.py --config configs/xview2_sem.yaml
python train.py --config configs/spacenet7_cd.yaml
python train.py --config configs/spacenet7_sem.yaml
```

Override any config value on the command line:

```bash
python train.py --config configs/whu_cd.yaml training.batch_size=4 training.num_epochs=100
```

SLURM example scripts are in `scripts/`.

---

## Evaluation

```bash
python evaluate.py --config configs/whu_cd.yaml --checkpoint path/to/best_model.pth
```

Outputs per-class and mean F1, IoU, Precision, Recall, and Accuracy.

---

## Citation

```bibtex
@article{shibli2026noise2map,
  title={Noise2Map: End-to-End Diffusion Model for Semantic Segmentation and Change Detection},
  author={Shibli, Ali and Nascetti, Andrea and Ban, Yifang},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2026},
}
```

---

## Acknowledgement

This research is part of the EO-AI4GlobalChange project funded by Digital Futures, Stockholm, Sweden.
