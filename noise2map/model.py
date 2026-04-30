"""
Noise2Map denoising model.

Wraps a pretrained attention UNet (UNet2DModelFlex) with lightweight
input/output projection layers to support arbitrary channel counts and
image scales. The model is pretrained via self-supervised denoising on
unlabeled satellite imagery and fine-tuned end-to-end for discriminative
tasks (semantic segmentation and change detection).

"""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger

from .unet_2d_flex import UNet2DModelFlex


# ---------------------------------------------------------------------------
# Pretrained weight registry
# ---------------------------------------------------------------------------

_WEIGHT_REGISTRY: dict[str, str] = {
    "aid_10k":      ("ali97/noise2map", "aid-10k"),
    "satellite":    ("ali97/noise2map", "sat2gen"),         # S2
    "imagenet":     ("ali97/noise2map", "imagenet2gen"),
    "google":       ("ali97/noise2map", "ddpm-church"),
}


class Noise2Map(nn.Module):
    """Noise2Map denoising UNet wrapper.

    Parameters
    ----------
    in_channels : int
        Number of input channels.  3 for single-image SS; 6 for bi-temporal CD.
    out_channels : int
        Number of output channels (number of classes, typically 2).
    img_scale : int
        Spatial size of the input/output images.  The internal UNet always
        operates at 256×256; inputs at other scales are up/down-sampled
        transparently.
    pretrained : str or None
        Key from ``_WEIGHT_REGISTRY`` (e.g. ``"aid_10k"``) or a
        raw HuggingFace repo-id string.  Pass ``None`` to start from random
        weights.
    freeze_unet : bool
        Freeze UNet backbone weights (useful for linear-probing experiments).
    use_timestep : bool
        Whether to pass timestep embeddings to the UNet.  Should be ``True``
        for the Noise2Map formulation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_scale: int = 256,
        pretrained: str | None = "aid_10k",
        freeze_unet: bool = False,
        use_timestep: bool = True,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        # 1. Load backbone
        # ------------------------------------------------------------------ #
        entry = _WEIGHT_REGISTRY.get(pretrained, (pretrained, None))
        repo_id, subfolder = entry if isinstance(entry, tuple) else (entry, None)


        if repo_id is not None:
            self.unet = UNet2DModelFlex.from_pretrained(
                repo_id,
                subfolder=subfolder,
                use_timestep=use_timestep,
            )
        else:
            # Random initialisation — useful for ablation studies
            self.unet = UNet2DModelFlex(use_timestep=use_timestep)
            logger.info("Initialised UNet backbone with random weights")

        if freeze_unet:
            for param in self.unet.parameters():
                param.requires_grad = False
            logger.info("UNet backbone frozen")

        # ------------------------------------------------------------------ #
        # 2. Channel-adapting projections
        # ------------------------------------------------------------------ #
        # in_channels -> 3 (UNet expects 3-channel input)
        self.input_conv = nn.Conv2d(in_channels, 3, kernel_size=3, stride=1, padding=1)
        # 3 -> out_channels
        self.output_conv = nn.Conv2d(3, out_channels, kernel_size=3, stride=1, padding=1)

        # ------------------------------------------------------------------ #
        # 3. Optional spatial re-sampling (when img_scale != 256)
        # ------------------------------------------------------------------ #
        if img_scale != 256:
            scale_up = 256 // img_scale
            scale_down = img_scale / 256

            self.input_upscaling: nn.Module | None = nn.Sequential(
                nn.Upsample(scale_factor=scale_up, mode="bilinear", align_corners=True),
                nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1),
            )
            self.output_downscaling: nn.Module | None = nn.Sequential(
                nn.Upsample(scale_factor=scale_down, mode="bilinear", align_corners=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.input_upscaling = None
            self.output_downscaling = None

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, in_channels, H, W)``.
        timesteps : torch.Tensor
            1-D integer tensor of shape ``(B,)`` with diffusion timestep indices.

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``(B, out_channels, H, W)``.
            Use ``torch.argmax(output, dim=1)`` for predictions.
        """
        x = self.input_conv(x)

        if self.input_upscaling is not None:
            x = self.input_upscaling(x)

        x = self.unet(x, timesteps).sample

        x = self.output_conv(x)

        if self.output_downscaling is not None:
            x = self.output_downscaling(x)

        return x  # raw logits — no sigmoid/softmax here

    # ---------------------------------------------------------------------- #
    # Utilities
    # ---------------------------------------------------------------------- #

    def print_trainable_parameters(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"Trainable: {trainable:,} / Total: {total:,} "
            f"({100 * trainable / total:.2f}%)"
        )

    @torch.no_grad()
    def get_backbone_features(
        self,
        x: torch.Tensor,
        layers_to_hook: tuple[str, ...] = ("down_blocks.0", "down_blocks.1", "mid_block"),
    ) -> torch.Tensor:
        """Extract and pool intermediate UNet features.

        Returns a tensor of shape ``(B, D)`` where D is the sum of
        channel dimensions of the hooked layers after global average pooling.
        """
        x = self.input_conv(x)
        if self.input_upscaling is not None:
            x = self.input_upscaling(x)

        features: dict[str, torch.Tensor] = {}
        hooks = []

        def _make_hook(name: str):
            def _hook(module, inp, out):
                tensor = out[0] if isinstance(out, tuple) else out
                if isinstance(tensor, torch.Tensor):
                    features[name] = tensor
            return _hook

        for name, module in self.unet.named_modules():
            if any(layer in name for layer in layers_to_hook):
                hooks.append(module.register_forward_hook(_make_hook(name)))

        dummy_t = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        self.unet(x, dummy_t)

        for h in hooks:
            h.remove()

        pooled = []
        for name in sorted(features):
            feat = features[name]
            if feat.ndim == 4:
                pooled.append(feat.mean(dim=(2, 3)))
            elif feat.ndim == 3:
                pooled.append(feat.mean(dim=2))
            elif feat.ndim == 2:
                pooled.append(feat)

        return torch.cat(pooled, dim=1)
