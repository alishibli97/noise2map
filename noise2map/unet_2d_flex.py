"""
UNet2DModelFlex — a flexible variant of the standard diffusion UNet2DModel
that allows timestep conditioning to be toggled off at construction time.

This is a lightly modified copy of diffusers' UNet2DModel that adds a single
``use_timestep`` flag.  All other behaviour is identical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.models.embeddings import GaussianFourierProjection, TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_2d_blocks import UNetMidBlock2D, get_down_block, get_up_block


@dataclass
class UNet2DOutput(BaseOutput):
    """Output type for UNet2DModelFlex."""
    sample: torch.Tensor


class UNet2DModelFlex(ModelMixin, ConfigMixin):
    """
    A 2D UNet that optionally accepts timestep conditioning.

    Identical to ``diffusers.UNet2DModel`` except for the added
    ``use_timestep`` constructor argument that disables the time-embedding
    branch entirely when set to ``False``.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["norm"]

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[Union[int, Tuple[int, int]]] = None,
        in_channels: int = 3,
        out_channels: int = 3,
        center_input_sample: bool = False,
        time_embedding_type: str = "positional",
        time_embedding_dim: Optional[int] = None,
        freq_shift: int = 0,
        flip_sin_to_cos: bool = True,
        down_block_types: Tuple[str, ...] = (
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        mid_block_type: Optional[str] = "UNetMidBlock2D",
        up_block_types: Tuple[str, ...] = (
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels: Tuple[int, ...] = (224, 448, 672, 896),
        layers_per_block: int = 2,
        mid_block_scale_factor: float = 1,
        downsample_padding: int = 1,
        downsample_type: str = "conv",
        upsample_type: str = "conv",
        dropout: float = 0.0,
        act_fn: str = "silu",
        attention_head_dim: Optional[int] = 8,
        norm_num_groups: int = 32,
        attn_norm_num_groups: Optional[int] = None,
        norm_eps: float = 1e-5,
        resnet_time_scale_shift: str = "default",
        add_attention: bool = True,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        num_train_timesteps: Optional[int] = None,
        use_timestep: bool = True,
    ):
        super().__init__()

        self.sample_size = sample_size
        self.use_timestep = use_timestep
        time_embed_dim = time_embedding_dim or block_out_channels[0] * 4

        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"`down_block_types` and `up_block_types` must have the same length. "
                f"Got {down_block_types} and {up_block_types}."
            )
        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"`block_out_channels` and `down_block_types` must have the same length. "
                f"Got {block_out_channels} and {down_block_types}."
            )

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1)

        # ---- Time embedding (optional) ----
        if self.use_timestep:
            if time_embedding_type == "fourier":
                self.time_proj = GaussianFourierProjection(
                    embedding_size=block_out_channels[0], scale=16
                )
                timestep_input_dim = 2 * block_out_channels[0]
            elif time_embedding_type == "positional":
                self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
                timestep_input_dim = block_out_channels[0]
            elif time_embedding_type == "learned":
                if num_train_timesteps is None:
                    raise ValueError(
                        "`num_train_timesteps` must be set when `time_embedding_type='learned'`."
                    )
                self.time_proj = nn.Embedding(num_train_timesteps, block_out_channels[0])
                timestep_input_dim = block_out_channels[0]
            else:
                raise ValueError(f"Unknown `time_embedding_type`: {time_embedding_type}")

            self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
            temb_channels = time_embed_dim
        else:
            self.time_proj = None
            self.time_embedding = None
            temb_channels = None

        # ---- Class embedding (optional) ----
        if not self.use_timestep and class_embed_type == "timestep":
            raise ValueError("`class_embed_type='timestep'` requires `use_timestep=True`.")

        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(
                timestep_input_dim if self.use_timestep else time_embed_dim, time_embed_dim
            )
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        # ---- Down / Mid / Up blocks ----
        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            self.down_blocks.append(
                get_down_block(
                    down_block_type,
                    num_layers=layers_per_block,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=temb_channels,
                    add_downsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    attention_head_dim=(
                        attention_head_dim if attention_head_dim is not None else output_channel
                    ),
                    downsample_padding=downsample_padding,
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    downsample_type=downsample_type,
                    dropout=dropout,
                )
            )

        if mid_block_type is not None:
            self.mid_block = UNetMidBlock2D(
                in_channels=block_out_channels[-1],
                temb_channels=temb_channels,
                dropout=dropout,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                attention_head_dim=(
                    attention_head_dim if attention_head_dim is not None else block_out_channels[-1]
                ),
                resnet_groups=norm_num_groups,
                attn_groups=attn_norm_num_groups,
                add_attention=add_attention,
            )

        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[
                min(i + 1, len(block_out_channels) - 1)
            ]
            is_final_block = i == len(block_out_channels) - 1

            self.up_blocks.append(
                get_up_block(
                    up_block_type,
                    num_layers=layers_per_block + 1,
                    in_channels=input_channel,
                    out_channels=output_channel,
                    prev_output_channel=prev_output_channel,
                    temb_channels=temb_channels,
                    add_upsample=not is_final_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    attention_head_dim=(
                        attention_head_dim if attention_head_dim is not None else output_channel
                    ),
                    resnet_time_scale_shift=resnet_time_scale_shift,
                    upsample_type=upsample_type,
                    dropout=dropout,
                )
            )

        num_groups_out = norm_num_groups if norm_num_groups is not None else min(
            block_out_channels[0] // 4, 32
        )
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=num_groups_out, eps=norm_eps
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DOutput, Tuple]:
        """Forward pass.  If ``use_timestep=False`` the ``timestep`` argument
        is accepted but ignored."""

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # Time embedding
        if self.use_timestep:
            timesteps = timestep
            if not torch.is_tensor(timesteps):
                timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
            elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(sample.device)

            timesteps = timesteps * torch.ones(
                sample.shape[0], dtype=timesteps.dtype, device=timesteps.device
            )
            t_emb = self.time_proj(timesteps).to(dtype=self.dtype)
            emb = self.time_embedding(t_emb)
        else:
            emb = None

        # Class conditioning
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "`class_labels` must be provided when using class conditioning."
                )
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = class_emb if emb is None else emb + class_emb
        elif self.class_embedding is None and class_labels is not None:
            raise ValueError(
                "`class_embedding` must be initialised to use class conditioning."
            )

        # Encoder
        skip_sample = sample
        sample = self.conv_in(sample)
        down_block_res_samples = (sample,)

        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        # Bottleneck
        if self.mid_block is not None:
            sample = self.mid_block(sample, emb)

        # Decoder
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[:-len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)

        # Output projection
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if skip_sample is not None:
            sample += skip_sample

        if self.use_timestep and self.config.time_embedding_type == "fourier":
            timesteps = timesteps.reshape((sample.shape[0], *([1] * len(sample.shape[1:]))))
            sample = sample / timesteps

        if not return_dict:
            return (sample,)
        return UNet2DOutput(sample=sample)
