from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict,Iterable

from transformers import AutoModel

class EncodingWrapper(nn.Module):

    def __init__(
        self,
        encoder: Dict[str, AutoModel],
        state_latent_dim: int = 64,
        img_obs_keys: Iterable[str] = ("image",),
        state_obs_keys: Iterable[str] = ("state",),
        other_obs_keys: Iterable[str] = (),
        state_input_dim: int = 64,
    ):
        super().__init__()
        self.encoder = nn.ModuleDict(encoder)
        self.state_latent_dim = state_latent_dim
        self.img_obs_keys = list(img_obs_keys)
        self.state_obs_keys = list(state_obs_keys)
        self.other_obs_keys = list(other_obs_keys)

        self.state_dense = nn.Linear(
            in_features=state_input_dim, 
            out_features=self.state_latent_dim
        )
        nn.init.xavier_uniform_(self.state_dense.weight)
        self.state_layer_norm = nn.LayerNorm(self.state_latent_dim)

    def forward(
        self,
        obs_groups: Dict[str, torch.Tensor],
        train: bool = False,
        stop_gradient: bool = False,
        is_encoded: bool = False,
    ) -> torch.Tensor:
        # Encode all images
        encoded = []
        for image_obs_key in self.img_obs_keys:
            image = obs_groups[image_obs_key]

            if image.dim() == 4 and image.shape[-1] in [1, 3, 4]:  # NHWC格式
                image = image.permute(0, 3, 1, 2)

            # Encode image
            image_encoding = self.encoder[image_obs_key](image)

            # Extract the appropriate tensor from the model output
            if hasattr(image_encoding, 'last_hidden_state'):
                # For vision models, we typically want the pooled output or a specific representation
                if hasattr(image_encoding, 'pooler_output') and image_encoding.pooler_output is not None:
                    image_encoding = image_encoding.pooler_output
                else:
                    # If no pooler_output, use the first token (usually the CLS token for transformers)
                    image_encoding = image_encoding.last_hidden_state[:, 0]
            # For other model types that directly return tensors, keep as is
            elif not isinstance(image_encoding, torch.Tensor):
                # If it's not a tensor and not a known model output object, try to extract tensor
                if hasattr(image_encoding, 'pooler_output'):
                    image_encoding = image_encoding.pooler_output

            # Stop gradient if requested
            if stop_gradient:
                image_encoding = image_encoding.detach()

            encoded.append(image_encoding)

        # Concatenate all image encodings
        encoded = torch.cat(encoded, dim=-1)
        encoded = encoded.squeeze(-1).squeeze(-1)

        for state_obs_key in self.state_obs_keys:
            state_obs = obs_groups[state_obs_key]

            # Project stateception
            state_obs = self.state_dense(state_obs)
            state_obs = self.state_layer_norm(state_obs)
            state_obs = torch.tanh(state_obs)

            # Concatenate with image encodings
            encoded = torch.cat([encoded, state_obs], dim=-1)

        for other_obs_key in self.other_obs_keys:
            other_obs = obs_groups[other_obs_key]
            encoded = torch.cat([encoded, other_obs], dim=-1)

        return encoded


# import torch.nn.functional as F
# from typing import Dict, List, Optional, Tuple, Iterable


# # ---------------------------
# # ResNet Encoder Components
# # ---------------------------
# class ResNetBlock(nn.Module):
#     """ResNet block for encoder"""

#     def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
#         super().__init__()
#         self.conv1 = nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size=3,
#             stride=stride,
#             padding=1,
#             bias=False,
#         )
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(
#             out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
#         )
#         self.bn2 = nn.BatchNorm2d(out_channels)

#         self.downsample = None
#         if stride != 1 or in_channels != out_channels:
#             self.downsample = nn.Sequential(
#                 nn.Conv2d(
#                     in_channels, out_channels, kernel_size=1, stride=stride, bias=False
#                 ),
#                 nn.BatchNorm2d(out_channels),
#             )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         residual = x
#         if self.downsample is not None:
#             residual = self.downsample(residual)

#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += residual
#         out = F.relu(out)
#         return out


# class ResNetEncoder(nn.Module):
#     """ResNet encoder for images"""

#     def __init__(
#         self,
#         stage_sizes: Tuple[int, int, int, int],
#         block_cls: nn.Module,
#         pre_pooling: bool = True,
#     ):
#         super().__init__()
#         self.pre_pooling = pre_pooling

#         # Initial convolution
#         self.in_channels = 64
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)

#         # ResNet stages
#         self.stage1 = self._make_stage(block_cls, 64, stage_sizes[0], stride=1)
#         self.stage2 = self._make_stage(block_cls, 128, stage_sizes[1], stride=2)
#         self.stage3 = self._make_stage(block_cls, 256, stage_sizes[2], stride=2)
#         self.stage4 = self._make_stage(block_cls, 512, stage_sizes[3], stride=2)

#         # Freeze all parameters (as in original "frozen" config)
#         for param in self.parameters():
#             param.requires_grad = False

#     def _make_stage(
#         self, block_cls: nn.Module, out_channels: int, num_blocks: int, stride: int
#     ) -> nn.Sequential:
#         strides = [stride] + [1] * (num_blocks - 1)
#         blocks = []
#         for stride in strides:
#             blocks.append(block_cls(self.in_channels, out_channels, stride))
#             self.in_channels = out_channels
#         return nn.Sequential(*blocks)

#     def forward(self, x: torch.Tensor, train: bool = True) -> torch.Tensor:
#         # Input: B x H x W x C -> convert to B x C x H x W (PyTorch format)
#         if x.shape[-1] == 3:  # if channel last
#             x = x.permute(0, 3, 1, 2)

#         # ResNet forward pass
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.stage1(x)
#         x = self.stage2(x)
#         x = self.stage3(x)
#         x = self.stage4(x)

#         # Convert back to B x H x W x C for SpatialLearnedEmbeddings
#         x = x.permute(0, 2, 3, 1)
#         return x


# class SpatialLearnedEmbeddings(nn.Module):
#     """Spatial learned embeddings for image features"""

#     def __init__(
#         self,
#         height: int,
#         width: int,
#         channel: int,
#         num_features: int = 5,
#         kernel_init: Optional[nn.init.Initializer] = None,
#     ):
#         super().__init__()
#         self.height = height
#         self.width = width
#         self.channel = channel
#         self.num_features = num_features

#         # Initialize kernel
#         kernel_init = kernel_init or nn.init.lecun_normal_
#         self.kernel = nn.Parameter(
#             torch.empty(height, width, channel, num_features, dtype=torch.float32)
#         )
#         kernel_init(self.kernel)

#     def forward(self, features: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             features: B x H x W x C
#         Returns:
#             B x (C * num_features)
#         """
#         no_batch_dim = len(features.shape) < 4
#         if no_batch_dim:
#             features = features.unsqueeze(0)

#         batch_size = features.shape[0]

#         # Expand dimensions for multiplication: B x H x W x C x 1 * 1 x H x W x C x F
#         features_expanded = features.unsqueeze(-1)  # B x H x W x C x 1
#         kernel_expanded = self.kernel.unsqueeze(0)  # 1 x H x W x C x F

#         # Element-wise multiplication and sum over H, W
#         features = torch.sum(
#             features_expanded * kernel_expanded, dim=(1, 2)
#         )  # B x C x F
#         features = features.reshape(batch_size, -1)  # B x (C * F)

#         if no_batch_dim:
#             features = features.squeeze(0)

#         return features


# class PreTrainedResNetEncoder(nn.Module):
#     """Wrapper for pre-trained ResNet encoder with spatial embeddings"""

#     def __init__(
#         self,
#         num_spatial_blocks: int = 8,
#         img_bottleneck_dim: int = 256,
#         pretrained_encoder: Optional[nn.Module] = None,
#         name: str = "pretrained_encoder",
#     ):
#         super().__init__()
#         self.name = name
#         self.num_spatial_blocks = num_spatial_blocks
#         self.img_bottleneck_dim = img_bottleneck_dim

#         # Initialize pretrained encoder if not provided
#         if pretrained_encoder is None:
#             pretrained_encoder = ResNetEncoder(
#                 stage_sizes=(1, 1, 1, 1), block_cls=ResNetBlock, pre_pooling=True
#             )
#         self.pretrained_encoder = pretrained_encoder

#         # Get spatial dimensions from dummy input (assuming 224x224 input)
#         dummy_input = torch.randn(1, 224, 224, 3)
#         with torch.no_grad():
#             dummy_output = self.pretrained_encoder(dummy_input)
#             self.height, self.width, self.channel = dummy_output.shape[-3:]

#         # Spatial learned embeddings
#         self.spatial_embeddings = SpatialLearnedEmbeddings(
#             height=self.height,
#             width=self.width,
#             channel=self.channel,
#             num_features=num_spatial_blocks,
#         )

#         # Bottleneck layers
#         self.dropout = nn.Dropout(0.1)
#         self.dense = nn.Linear(self.channel * num_spatial_blocks, img_bottleneck_dim)
#         self.layer_norm = nn.LayerNorm(img_bottleneck_dim)

#     def forward(self, observations: torch.Tensor, train: bool = True) -> torch.Tensor:
#         """
#         Args:
#             observations: B x H x W x C (image observations)
#             train: whether in training mode
#         Returns:
#             B x img_bottleneck_dim
#         """
#         # Pre-trained encoder forward pass
#         x = self.pretrained_encoder(observations, train=train)

#         # Spatial embeddings
#         x = self.spatial_embeddings(x)

#         # Dropout (only in training)
#         x = self.dropout(x) if train else x

#         # Bottleneck
#         x = self.dense(x)
#         x = self.layer_norm(x)
#         x = torch.tanh(x)

#         return x
