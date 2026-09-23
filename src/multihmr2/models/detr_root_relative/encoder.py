# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""ViT-based image encoder with horizontal field-of-view prediction."""

import torch
from torch import nn
from ...utils import unpatch
import math
from dataclasses import dataclass
from enum import Enum


class BackboneType(Enum):
    DINO_V3 = 'dinov3_vitl16'


@dataclass
class EncoderConfig:
    backbone: BackboneType = BackboneType.DINO_V3
    # The full checkpoint already contains the (fine-tuned) backbone weights, so
    # there is no need to download/load the DINOv3 pretrained weights -- they
    # would just be overwritten. Set True only when training from scratch or when
    # re-saving a complete checkpoint (see scripts/resave_full_checkpoint.py).
    pretrained: bool = False


@dataclass
class EncoderOutput:
    K: torch.Tensor  # [bs,3,3]
    feat: torch.Tensor  # [bs,sqrt(np),sqrt(np),emb]


class Encoder(nn.Module):
    """
    Image encoding with a ViT followed by patch-level detection and camera instrinsics estimation.
    """

    def __init__(self, config: EncoderConfig):
        """Build the image encoder.

        Loads a DINOv3 ViT backbone (optionally pretrained) and adds a small MLP
        that predicts the horizontal field-of-view angle from the CLS token.

        Args:
            config: Encoder configuration specifying the backbone type and whether
                to load pretrained weights.
        """
        super().__init__()
        match config.backbone:
            case BackboneType.DINO_V3:
                repo = 'facebookresearch/dinov3'
            case _:
                raise NotImplementedError(f"Backbone {config.backbone} not implemented")

        self.name = config.backbone.value
        self.backbone = torch.hub.load(repo, self.name, pretrained=config.pretrained)

        self.patch_size = self.backbone.patch_size
        self.embed_dim = self.backbone.embed_dim

        # FOV
        self.mlp_fov_unique = nn.Sequential(*[nn.Linear(self.embed_dim, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim,1)]) # between 0 and np.pi
        fov_max = torch.tensor([math.pi])
        self.register_buffer("fov_max", fov_max)

    def forward(self, x):
        """
        Encode a RGB image using a ViT-backbone
        Args:
            - x: torch.Tensor of shape [bs,3,h,w]
        Return:
            - EncoderOutput: a dataclass containing the camera intrinsics and the patch-level features
        """
        assert len(x.shape) == 4
        h,w = x.shape[-2:]

        # Encode RGB image using a ViT
        y, cls = self.backbone.get_intermediate_layers(x, return_class_token=True)[0] # [bs,np,emb]
        y = unpatch(y, patch_size=1, img_size=(h//self.backbone.patch_size,w//self.backbone.patch_size)) # [bs,emb,h,w]
        y = y.permute(0,2,3,1) # [bs,h,w,emb]

        # Horizontal field-of-view prediction
        fov = self.fov_max * torch.sigmoid(self.mlp_fov_unique(cls)) # range [0,fov_max]
        focal_length = (w / 2) / torch.tan(fov / 2) # assuming horizontal focal length = vertical focal length

        # Camera intrinsics 3x3 matrix - becarful 1st dim is for width and 2nd for height (because of the convention used in the perspective projection function)
        K = torch.eye(3).float().to(cls.device).reshape(1,3,3).repeat(cls.shape[0], 1, 1)
        K[:,[0,1],[0,1]] = focal_length[:,[0]] # for simplificty we assume the focal lenght is the same for w and h - but fov can be different depending on the image resolution (i.e. if not squared) - we care about the horizontal fov here
        K[:,0,-1] = w / 2. # principal point of the width
        K[:,1,-1] = h / 2. # principal point of the height

        return EncoderOutput(
            K=K,  # [bs,3,3]
            feat=y,  # [bs,sqrt(np),sqrt(np),emb]
        )
