# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""Multi-HMR top-level model: image encoder followed by a full-body decoder."""

from torch import nn
from .encoder import Encoder, EncoderConfig
from .decoder import Decoder, DecoderConfig, DecoderOutput
from dataclasses import dataclass, field


@dataclass
class DETRConfig:
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    decoder_config: DecoderConfig = field(default_factory=DecoderConfig)
    body_model: str = "anny"


class DETR_Root_Relative(nn.Module):
    """A ViT backbone followed by a "HPH" head (stack of cross attention layers with a fixed number of queries.)"""

    def __init__(self, config: DETRConfig):
        """Build the full Multi-HMR model: an image encoder followed by a decoder.

        Args:
            config: DETRConfig specifying encoder and decoder sub-configurations and
                the body model name (must be "anny").
        """
        super().__init__()
        assert config.body_model == "anny", "SMPLX version not implemented"

        # Encoder
        self.img_size = 768
        self.encoder = Encoder(config.encoder_config)
        assert self.img_size % self.encoder.patch_size == 0, "Invalid img size"
        self.patch_size = self.encoder.patch_size

        # Decoders
        self.full_body_decoder = Decoder(
            config.decoder_config,
            grid_size=self.img_size // self.patch_size,
            patch_size=self.patch_size,
            embed_dim=self.encoder.embed_dim,
        )

    def forward(
        self,
        x,
        conf_thresh: float = 0.4,
        dist_thresh_nms: float = 0.25,
        lowres: bool = False,
    ) -> list[DecoderOutput]:
        """Run the full Multi-HMR pipeline: encode the image, then decode.

        Args:
            x: Input image tensor of shape (bs, 3, H, W).
            conf_thresh: Confidence threshold for the decoder.
            dist_thresh_nms: 3D pelvis NMS distance threshold.
            lowres: If True, uses the low-resolution body model in the decoder.

        Returns:
            List of DecoderOutput, one per element in the batch.
        """
        z = self.encoder(x)
        return self.full_body_decoder(
            z,
            K=z.K,
            conf_thresh=conf_thresh,
            dist_thresh_nms=dist_thresh_nms,
            lowres=lowres,
        )
