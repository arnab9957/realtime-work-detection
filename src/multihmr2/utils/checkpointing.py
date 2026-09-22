# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""Checkpoint loading helpers."""

import os
import torch
import logging

logger = logging.getLogger(__name__)


def get_best_checkpoint(pretrained):
    """Load a pretrained checkpoint file."""
    if pretrained is not None:
        assert os.path.isfile(pretrained), f"Wrong path to pretrained: {pretrained}"
        return torch.load(pretrained, weights_only=False), pretrained

    return None, None


def load_from_checkpoint(model, checkpoint, ckpt_path):
    """Load model weights from a checkpoint (parameters and buffers).

    Restores every checkpoint entry whose key exists in the model (parameters and
    registered buffers alike), so the checkpoint fully determines the weights --
    including the backbone, which can therefore be built with ``pretrained=False``.
    The state dict is loaded with strict=False to allow partial loading.

    Args:
        model: The model to load weights into.
        checkpoint: Dictionary loaded from a .pt file, expected to have a
            "model_state_dict" key. If None, this function is a no-op.
        ckpt_path: Path string used only for logging.
    """
    if checkpoint is not None:
        logger.debug(f"Loading weights from {ckpt_path} ...")

        # keep parameters *and* buffers (model.state_dict() includes both)
        state_dict = checkpoint["model_state_dict"]
        model_keys = set(model.state_dict().keys())
        state_dict = {k: v for k, v in state_dict.items() if k in model_keys}

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        logger.debug(
            f"Ckpt loaded with missing keys: {missing} and unexpected keys: {unexpected}"
        )
