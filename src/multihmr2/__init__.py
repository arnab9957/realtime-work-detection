# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""Multi-HMR 2: multi-person 3D human mesh recovery from images and videos."""

from . import utils
from .api import (
    InferenceSession,
    init_hmr_session,
    infer_image,
    save_results_image,
    render_results_image,
    infer_video,
    save_results_video,
    render_results_video,
)
from .models.detr_root_relative.decoder import DecoderOutput, PersonOutput

__all__ = [
    "InferenceSession",
    "init_hmr_session",
    "infer_image",
    "save_results_image",
    "render_results_image",
    "infer_video",
    "save_results_video",
    "render_results_video",
    "DecoderOutput",
    "PersonOutput",
]
