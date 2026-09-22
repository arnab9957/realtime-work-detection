# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""Shared utilities: camera projection, image preprocessing, rendering, logging, checkpointing."""

from . import logger
from .camera import perspective_projection, inverse_perspective_projection
from .image import normalize_rgb, unpatch, denormalize_rgb, nonsquare_and_resize_image_without_K
from .render import render_meshes
from .tensor_manip import rotation_to_homogeneous
from .color import demo_color
from .checkpointing import get_best_checkpoint, load_from_checkpoint