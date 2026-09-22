# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""Image normalization and patch reshaping helpers."""

import torch
import numpy as np
from PIL import ImageOps

IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]


def normalize_rgb(img):
    """
    Args:
        - img: np.array - (H,W,3) - np.uint8 - 0/255
    Return:
        - img: np.array - (3,H,W) - np.float - -3/3
    """
    img = img.astype(np.float32) / 255.
    img = np.transpose(img, (2,0,1))
    img = (img - np.asarray(IMG_NORM_MEAN).reshape(3,1,1)) / np.asarray(IMG_NORM_STD).reshape(3,1,1)
    return img.astype(np.float32)

def denormalize_rgb(img):
    """
    Args:
        - img: np.array - (3,H,W) - np.float - -3/3
    Return:
        - img: np.array - (H,W,3) - np.uint8 - 0/255
    """
    img = (img * np.asarray(IMG_NORM_STD).reshape(3,1,1)) + np.asarray(IMG_NORM_MEAN).reshape(3,1,1)
    img = np.transpose(img, (1,2,0)) * 255.
    return img.astype(np.uint8)

def unpatch(data, patch_size=14, img_size=224):
    """Reshape patch tokens back into a spatial feature map.

    Args:
        data: Tensor of shape (B, N, HWC) where N is the number of patches and HWC
            is patch_size**2 * C. If data has shape (B, N), it is expanded to
            (B, N, patch_size**2) by repeating along the last axis.
        patch_size: Side length of each square patch in pixels.
        img_size: Spatial grid size as an int (h=w=img_size) or a (h, w) tuple
            giving the number of patches per spatial dimension.

    Returns:
        Tensor of shape (B, C, h*patch_size, w*patch_size).
    """
    if len(data.shape) == 2:
        data = data[:,:,None].repeat([1,1,patch_size**2])

    if isinstance(img_size,int):
        h,w = img_size, img_size
    else:
        h,w = img_size

    B,N,HWC = data.shape
    HW = patch_size**2
    c = int(HWC / HW)
    p = q = int(HW**.5)
    data = data.reshape([B,h,w,p,q,c])
    data = torch.einsum('nhwpqc->nchpwq', data)
    return data.reshape([B,c,h,w])


def nonsquare_and_resize_image_without_K(img_pil, img_size, patch_size):
    """Resize a PIL image to fit within img_size x img_size and pad to the next multiple of patch_size.

    The image is first scaled with ImageOps.contain so that both dimensions are at
    most img_size (preserving aspect ratio), then zero-padded with ImageOps.pad to
    the next multiple of patch_size.

    Args:
        img_pil: Input PIL image.
        img_size: Maximum size for each dimension in pixels.
        patch_size: The output dimensions will be multiples of this value.

    Returns:
        Resized and padded PIL image.
    """
    img_resized_pil = ImageOps.contain(img_pil, (img_size, img_size))
    resized_width, resized_height = img_resized_pil.size
    new_width = int(np.ceil(resized_width / patch_size)) * patch_size
    new_height = int(np.ceil(resized_height / patch_size)) * patch_size
    img_resized_pil = ImageOps.pad(img_resized_pil, size=(new_width, new_height))  # zero-padding

    return img_resized_pil