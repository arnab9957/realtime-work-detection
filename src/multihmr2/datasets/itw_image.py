# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""Image loading and preprocessing for in-the-wild inputs."""

import torch
import numpy as np
from PIL import Image, ImageFile
import torchvision.transforms.v2 as T

from ..utils.image import IMG_NORM_MEAN, IMG_NORM_STD
from ..utils import normalize_rgb, nonsquare_and_resize_image_without_K

ImageFile.LOAD_TRUNCATED_IMAGES = True  # to avoid "OSError: image file is truncated"


def preprocess_image(img_path):
    """Load and preprocess an image for inference.

    Opens the image, resizes it so that the longest side is 768 pixels while
    preserving the aspect ratio, pads it to be divisible by 16, normalizes with
    ImageNet mean and std, and returns it as a float32 tensor.

    Args:
        img_path: Path to the input image file.

    Returns:
        Tensor of shape (1, C, H, W) and dtype float32.
    """
    img_pil = Image.open(img_path).convert("RGB")

    img_pil = nonsquare_and_resize_image_without_K(img_pil, 768, 16)

    # Go to numpy
    resize_img = np.asarray(img_pil)

    # Normalize and go to torch.
    resize_img = normalize_rgb(resize_img)
    x = torch.from_numpy(resize_img)

    return x.unsqueeze(0)


class ImagePreproc(T.Transform):
    """Preprocessing for HMR inference.

    The image is resized such that the longest side match the specified size,
    and then centered padded to be divisible by the patch size. Finally, the image
    is normalized with the same mean and std as the training data.

    Parameters:
        size (int): The target size for the longest side of the image. The
            aspect ratio is preserved.
        patch_size (int): The patch size used in the model. The image will be
            padded to be divisible by this patch size.

    Attributes:
        size (int): The target size for the longest side of the image.
        patch_size (int): The patch size used in the model.
        _tf (T.Compose): The composed transform for resizing and normalizing the image.
    """

    size: int
    patch_size: int
    _tf: T.Compose

    def __init__(self, size: int, patch_size: int):
        """Build the resize and normalize transform pipeline.

        Args:
            size: Target size for the longest side of the image (pixels).
            patch_size: The model patch size; the image will be padded to be
                divisible by this value.
        """
        super().__init__()
        self.size = size
        self.patch_size = patch_size

        self._tf = T.Compose(
            [
                T.ToImage(),
                T.Resize(None, T.InterpolationMode.BICUBIC, self.size),
                T.ToDtype(torch.float32, scale=True),
            ]
        )

        self._norm = T.Normalize(mean=IMG_NORM_MEAN, std=IMG_NORM_STD)

    def forward(self, img: np.ndarray) -> torch.tensor:
        """Resize, center-pad, and normalize the image.

        Applies self._tf (ToImage, bicubic resize to longest side = self.size,
        scale to float32), then zero-pads to the next multiple of patch_size on each
        side (split evenly left/right and top/bottom), then normalizes with ImageNet
        mean and std.

        Args:
            img: Input image as a numpy array of shape (H, W, C) and dtype uint8.

        Returns:
            Tensor of shape (C, H', W') and dtype float32, where H' and W' are
            multiples of patch_size.
        """
        img = self._tf(img)

        _, h, w = img.shape
        new_w = int(np.ceil(w / self.patch_size)) * self.patch_size
        new_h = int(np.ceil(h / self.patch_size)) * self.patch_size
        p_left, p_top = (new_w - w) // 2, (new_h - h) // 2
        p_right, p_bottom = new_w - w - p_left, new_h - h - p_top

        img = T.functional.pad(img, [p_left, p_top, p_right, p_bottom], fill=0)
        assert img.shape[-2:] == (new_h, new_w), (
            f"unexpected pad result {tuple(img.shape)}, expected (..., {new_h}, {new_w})"
        )
        return self._norm(img)
