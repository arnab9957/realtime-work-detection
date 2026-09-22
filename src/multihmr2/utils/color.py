# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""Color palette used to draw per-person overlays in the demo."""

import numpy as np

def hex_to_rgb(hex):
    """Convert a 6-character hex color string to a normalized RGB tuple.

    Args:
        hex: 6-character hexadecimal string (e.g. "FF9999"), without a leading '#'.

    Returns:
        Tuple of three floats in [0, 1] representing (R, G, B).
    """
    y = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
    return (y[0]/255,y[1]/255,y[2]/255)

# Define colors for the demo
demo_color = ['0047AB', # cobaltblue
        'FF9999', 'FF9933', '00CC66', '66B2FF', 'FF6666', 'FF3333', 'C0C0C0', '9933FF'] # rosé - orange - green - blue - red - grey - violet
demo_color = [ hex_to_rgb(x) for x in demo_color]

n_colors = 1000
_rng = np.random.default_rng(seed=0)
for i in range(n_colors):
        color_i = _rng.integers(0, 256, size=3)
        demo_color.append((color_i[0]/255, color_i[1]/255, color_i[2]/255))