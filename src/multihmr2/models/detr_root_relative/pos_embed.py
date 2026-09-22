# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""2D sinusoidal positional embeddings."""

import numpy as np

# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# https://github.com/naver/croco/blob/743ee71a2a9bf57cea6832a9064a70a0597fcfcb/models/pos_embed.py
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim]
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """Compute 2D sinusoidal positional embeddings from a precomputed coordinate grid.

    embed_dim must be even. The first half of each embedding encodes the H (row)
    coordinate and the second half encodes the W (column) coordinate.

    Args:
        embed_dim: Total embedding dimension (must be even).
        grid: Array of shape (2, 1, grid_size, grid_size) where grid[0] contains W
            coordinates and grid[1] contains H coordinates.

    Returns:
        Array of shape (grid_size*grid_size, embed_dim).
    """
    assert embed_dim % 2 == 0

    # grid[0] holds the W (x) coordinates, grid[1] the H (y) coordinates
    # (np.meshgrid(grid_w, grid_h) returns [X, Y]).
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_w, emb_h], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb