# Multi-HMR 2
# Copyright (c) 2026-present NAVER Corp.

"""Human Perception Head: cross-attention transformer used by the decoder."""

import torch
from torch import nn
from einops import rearrange

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        """Wrap a module with a LayerNorm applied to its input.

        Args:
            dim: Feature dimension passed to nn.LayerNorm.
            fn: The module to wrap.
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        """Apply layer normalization to x then pass the result to self.fn.

        Args:
            x: Input tensor.
            **kwargs: Additional keyword arguments forwarded to self.fn.

        Returns:
            Output of self.fn applied to the normalized input.
        """
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        """Build a two-layer MLP with GELU activation and dropout.

        Args:
            dim: Input and output feature dimension.
            hidden_dim: Hidden layer size.
            dropout: Dropout probability applied after each linear layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Apply the two-layer MLP to x.

        Args:
            x: Input tensor of shape (..., dim).

        Returns:
            Tensor of shape (..., dim).
        """
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        """Build a multi-head self-attention module.

        Args:
            dim: Input and output feature dimension.
            heads: Number of attention heads.
            dim_head: Dimension per attention head.
            dropout: Dropout probability applied to attention weights and the
                output projection.
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        """Apply multi-head self-attention to x.

        Projects x to queries, keys, and values, computes scaled dot-product
        attention, and projects the output back to dim.

        Args:
            x: Input tensor of shape (b, n, dim).

        Returns:
            Tensor of shape (b, n, dim).
        """
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, dim, context_dim, heads, dim_head, dropout):
        """Build a multi-head cross-attention module.

        Args:
            dim: Query feature dimension and output dimension.
            context_dim: Key/value feature dimension. If None, defaults to dim.
            heads: Number of attention heads.
            dim_head: Dimension per attention head.
            dropout: Dropout probability applied to attention weights and the
                output projection.
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        context_dim = context_dim if context_dim is not None else dim
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, context):
        """Apply multi-head cross-attention.

        Keys and values are derived from context; queries are derived from x.

        Args:
            x: Query input tensor of shape (b, n, dim).
            context: Context tensor of shape (b, m, context_dim).

        Returns:
            Tensor of shape (b, n, dim).
        """
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q = self.to_q(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), [q, k, v])

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)

class TransformerCrossAttn(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout, context_dim):
        """Build a stack of (self-attention, cross-attention, feed-forward) blocks.

        Each block applies pre-norm self-attention, pre-norm cross-attention, and a
        pre-norm feed-forward layer, all with residual connections.

        Args:
            dim: Feature dimension for queries.
            depth: Number of stacked blocks.
            heads: Number of attention heads.
            dim_head: Dimension per head.
            mlp_dim: Hidden dimension of the feed-forward layer.
            dropout: Dropout probability.
            context_dim: Feature dimension of the context (keys/values).
        """
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            sa = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ca = CrossAttention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            ff = FeedForward(dim, mlp_dim, dropout=dropout)
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, sa),
                        PreNorm(dim, ca),
                        PreNorm(dim, ff),
                    ]
                )
            )

    def forward(self, x, context):
        """Apply all (self-attention, cross-attention, feed-forward) blocks in sequence.

        Args:
            x: Query tensor of shape (b, n, dim).
            context: Context tensor of shape (b, m, context_dim).

        Returns:
            Updated query tensor of shape (b, n, dim).
        """
        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x) + x
            x = cross_attn(x, context=context) + x
            x = ff(x) + x

        return x

class HPH(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        """Build the Human Perception Head transformer.

        Args:
            dim: Feature dimension (used for both queries and context).
            depth: Number of cross-attention blocks.
            heads: Number of attention heads.
            dim_head: Dimension per head.
            mlp_dim: Hidden dimension of the feed-forward layer.
            dropout: Dropout probability.
        """
        super().__init__()

        self.transformer = TransformerCrossAttn(dim, depth, heads, dim_head, mlp_dim, dropout, context_dim=dim)
        self.dim = dim

    def forward(self, x, context):
        """Run the cross-attention transformer.

        Args:
            x: Query tensor of shape (b, n, dim).
            context: Context tensor of shape (b, m, dim).

        Returns:
            Updated query tensor of shape (b, n, dim).
        """
        return self.transformer(x, context=context)

    
