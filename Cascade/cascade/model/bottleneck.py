import torch
import torch.nn as nn


def default_surface_code_conv(bottleneck_dim: int) -> nn.Module:
    """The surface-code "geometry-aware convolution": a standard, zero-padded
    3x3x3 Conv3d over the (time, row, col) spacetime lattice (Methods:
    "for surface codes, a standard 3D convolution over the spacetime lattice").
    """
    return nn.Conv3d(
        bottleneck_dim,
        bottleneck_dim,
        kernel_size=3,
        padding=1,
        padding_mode="zeros",
        bias=False,
    )


class BottleneckBlock3d(nn.Module):
    """One layer of the Cascade backbone (Extended Data Fig. 1):

        h -> BN -> SiLU -> Conv1x1(dim -> dim/ratio)          [reduce]
          -> BN -> SiLU -> message_passing(dim/ratio)         [code-specific conv]
          -> BN -> SiLU -> Conv1x1(dim/ratio -> dim)          [restore]
        out = h + residual_scale * (...)

    `message_passing` is the one seam that varies by code family (paper's own
    Extended Data Fig. 3 decomposes every architecture variant into two
    pointwise projections plus one spatial operation). It is accepted as an
    already-constructed nn.Module rather than hardcoded, defaulting to the
    surface code's 3x3x3 Conv3d -- this is not a speculative abstraction, it's
    naming the exact generality point the paper documents, at the cost of one
    constructor parameter.

    residual_scale is expected to be `1 / sqrt(2 * depth)`, computed once by
    the owning model and shared identically across all of its blocks, so this
    class stays trivially unit-testable in isolation.
    """

    def __init__(
        self,
        dim: int,
        bottleneck_ratio: int,
        residual_scale: float,
        message_passing: nn.Module | None = None,
    ):
        super().__init__()
        if dim % bottleneck_ratio != 0:
            raise ValueError(f"dim={dim} not divisible by bottleneck_ratio={bottleneck_ratio}")
        bdim = dim // bottleneck_ratio

        self.norm1 = nn.BatchNorm3d(dim)
        self.reduce = nn.Conv3d(dim, bdim, kernel_size=1, bias=False)

        self.norm2 = nn.BatchNorm3d(bdim)
        self.message_passing = message_passing or default_surface_code_conv(bdim)

        self.norm3 = nn.BatchNorm3d(bdim)
        self.restore = nn.Conv3d(bdim, dim, kernel_size=1)  # keeps bias: nothing normalizes it after

        self.act = nn.SiLU()
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, dim, T, G, G)
        h = self.reduce(self.act(self.norm1(x)))
        h = self.message_passing(self.act(self.norm2(h)))
        h = self.restore(self.act(self.norm3(h)))
        return x + self.residual_scale * h
