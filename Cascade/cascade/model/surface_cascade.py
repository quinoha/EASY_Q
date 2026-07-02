import math

import torch
import torch.nn as nn

from cascade.model.bottleneck import BottleneckBlock3d
from cascade.model.embedding import SyndromeEmbedding
from cascade.model.readout import Readout


class SurfaceCascade(nn.Module):
    """Cascade backbone for surface-code memory-experiment decoding.

    Wires: SyndromeEmbedding -> `depth` x BottleneckBlock3d -> Readout.

    `depth` (L) is an explicit, required argument rather than derived from
    `distance` (d): the paper states the qualitative guidance "L ~ d" (so the
    stacked 3x3x3 receptive field spans the full code distance) but does not
    pin an exact proportionality constant, so silently guessing one here would
    hide a real modeling choice from the caller.

    MuP (Maximal Update Parameterization, used by the paper for width-stable
    training hyperparameters across (H, L) configs) is intentionally NOT
    implemented here: it rescales initialization and per-layer learning rates,
    which is a training-time concern, not part of this forward-pass
    architecture. A future training module should read `(hidden_dim, depth)`
    off this class to build its MuP base-shapes config.
    """

    def __init__(
        self,
        distance: int,
        rounds: int,
        hidden_dim: int,
        depth: int,
        data_qubit_mask: torch.Tensor,
        logical_masks: torch.Tensor,
        bottleneck_ratio: int = 4,
        mlp_hidden_dim: int | None = None,
    ):
        super().__init__()
        grid = distance + 1
        if data_qubit_mask.shape != (grid, grid):
            raise ValueError(
                f"data_qubit_mask shape {tuple(data_qubit_mask.shape)} does not match "
                f"expected (d+1, d+1) = ({grid}, {grid}) for distance={distance}"
            )

        self.distance = distance
        self.rounds = rounds
        self.hidden_dim = hidden_dim
        self.depth = depth

        self.embedding = SyndromeEmbedding(hidden_dim)

        residual_scale = 1.0 / math.sqrt(2 * depth)
        self.blocks = nn.ModuleList(
            [
                BottleneckBlock3d(hidden_dim, bottleneck_ratio, residual_scale)
                for _ in range(depth)
            ]
        )

        self.readout = Readout(
            hidden_dim, rounds, data_qubit_mask, logical_masks, mlp_hidden_dim
        )

    def forward(self, syndrome_idx: torch.Tensor) -> torch.Tensor:
        # syndrome_idx: LongTensor (B, T, G, G), T == self.rounds, values in {0,1,2}
        if syndrome_idx.shape[1] != self.rounds:
            raise ValueError(
                f"expected {self.rounds} rounds (dim 1), got {syndrome_idx.shape[1]}; "
                f"Readout's scatter conv has a time kernel sized for this model's `rounds`"
            )
        x = self.embedding(syndrome_idx)
        for block in self.blocks:
            x = block(x)
        return self.readout(x)
