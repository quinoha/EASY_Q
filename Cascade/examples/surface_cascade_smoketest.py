"""Printed shape-trace walkthrough for SurfaceCascade.

Not executed in the environment this was written in (no Python interpreter
available there). Run with:
    pip install -r requirements.txt
    python examples/surface_cascade_smoketest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from cascade import SurfaceCascade
from cascade.geometry.surface_code import (
    checkerboard_ancilla_mask,
    synthetic_data_qubit_mask,
    synthetic_logical_masks,
)
from cascade.model.embedding import syndrome_indices_from_detections


def main():
    torch.manual_seed(0)
    distance, rounds, hidden_dim, depth, batch = 5, 5, 64, 5, 2
    grid = distance + 1

    ancilla_mask = checkerboard_ancilla_mask(distance)
    data_qubit_mask = synthetic_data_qubit_mask(distance, ancilla_mask)
    logical_masks = synthetic_logical_masks(distance, data_qubit_mask)
    print(f"grid={grid}x{grid}  ancilla sites={ancilla_mask.sum().item()}  "
          f"data-qubit sites={data_qubit_mask.sum().item()}")

    model = SurfaceCascade(
        distance=distance,
        rounds=rounds,
        hidden_dim=hidden_dim,
        depth=depth,
        data_qubit_mask=data_qubit_mask,
        logical_masks=logical_masks,
    )

    detections = torch.rand(batch, rounds, grid, grid) > 0.5
    syndrome_idx = syndrome_indices_from_detections(detections, ancilla_mask)
    print(f"syndrome_idx:            {tuple(syndrome_idx.shape)}")

    x = model.embedding(syndrome_idx)
    print(f"after embedding:         {tuple(x.shape)}")

    for i, block in enumerate(model.blocks):
        x = block(x)
        print(f"after bottleneck block {i}: {tuple(x.shape)}")

    logits = model.readout(x)
    print(f"logits:                  {tuple(logits.shape)}")

    targets = torch.randint(0, 2, logits.shape).float()
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    loss.backward()
    n_missing = sum(1 for p in model.parameters() if p.grad is None)
    print(f"loss={loss.item():.4f}  params_without_grad={n_missing}")


if __name__ == "__main__":
    main()
