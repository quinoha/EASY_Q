"""Shape + gradient-flow self-test for the surface-code backbone.

Torch only -- no stim dependency. This does NOT validate physical correctness
of the placeholder geometry (see cascade/geometry/surface_code.py); it only
proves the model is wired together correctly: shapes match end to end, and
every parameter receives a gradient from a single training step.

Not executed in the environment this was written in (no Python interpreter
available there). Run with:
    pip install -r requirements.txt
    python -m pytest tests/test_shapes.py -v
"""

import torch
import torch.nn.functional as F

from cascade import SurfaceCascade
from cascade.geometry.surface_code import (
    checkerboard_ancilla_mask,
    synthetic_data_qubit_mask,
    synthetic_logical_masks,
)
from cascade.model.embedding import syndrome_indices_from_detections


def test_forward_shape():
    torch.manual_seed(0)
    distance, rounds, batch = 5, 5, 2
    grid = distance + 1

    ancilla_mask = checkerboard_ancilla_mask(distance)
    data_qubit_mask = synthetic_data_qubit_mask(distance, ancilla_mask)
    logical_masks = synthetic_logical_masks(distance, data_qubit_mask)

    model = SurfaceCascade(
        distance=distance,
        rounds=rounds,
        hidden_dim=16,
        depth=3,
        data_qubit_mask=data_qubit_mask,
        logical_masks=logical_masks,
    )

    detections = torch.rand(batch, rounds, grid, grid) > 0.5
    syndrome_idx = syndrome_indices_from_detections(detections, ancilla_mask)

    logits = model(syndrome_idx)
    assert logits.shape == (batch, logical_masks.shape[0])
    assert torch.isfinite(logits).all()


def test_backward_all_params_receive_gradients():
    torch.manual_seed(0)
    distance, rounds, batch = 5, 5, 2
    grid = distance + 1

    ancilla_mask = checkerboard_ancilla_mask(distance)
    data_qubit_mask = synthetic_data_qubit_mask(distance, ancilla_mask)
    logical_masks = synthetic_logical_masks(distance, data_qubit_mask)

    model = SurfaceCascade(
        distance=distance,
        rounds=rounds,
        hidden_dim=16,
        depth=3,
        data_qubit_mask=data_qubit_mask,
        logical_masks=logical_masks,
    )
    model.train()

    detections = torch.rand(batch, rounds, grid, grid) > 0.5
    syndrome_idx = syndrome_indices_from_detections(detections, ancilla_mask)
    targets = torch.randint(0, 2, (batch, logical_masks.shape[0])).float()

    logits = model(syndrome_idx)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    assert not torch.isnan(loss)

    loss.backward()

    missing_grad = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing_grad, f"parameters with no gradient: {missing_grad}"


def test_eval_mode_runs():
    torch.manual_seed(0)
    distance, rounds, batch = 5, 5, 1
    grid = distance + 1

    ancilla_mask = checkerboard_ancilla_mask(distance)
    data_qubit_mask = synthetic_data_qubit_mask(distance, ancilla_mask)
    logical_masks = synthetic_logical_masks(distance, data_qubit_mask)

    model = SurfaceCascade(
        distance=distance,
        rounds=rounds,
        hidden_dim=16,
        depth=3,
        data_qubit_mask=data_qubit_mask,
        logical_masks=logical_masks,
    )
    model.eval()

    detections = torch.rand(batch, rounds, grid, grid) > 0.5
    syndrome_idx = syndrome_indices_from_detections(detections, ancilla_mask)
    with torch.no_grad():
        logits = model(syndrome_idx)
    assert logits.shape == (batch, logical_masks.shape[0])
