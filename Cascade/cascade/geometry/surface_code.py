"""Placeholder surface-code geometry.

Everything in this module is a SYNTHETIC STAND-IN, not physically validated
surface-code combinatorics. Its only job is to produce shape-correct,
self-consistent (G, G) boolean masks so `SurfaceCascade` can be exercised
end-to-end (forward pass, backward pass, shapes) before a real stim-based data
pipeline exists.

In particular, `synthetic_data_qubit_mask` claims data qubits occupy the
complement of the ancilla checkerboard on the same (d+1, d+1) grid. That is
NOT geometrically correct at real code distances: a distance-d surface code
has d^2 data qubits but only (d+1)^2 - (checkerboard ancilla count) = 2d+2
leftover sites on that grid -- nowhere near enough room once d is large enough
to matter. Real data-qubit coordinates must come from stim (e.g.
`circuit.get_final_qubit_coordinates()` on a
`stim.Circuit.generated("surface_code_rotated_memory_z", ...)` circuit).

`cascade.model.readout.Readout` already converts whatever mask it's given into
flat indices internally, so replacing these functions with real stim-derived
geometry later requires no changes to the model's forward() code -- only to
how the mask/index tensors passed into the constructor are built.
"""

import torch


def checkerboard_ancilla_mask(distance: int) -> torch.Tensor:
    """Placeholder ancilla (check-site) mask: True at (i+j) even sites on the
    (d+1, d+1) grid. NOT validated against real surface-code boundary
    combinatorics.
    """
    if distance < 1:
        raise ValueError(f"distance must be >= 1, got {distance}")
    grid = distance + 1
    i = torch.arange(grid).unsqueeze(1)
    j = torch.arange(grid).unsqueeze(0)
    return ((i + j) % 2 == 0).expand(grid, grid).clone()


def synthetic_data_qubit_mask(distance: int, ancilla_mask: torch.Tensor) -> torch.Tensor:
    """Placeholder data-qubit mask: complement of `ancilla_mask` on the same
    (d+1, d+1) grid. Shape-compatible stand-in only -- see module docstring
    for why this does not hold at real code distances.
    """
    grid = distance + 1
    if ancilla_mask.shape != (grid, grid):
        raise ValueError(f"ancilla_mask shape {tuple(ancilla_mask.shape)} != ({grid}, {grid})")
    return ~ancilla_mask


def synthetic_logical_masks(distance: int, data_qubit_mask: torch.Tensor) -> torch.Tensor:
    """Placeholder logical-operator support: one middle row and one middle
    column of `data_qubit_mask`, standing in for the boundary-to-boundary
    XL/ZL supports of a real surface code. Returns (2, d+1, d+1) bool.
    """
    grid = distance + 1
    if data_qubit_mask.shape != (grid, grid):
        raise ValueError(f"data_qubit_mask shape {tuple(data_qubit_mask.shape)} != ({grid}, {grid})")

    row_mask = torch.zeros(grid, grid, dtype=torch.bool)
    row_mask[grid // 2, :] = True
    row_mask &= data_qubit_mask

    col_mask = torch.zeros(grid, grid, dtype=torch.bool)
    col_mask[:, grid // 2] = True
    col_mask &= data_qubit_mask

    if not row_mask.any() or not col_mask.any():
        raise ValueError(
            "synthetic logical mask has empty support for this distance -- "
            "the middle-row/column placeholder needs a larger distance"
        )
    return torch.stack([row_mask, col_mask], dim=0)
