import torch
import torch.nn as nn


class Readout(nn.Module):
    """Aggregate + Classify stage (Fig. 1a): scatter check-node representations
    to data qubits, pool over each logical operator's support, predict a logit
    per logical observable.

    Two design choices worth calling out (paper under-specifies both):

    1. Time collapse is fused into the "scatter" convolution itself: a single
       Conv3d with kernel_size=(rounds, 3, 3) and padding=(0, 1, 1) is valid in
       time (collapses T -> 1) and same-padded in space. This is a *learned*
       per-round weighting rather than a hardcoded uniform time-average -- more
       expressive, at the cost of fixing this module (and therefore the whole
       model) to a specific `rounds` value at construction time. That matches
       the paper's own methodology of training a separate model per
       configuration, so it isn't treated as a real limitation.

    2. `data_qubit_mask` is accepted as a (G, G) boolean mask (matching the
       paper's literal same-grid picture) but immediately converted to flat
       indices internally. At real code distances, data qubits and ancillas
       cannot literally be disjoint masks on the same (d+1, d+1) grid (there
       are only 2d+2 leftover non-stabilizer sites for d^2 data qubits) -- but
       because forward() only ever consumes the derived flat indices / pooling
       matrix, swapping in real stim-derived qubit coordinates later (a
       differently-shaped index map, or a many-to-one incidence matrix instead
       of a 1:1 selection) requires no forward() changes.
    """

    def __init__(
        self,
        hidden_dim: int,
        rounds: int,
        data_qubit_mask: torch.Tensor,
        logical_masks: torch.Tensor,
        mlp_hidden_dim: int | None = None,
    ):
        super().__init__()
        grid = data_qubit_mask.shape[0]
        if data_qubit_mask.shape != (grid, grid):
            raise ValueError(f"data_qubit_mask must be square (G,G), got {tuple(data_qubit_mask.shape)}")
        if logical_masks.shape[1:] != (grid, grid):
            raise ValueError(
                f"logical_masks must be (K,G,G) matching data_qubit_mask's grid, "
                f"got {tuple(logical_masks.shape)} vs grid={grid}"
            )

        flat_data_mask = data_qubit_mask.reshape(-1)
        data_qubit_flat_indices = flat_data_mask.nonzero(as_tuple=True)[0]
        if data_qubit_flat_indices.numel() == 0:
            raise ValueError("data_qubit_mask selects zero sites")
        self.register_buffer("data_qubit_flat_indices", data_qubit_flat_indices)

        flat_logical = logical_masks.reshape(logical_masks.shape[0], -1)  # (K, G*G)
        logical_over_qubits = flat_logical[:, data_qubit_flat_indices].float()  # (K, Q)
        support_size = logical_over_qubits.sum(dim=-1, keepdim=True)
        if torch.any(support_size == 0):
            raise ValueError("every logical observable needs a nonempty data-qubit support")
        self.register_buffer("logical_pool_weights", logical_over_qubits / support_size)  # rows sum to 1

        num_logicals = logical_masks.shape[0]
        mlp_hidden_dim = mlp_hidden_dim or 2 * hidden_dim

        self.pre_norm = nn.BatchNorm3d(hidden_dim)
        self.act = nn.SiLU()
        self.scatter_conv = nn.Conv3d(
            hidden_dim,
            hidden_dim,
            kernel_size=(rounds, 3, 3),
            padding=(0, 1, 1),
            padding_mode="zeros",
            bias=False,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, 1),
        )
        self.num_logicals = num_logicals

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, G, G) with T == rounds passed at construction time.
        h = self.scatter_conv(self.act(self.pre_norm(x)))  # (B, C, 1, G, G)
        h = h.squeeze(2)  # (B, C, G, G)
        b, c, g, _ = h.shape
        h_flat = h.reshape(b, c, g * g)  # (B, C, G*G)

        qubit_feats = h_flat[:, :, self.data_qubit_flat_indices]  # (B, C, Q)
        qubit_feats = qubit_feats.transpose(1, 2)  # (B, Q, C)

        pooled = torch.einsum("kq,bqc->bkc", self.logical_pool_weights, qubit_feats)  # (B, K, C)
        logits = self.head(pooled).squeeze(-1)  # (B, K)
        
        # --- MuP Scaling ---
        # Scale down logits proportionally to hidden_dim growth. Base width is assumed to be 32.
        mup_scale = 32.0 / self.head[0].in_features
        return logits * mup_scale
