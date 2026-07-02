import torch
import torch.nn as nn

# Vocabulary for the per-site embedding: a detection event either did or didn't
# fire, or the grid site isn't a real stabilizer at all (checkerboard padding on
# the (d+1, d+1) surface-code grid). All three are informative structural
# classes, not one real class plus an inert padding token.
NO_EVENT = 0
EVENT = 1
NOT_A_CHECK_SITE = 2
VOCAB_SIZE = 3


class SyndromeEmbedding(nn.Module):
    """Embeds the per-site syndrome class into an H-dimensional representation.

    Paper: "Binary detection events are first embedded into H-dimensional
    representations at each syndrome location" (Methods, Structure-aware
    neural decoding).

    Deliberately does NOT pass `padding_idx=NOT_A_CHECK_SITE` to nn.Embedding:
    that flag freezes a row at its zero-initialized value and zeros its
    gradient, which is correct for inert NLP padding but wrong here -- "not a
    check site" is a real, fixed structural signal (lattice/boundary geometry)
    that the network should be free to learn a useful representation for.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(VOCAB_SIZE, hidden_dim)

    def forward(self, syndrome_idx: torch.Tensor) -> torch.Tensor:
        """syndrome_idx: LongTensor (B, T, G, G) with values in {0, 1, 2}.

        Returns (B, C, T, G, G), channel-first, ready for Conv3d.
        """
        x = self.embed(syndrome_idx)  # (B, T, G, G, C)
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # (B, C, T, G, G)
        return x


def syndrome_indices_from_detections(
    detections: torch.Tensor, valid_site_mask: torch.Tensor
) -> torch.Tensor:
    """Builds the {0,1,2}-valued input for SyndromeEmbedding.

    detections: bool/float/long tensor (B, T, G, G); nonzero == detection event
        fired at that site and round. Values at invalid sites are ignored.
    valid_site_mask: bool tensor (G, G), True at real stabilizer sites.

    This is the exact seam a future stim-based data pipeline plugs into: it
    only needs to produce `detections` (from a compiled sampler) and reuse the
    fixed `valid_site_mask` for a given code distance (see
    cascade.geometry.surface_code for how that mask is derived today from a
    placeholder, and how it should be derived from stim coordinates later).
    """
    if valid_site_mask.shape != detections.shape[-2:]:
        raise ValueError(
            f"valid_site_mask shape {tuple(valid_site_mask.shape)} does not match "
            f"detections spatial shape {tuple(detections.shape[-2:])}"
        )
    idx = detections.long()
    fill = torch.full_like(idx, NOT_A_CHECK_SITE)
    idx = torch.where(valid_site_mask, idx, fill)
    return idx
