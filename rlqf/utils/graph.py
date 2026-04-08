"""Graph construction and state encoding utilities."""

from __future__ import annotations

import torch
from torch import Tensor


def build_neighbor_list(
    positions: Tensor,
    cutoff: float = 5.0,
    batch: Tensor | None = None,
) -> Tensor:
    """Build a neighbor list (edge_index) from atomic positions.

    Parameters
    ----------
    positions : (N, 3) atomic positions.
    cutoff : float, radial cutoff in Angstroms.
    batch : (N,) optional batch indices.

    Returns
    -------
    edge_index : (2, E) neighbor pairs.
    """
    N = positions.shape[0]
    # Compute pairwise distances
    diff = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, 3)
    dist = diff.norm(dim=-1)  # (N, N)

    # Mask self-interactions
    mask = (dist < cutoff) & (dist > 0)

    # Mask cross-batch interactions if batched
    if batch is not None:
        same_batch = batch.unsqueeze(0) == batch.unsqueeze(1)
        mask = mask & same_batch

    edge_index = mask.nonzero(as_tuple=False).t()  # (2, E)
    return edge_index


def encode_state(
    graph_features: Tensor,
    actor_summary: Tensor | None = None,
    target_dim: int = 256,
) -> Tensor:
    """Encode the RLQF state s = (R, theta) into a fixed-dim vector.

    Combines molecular graph features with an optional summary of the
    actor's current parameters to form the state representation for
    the exploration policy.

    Parameters
    ----------
    graph_features : (B, D_graph) graph-level features from actor/critic.
    actor_summary : (B, D_actor) optional parameter summary statistics.
    target_dim : int, desired output dimension.

    Returns
    -------
    state : (B, target_dim) encoded state.
    """
    if actor_summary is not None:
        combined = torch.cat([graph_features, actor_summary], dim=-1)
    else:
        combined = graph_features

    # Project to target dimension if needed
    if combined.shape[-1] != target_dim:
        proj = torch.nn.functional.adaptive_avg_pool1d(
            combined.unsqueeze(1), target_dim
        ).squeeze(1)
        return proj

    return combined
