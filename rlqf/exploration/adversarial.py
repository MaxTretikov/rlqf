"""Adversarial configuration generation via Langevin dynamics.

From docs/exploration/adversarial-generation.md:

    R_{k+1} = R_k + eta_R * grad_R C_phi(R_k, V_theta(R_k)) + sigma * xi_k

Directly optimizes in configuration space to find where the MM network
is most wrong, using gradient ascent through the differentiable critic
with Langevin noise for exploration.

This exploits the verification-generation asymmetry most directly:
each gradient step through the critic costs O(C_ver) per step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class LangevinConfig:
    """Configuration for adversarial Langevin dynamics.

    Attributes
    ----------
    num_steps : int
        Number of Langevin steps per adversarial generation.
    step_size : float
        Gradient ascent step size eta_R.
    noise_scale : float
        Langevin noise sigma (0 = pure gradient ascent).
    max_displacement : float
        Maximum total displacement from initial config (prevents
        unphysical configurations per docs section "Physical Validity").
    min_bond_length : float
        Minimum allowed interatomic distance (steric constraint).
    """

    num_steps: int = 50
    step_size: float = 0.01
    noise_scale: float = 0.005
    max_displacement: float = 2.0
    min_bond_length: float = 0.8


def adversarial_langevin(
    positions: Tensor,
    atomic_numbers: Tensor,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    config: LangevinConfig | None = None,
    build_graph_fn=None,
) -> dict[str, Tensor]:
    """Generate adversarial configurations via Langevin dynamics in the error landscape.

    Implements docs/exploration/adversarial-generation.md:
        R* = argmax_R C_phi(R, V_theta(R))
    approximated by Langevin dynamics with gradient ascent through the critic.

    Parameters
    ----------
    positions : (N, 3) initial atomic positions.
    atomic_numbers : (N,) atomic numbers.
    actor : MACEActor, the MM potential V_theta.
    critic : OrbNetCritic, the QM error estimator C_phi.
    config : LangevinConfig, dynamics parameters.
    build_graph_fn : callable, builds neighbor list from positions.

    Returns
    -------
    dict with:
      - "positions": (N, 3) adversarial configuration
      - "error_trajectory": (num_steps,) critic scores along the path
      - "total_displacement": scalar total displacement from start
    """
    from rlqf.utils.graph import build_neighbor_list

    cfg = config or LangevinConfig()
    if build_graph_fn is None:
        build_graph_fn = build_neighbor_list

    R = positions.detach().clone()
    R_init = R.clone()
    batch = torch.zeros(R.shape[0], dtype=torch.long, device=R.device)
    error_trajectory = []

    for k in range(cfg.num_steps):
        R.requires_grad_(True)

        # Build graph and evaluate actor + critic
        edges = build_graph_fn(R.detach(), cutoff=5.0)
        data = {
            "positions": R,
            "atomic_numbers": atomic_numbers,
            "edge_index": edges,
            "batch": batch,
        }

        # Actor forward (need grad through R for Langevin)
        actor_out = actor(data)
        energy = actor_out["energy"]

        # Critic forward (need grad through R)
        critic_out = critic(data, energy)
        error_score = critic_out["error_score"]
        error_trajectory.append(error_score.item())

        # Gradient ascent: maximize C_phi w.r.t. R
        grad_R = torch.autograd.grad(
            error_score.sum(), R, retain_graph=False
        )[0]

        with torch.no_grad():
            # Langevin step: R_{k+1} = R_k + eta * grad + sigma * noise
            noise = torch.randn_like(R) * cfg.noise_scale
            R = R + cfg.step_size * grad_R + noise

            # --- Physical validity constraints (docs Section "Physical Validity") ---

            # 1. Clamp total displacement from initial config
            displacement = R - R_init
            disp_norm = displacement.norm()
            if disp_norm > cfg.max_displacement:
                R = R_init + displacement * (cfg.max_displacement / disp_norm)

            # 2. Enforce minimum bond length (steric constraint)
            R = _enforce_min_distance(R, cfg.min_bond_length)

    return {
        "positions": R.detach(),
        "error_trajectory": torch.tensor(error_trajectory),
        "total_displacement": (R.detach() - R_init).norm().item(),
    }


def _enforce_min_distance(positions: Tensor, min_dist: float) -> Tensor:
    """Push apart atoms that are closer than min_dist."""
    N = positions.shape[0]
    for i in range(N):
        for j in range(i + 1, N):
            diff = positions[i] - positions[j]
            d = diff.norm()
            if d < min_dist and d > 1e-6:
                # Push apart along the connecting vector
                correction = (min_dist - d) / 2.0 * diff / d
                positions[i] = positions[i] + correction
                positions[j] = positions[j] - correction
    return positions
