"""Energy-force loss: the inner-loop MM training objective.

Implements the weighted MSE loss on energies and forces from
docs/losses/energy-force-loss.md:

    L_EF(theta; D) = (1/|D|) sum w(R) (V_theta(R) - E_0(R))^2
                   + (mu/|D|) sum w(R) ||F_theta(R) - F_0(R)||^2

where w(R) are critic-derived importance weights with prioritization
exponent kappa.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class EnergyForceLoss(nn.Module):
    """Weighted MSE loss on energies and forces.

    Parameters
    ----------
    mu : float
        Balance coefficient between energy and force terms.
        Typical range: 1.0 - 1000.0 (forces are often weighted higher).
    kappa : float
        Prioritization exponent for importance weights.
        0 = uniform, 1 = linear, large = focus on hardest samples.
    """

    def __init__(self, mu: float = 100.0, kappa: float = 1.0):
        super().__init__()
        self.mu = mu
        self.kappa = kappa

    def forward(
        self,
        energy_pred: Tensor,
        forces_pred: Tensor,
        energy_ref: Tensor,
        forces_ref: Tensor,
        batch: Tensor,
        importance_weights: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute the energy-force loss.

        Parameters
        ----------
        energy_pred : (B,) predicted energies V_theta(R).
        forces_pred : (N_total, 3) predicted forces F_theta(R).
        energy_ref : (B,) reference QM energies E_0(R).
        forces_ref : (N_total, 3) reference QM forces F_0(R).
        batch : (N_total,) batch assignment indices.
        importance_weights : (B,) optional critic-derived weights w(R).
            If None, uses uniform weights (kappa=0 behavior).

        Returns
        -------
        dict with keys:
          - "loss": scalar total loss
          - "energy_loss": scalar energy component
          - "force_loss": scalar force component
        """
        B = energy_pred.shape[0]

        # Compute per-sample weights
        if importance_weights is not None:
            w = importance_weights.pow(self.kappa)
            w = w / (w.sum() + 1e-8)
        else:
            w = torch.ones(B, device=energy_pred.device) / B

        # Energy loss: weighted MSE
        energy_err = (energy_pred - energy_ref).pow(2)  # (B,)
        energy_loss = (w * energy_err).sum()

        # Force loss: weighted per-atom MSE, aggregated per graph
        force_err = (forces_pred - forces_ref).pow(2).sum(dim=-1)  # (N_total,)
        # Aggregate force errors per graph
        per_graph_force_err = torch.zeros(B, device=force_err.device, dtype=force_err.dtype)
        atoms_per_graph = torch.zeros(B, device=force_err.device, dtype=force_err.dtype)
        per_graph_force_err.scatter_add_(0, batch, force_err)
        atoms_per_graph.scatter_add_(0, batch, torch.ones_like(force_err))
        per_graph_force_err = per_graph_force_err / atoms_per_graph.clamp(min=1)

        force_loss = (w * per_graph_force_err).sum()

        total_loss = energy_loss + self.mu * force_loss

        return {
            "loss": total_loss,
            "energy_loss": energy_loss.detach(),
            "force_loss": force_loss.detach(),
        }
