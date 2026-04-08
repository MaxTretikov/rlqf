"""Critic loss: supervised regression for the QM critic.

From docs/losses/critic-loss.md:

    L_critic(phi) = (1/|D_cal|) sum (C_phi(R, V_theta(R)) - |V_theta(R) - E_0(R)|)^2

The critic learns to predict the absolute MM/QM error given a configuration
and the actor's energy prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class CriticLoss(nn.Module):
    """Supervised regression loss for the QM critic.

    Trains C_phi to predict |V_theta(R) - E_0(R)| from (R, V_theta(R)).
    """

    def forward(
        self,
        error_score: Tensor,
        energy_pred: Tensor,
        energy_ref: Tensor,
    ) -> dict[str, Tensor]:
        """Compute the critic loss.

        Parameters
        ----------
        error_score : (B,) critic's predicted error C_phi(R, V_theta(R)).
        energy_pred : (B,) actor's energy predictions V_theta(R).
        energy_ref : (B,) QM reference energies E_0(R).

        Returns
        -------
        dict with:
          - "loss": scalar critic loss
          - "target_error": (B,) the true absolute errors (detached)
        """
        # Target: absolute MM/QM error
        target_error = (energy_pred.detach() - energy_ref).abs()

        # MSE between predicted and actual error
        loss = (error_score - target_error).pow(2).mean()

        return {
            "loss": loss,
            "target_error": target_error.detach(),
        }
