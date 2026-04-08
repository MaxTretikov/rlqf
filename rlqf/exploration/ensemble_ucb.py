"""Ensemble UCB exploration reward augmentation.

From docs/exploration/ensemble-ucb.md:

    r_explore(R) = C_phi(R, V_theta(R))
                 + beta_1 * Var_m[V_theta^(m)(R)]^{1/2}
                 + beta_2 * Var_m[C_phi^(m)(R, V_theta(R))]^{1/2}

Maintains ensembles of both actor and critic to capture epistemic
uncertainty, combined via the Upper Confidence Bound principle.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class EnsembleUCBReward(nn.Module):
    """Augmented exploration reward with ensemble disagreement.

    Parameters
    ----------
    beta_1 : float
        Actor disagreement coefficient.
    beta_2 : float
        Critic uncertainty coefficient.
    """

    def __init__(self, beta_1: float = 1.0, beta_2: float = 1.0):
        super().__init__()
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def forward(
        self,
        critic_score: Tensor,
        actor_energies: Tensor,
        critic_scores: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute the UCB-augmented exploration reward.

        Parameters
        ----------
        critic_score : (B,) mean critic error estimate C_phi(R, V_theta).
        actor_energies : (M, B) energy predictions from M actor ensemble members.
        critic_scores : (M, B) optional, error scores from M critic ensemble members.

        Returns
        -------
        dict with:
          - "reward": (B,) augmented exploration reward
          - "actor_disagreement": (B,) actor ensemble std dev
          - "critic_uncertainty": (B,) critic ensemble std dev (if provided)
        """
        # Actor disagreement: std dev across ensemble members
        actor_std = actor_energies.std(dim=0)  # (B,)

        reward = critic_score + self.beta_1 * actor_std

        result = {
            "actor_disagreement": actor_std.detach(),
        }

        if critic_scores is not None:
            critic_std = critic_scores.std(dim=0)  # (B,)
            reward = reward + self.beta_2 * critic_std
            result["critic_uncertainty"] = critic_std.detach()

        result["reward"] = reward
        return result
