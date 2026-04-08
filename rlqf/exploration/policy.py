"""Exploration policy: pi_psi(a|s) for the RLQF outer loop.

Implements the Soft RLQF policy from docs/actor-critic/soft-rlqf.md:
an entropy-regularized policy that proposes molecular configurations
to maximize the critic's error signal while maintaining diverse exploration.

The policy gradient with critic baseline follows
docs/actor-critic/policy-gradient.md:

    grad_psi J(psi) = E_tau [ sum_t grad_psi log pi_psi(a_t|s_t) * A(s_t, a_t) ]

where A is the advantage function derived from the critic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal


@dataclass
class PolicyConfig:
    """Configuration for the exploration policy.

    Attributes
    ----------
    state_dim : int
        Dimension of the state representation (flattened config + actor summary).
    action_dim : int
        Dimension of the action space (3N for N atoms, or a reduced representation).
    hidden_dim : int
        Hidden layer width.
    num_layers : int
        Number of hidden layers.
    beta : float
        Entropy temperature for Soft RLQF (see docs/actor-critic/soft-rlqf.md).
        Higher beta encourages more diverse exploration.
    log_std_min : float
        Minimum log-std to prevent collapse.
    log_std_max : float
        Maximum log-std to prevent divergence.
    """

    state_dim: int = 256
    action_dim: int = 64
    hidden_dim: int = 256
    num_layers: int = 3
    beta: float = 0.1
    log_std_min: float = -5.0
    log_std_max: float = 2.0


class ExplorationPolicy(nn.Module):
    """Entropy-regularized exploration policy (Soft RLQF).

    Proposes perturbations to molecular configurations to find regions
    where the MM actor is maximally wrong. Outputs a Gaussian distribution
    over configuration-space displacements.

    The policy operates on a learned state representation that encodes:
      - Current molecular configuration R_t (via a graph encoder)
      - Summary statistics of the current actor parameters theta_t

    Parameters
    ----------
    config : PolicyConfig
        Policy hyperparameters.
    """

    def __init__(self, config: PolicyConfig | None = None):
        super().__init__()
        self.config = config or PolicyConfig()

        # State encoder: graph features -> fixed-dim state vector
        layers = []
        in_dim = self.config.state_dim
        for _ in range(self.config.num_layers):
            layers.extend([
                nn.Linear(in_dim, self.config.hidden_dim),
                nn.SiLU(),
            ])
            in_dim = self.config.hidden_dim
        self.encoder = nn.Sequential(*layers)

        # Mean and log-std heads for Gaussian policy
        self.mean_head = nn.Linear(self.config.hidden_dim, self.config.action_dim)
        self.log_std_head = nn.Linear(self.config.hidden_dim, self.config.action_dim)

        # Value head for baseline (shared encoder)
        self.value_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim // 2, 1),
        )

    def forward(self, state: Tensor) -> dict[str, Tensor]:
        """Compute policy distribution and value estimate.

        Parameters
        ----------
        state : (B, state_dim) encoded state representation.

        Returns
        -------
        dict with:
          - "mean": (B, action_dim) mean of the Gaussian policy
          - "log_std": (B, action_dim) log-std (clamped)
          - "value": (B,) state value estimate V^pi(s)
        """
        h = self.encoder(state)

        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        log_std = log_std.clamp(self.config.log_std_min, self.config.log_std_max)

        value = self.value_head(h).squeeze(-1)

        return {"mean": mean, "log_std": log_std, "value": value}

    def sample(self, state: Tensor) -> dict[str, Tensor]:
        """Sample an action and compute log-probability.

        Parameters
        ----------
        state : (B, state_dim) encoded state.

        Returns
        -------
        dict with:
          - "action": (B, action_dim) sampled configuration perturbation
          - "log_prob": (B,) log pi_psi(a|s)
          - "value": (B,) state value estimate
          - "entropy": (B,) policy entropy H(pi_psi(.|s))
        """
        out = self.forward(state)
        dist = Normal(out["mean"], out["log_std"].exp())

        # Reparameterized sample for gradient flow
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)  # (B,)
        entropy = dist.entropy().sum(dim=-1)  # (B,)

        return {
            "action": action,
            "log_prob": log_prob,
            "value": out["value"],
            "entropy": entropy,
        }

    def evaluate(self, state: Tensor, action: Tensor) -> dict[str, Tensor]:
        """Evaluate log-probability and value for given state-action pairs.

        Used in PPO-style updates where we need to re-evaluate old trajectories.

        Parameters
        ----------
        state : (B, state_dim)
        action : (B, action_dim)

        Returns
        -------
        dict with:
          - "log_prob": (B,)
          - "value": (B,)
          - "entropy": (B,)
        """
        out = self.forward(state)
        dist = Normal(out["mean"], out["log_std"].exp())

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return {
            "log_prob": log_prob,
            "value": out["value"],
            "entropy": entropy,
        }

    def compute_policy_loss(
        self,
        log_probs: Tensor,
        advantages: Tensor,
        entropies: Tensor,
        old_log_probs: Tensor | None = None,
        clip_epsilon: float = 0.2,
    ) -> dict[str, Tensor]:
        """Compute the Soft RLQF policy gradient loss.

        Implements the entropy-regularized objective from
        docs/actor-critic/soft-rlqf.md with optional PPO clipping.

        Parameters
        ----------
        log_probs : (B,) current log pi_psi(a|s).
        advantages : (B,) advantage estimates A(s, a).
        entropies : (B,) policy entropies H(pi_psi(.|s)).
        old_log_probs : (B,) optional, for PPO clipping.
        clip_epsilon : float
            PPO clipping parameter.

        Returns
        -------
        dict with:
          - "loss": scalar policy loss (to minimize)
          - "entropy": scalar mean entropy (detached)
          - "approx_kl": scalar approximate KL divergence (detached)
        """
        if old_log_probs is not None:
            # PPO-style clipped objective
            ratio = (log_probs - old_log_probs).exp()
            clipped_ratio = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon)
            pg_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            approx_kl = (old_log_probs - log_probs).mean().detach()
        else:
            # REINFORCE with baseline
            pg_loss = -(log_probs * advantages.detach()).mean()
            approx_kl = torch.tensor(0.0)

        # Entropy bonus (Soft RLQF)
        entropy_bonus = -self.config.beta * entropies.mean()

        loss = pg_loss + entropy_bonus

        return {
            "loss": loss,
            "entropy": entropies.mean().detach(),
            "approx_kl": approx_kl,
        }
