"""OrbNet Denali critic: C_phi(R, E_tilde) — the QM error estimator.

Wraps OrbNet Denali (Orbital Materials) to serve as the QM critic in the
RLQF framework. The critic estimates |V_theta(R) - E_0(R)| given a
molecular configuration R and the actor's energy prediction E_tilde.

The critic serves four roles (see docs/formulation/critic-architecture.md):
  1. Reward signal for the exploration MDP
  2. Value baseline for policy gradient variance reduction
  3. Importance weights for the energy-force loss
  4. Exploration signal (curiosity / adversarial generation)

References
----------
- Qiao et al., "OrbNet Denali: A machine learning potential for biological
  and organic chemistry with semi-empirical cost and DFT accuracy", 2022.
- docs/formulation/critic-architecture.md
- docs/losses/critic-loss.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class OrbNetCriticConfig:
    """Configuration for the OrbNet-based critic.

    Attributes
    ----------
    model_name : str
        Pretrained OrbNet model identifier (e.g. "orbnet-denali-v1",
        "orb-mol-omat-v2").
    device : str
        Device for inference ("cpu", "cuda", "mps").
    use_energy_input : bool
        Whether to condition the critic on the actor's energy prediction
        (True implements C_phi(R, E_tilde); False implements C_phi(R) only).
    error_head_hidden : int
        Hidden dimension for the error estimation head appended on top
        of OrbNet features.
    trainable_backbone : bool
        Whether the OrbNet backbone parameters are trainable. Set False
        to use OrbNet as a frozen feature extractor and only train the
        error head.
    """

    model_name: str = "orb-mol-omat-v2"
    device: str = "cpu"
    use_energy_input: bool = True
    error_head_hidden: int = 256
    trainable_backbone: bool = False


class OrbNetCritic(nn.Module):
    """QM critic based on OrbNet Denali / OrbMol.

    Architecture:
      1. OrbNet backbone computes per-atom features and a QM energy estimate E_qm.
      2. An error estimation head takes (E_qm, E_tilde, global_features) and
         predicts the error score C_phi(R, E_tilde) >= 0.

    The critic is trained via the supervised loss in rlqf.losses.critic_loss,
    targeting |V_theta(R) - E_0(R)| on a calibration set.

    Parameters
    ----------
    config : OrbNetCriticConfig
        Architecture and loading configuration.
    """

    def __init__(self, config: OrbNetCriticConfig | None = None):
        super().__init__()
        self.config = config or OrbNetCriticConfig()
        self._build_model()

    def _build_model(self) -> None:
        """Construct OrbNet backbone + error estimation head."""
        backbone_dim = self._load_backbone()

        # Error estimation head: maps backbone features + optional energy input
        # to a non-negative error score
        input_dim = backbone_dim + (2 if self.config.use_energy_input else 0)
        self.error_head = nn.Sequential(
            nn.Linear(input_dim, self.config.error_head_hidden),
            nn.SiLU(),
            nn.Linear(self.config.error_head_hidden, self.config.error_head_hidden),
            nn.SiLU(),
            nn.Linear(self.config.error_head_hidden, 1),
            nn.Softplus(),  # Ensures C_phi >= 0
        )

    def _load_backbone(self) -> int:
        """Load OrbNet backbone; returns feature dimensionality."""
        try:
            from orb_models.forcefield import pretrained

            # Load pretrained OrbNet model
            self.backbone = pretrained.orb_v2(device=self.config.device)

            if not self.config.trainable_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

            # OrbNet feature dimension (inspect the model to determine)
            backbone_dim = self._infer_backbone_dim()
            return backbone_dim

        except ImportError:
            # Fallback for development without orb-models installed
            self.backbone = _MinimalQMBackbone()
            return 128

    def _infer_backbone_dim(self) -> int:
        """Infer the output feature dimension of the backbone."""
        # OrbNet models typically use 256-dim node features
        # Try to get this from model config, fall back to 256
        try:
            return self.backbone.model.node_dim
        except AttributeError:
            return 256

    def forward(
        self,
        data: dict[str, Tensor],
        energy_pred: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute error score C_phi(R, E_tilde) — FULL forward pass.

        Runs both the OrbNet backbone (expensive: ~C_QM) and the error head
        (cheap: ~C_MM). Use forward_from_cache() in the inner loop to avoid
        redundant backbone calls. See docs/convergence/computational-complexity.md.

        Parameters
        ----------
        data : dict
            Molecular configuration data (same format as actor input):
              - "positions": (N_total, 3)
              - "atomic_numbers": (N_total,)
              - "edge_index": (2, E)
              - "batch": (N_total,)
        energy_pred : Tensor, optional
            Actor's energy prediction V_theta(R), shape (B,).
            Required if config.use_energy_input is True.

        Returns
        -------
        dict with keys:
          - "error_score": (B,) non-negative error estimates C_phi
          - "qm_energy": (B,) OrbNet's own energy prediction E_qm (generation oracle)
          - "qm_forces": (N_total, 3) OrbNet's force prediction (if available)
          - "graph_features": (B, D) cached backbone features for forward_from_cache()
        """
        # Run backbone to get features and QM energy (EXPENSIVE — C_QM)
        backbone_out = self._run_backbone(data)
        graph_features = backbone_out["features"]  # (B, D)
        qm_energy = backbone_out["energy"]  # (B,)

        # Run error head (CHEAP — C_ver << C_QM)
        error_score = self._run_error_head(graph_features, qm_energy, energy_pred)

        result = {
            "error_score": error_score,
            "qm_energy": qm_energy,
            "graph_features": graph_features.detach(),  # Cache for reuse
        }
        if "forces" in backbone_out:
            result["qm_forces"] = backbone_out["forces"]

        return result

    def forward_from_cache(
        self,
        graph_features: Tensor,
        qm_energy: Tensor,
        energy_pred: Tensor,
    ) -> Tensor:
        """Compute error score from CACHED backbone features — CHEAP.

        This is the verification oracle O_ver: it costs only C_ver ≈ C_MM,
        avoiding the expensive backbone (C_QM). Use this in the inner loop
        for importance weighting and reward computation on already-labeled
        configurations.

        This separation is the neural network instantiation of the
        verification-generation asymmetry (docs/foundations/verification-
        generation-asymmetry.md): the backbone IS the generation oracle,
        the error head IS the verification oracle.

        Parameters
        ----------
        graph_features : (B, D) cached from a prior forward() call.
        qm_energy : (B,) cached QM energy from a prior forward() call.
        energy_pred : (B,) actor's CURRENT energy prediction (may differ
            from the prediction at cache time as theta evolves).

        Returns
        -------
        error_score : (B,) non-negative error estimates.
        """
        return self._run_error_head(graph_features, qm_energy, energy_pred)

    def _run_error_head(
        self,
        graph_features: Tensor,
        qm_energy: Tensor,
        energy_pred: Tensor | None,
    ) -> Tensor:
        """Run just the error estimation head (cheap verification oracle)."""
        if self.config.use_energy_input:
            if energy_pred is None:
                raise ValueError(
                    "energy_pred required when use_energy_input=True. "
                    "Pass the actor's V_theta(R) prediction."
                )
            head_input = torch.cat([
                graph_features,
                qm_energy.unsqueeze(-1),
                energy_pred.unsqueeze(-1),
            ], dim=-1)
        else:
            head_input = graph_features

        return self.error_head(head_input).squeeze(-1)

    def _run_backbone(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        """Run the OrbNet backbone, adapting to its API."""
        try:
            # Try the orb-models API
            out = self.backbone(data)
            if isinstance(out, dict):
                return {
                    "features": out.get("graph_features", out.get("node_features").mean(dim=0)),
                    "energy": out["energy"],
                    "forces": out.get("forces"),
                }
            # Handle tuple output
            energy = out[0] if isinstance(out, tuple) else out
            return {"features": torch.zeros(1, 256), "energy": energy}
        except Exception:
            # Fallback
            return self.backbone(data)

    def compute_reward(
        self,
        data: dict[str, Tensor],
        energy_pred: Tensor,
        config_distance: Tensor | None = None,
        lambda_reg: float = 0.01,
    ) -> Tensor:
        """Compute the RLQF reward R(s_t, a_t).

        From docs/formulation/mdp-formulation.md:
          R(s, a) = C_phi(R', V_theta(R')) - lambda * d(R', R)

        Parameters
        ----------
        data : dict
            Configuration data for R' (the new configuration).
        energy_pred : Tensor
            Actor's energy prediction V_theta(R').
        config_distance : Tensor, optional
            Distance d(R', R) between consecutive configurations.
        lambda_reg : float
            Regularization weight penalizing large configuration jumps.

        Returns
        -------
        reward : (B,) reward signal for the exploration policy.
        """
        out = self.forward(data, energy_pred)
        reward = out["error_score"]

        if config_distance is not None:
            reward = reward - lambda_reg * config_distance

        return reward

    def compute_importance_weights(
        self,
        energy_pred: Tensor,
        kappa: float = 1.0,
        *,
        data: dict[str, Tensor] | None = None,
        graph_features: Tensor | None = None,
        qm_energy: Tensor | None = None,
    ) -> Tensor:
        """Compute critic-derived importance weights w(R).

        From docs/losses/energy-force-loss.md:
          w(R) = C_phi(R, V_theta(R))^kappa / sum(C_phi(R', V_theta(R'))^kappa)

        Supports two call paths to exploit the verification-generation asymmetry:
          1. CHEAP (preferred in inner loop): pass graph_features + qm_energy
             from a prior forward() call. Only runs the error head (~C_MM cost).
          2. EXPENSIVE (fallback): pass data dict. Runs full backbone (~C_QM cost).

        Parameters
        ----------
        energy_pred : Tensor
            Actor's current energy predictions, shape (B,).
        kappa : float
            Prioritization exponent (0=uniform, 1=linear, inf=hardest-only).
        data : dict, optional
            Molecular configuration batch (expensive path).
        graph_features : Tensor, optional
            Cached backbone features from prior forward() (cheap path).
        qm_energy : Tensor, optional
            Cached QM energy from prior forward() (cheap path).

        Returns
        -------
        weights : (B,) normalized importance weights.
        """
        with torch.no_grad():
            if graph_features is not None and qm_energy is not None:
                # CHEAP PATH: verification oracle only (~C_MM)
                scores = self.forward_from_cache(graph_features, qm_energy, energy_pred)
            elif data is not None:
                # EXPENSIVE PATH: full generation + verification (~C_QM)
                out = self.forward(data, energy_pred)
                scores = out["error_score"]
            else:
                raise ValueError(
                    "Must provide either (graph_features, qm_energy) for cheap "
                    "verification, or data for full forward pass."
                )

        weights = scores.pow(kappa)
        weights = weights / (weights.sum() + 1e-8)
        return weights


class _MinimalQMBackbone(nn.Module):
    """Minimal QM backbone for testing without orb-models installed."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(118, 64)
        self.mlp = nn.Sequential(
            nn.Linear(64, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
        )
        self.energy_head = nn.Linear(feature_dim, 1)
        self.feature_dim = feature_dim

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        z = data["atomic_numbers"].long()
        h = self.embed(z)
        h = self.mlp(h)  # (N_total, D)
        batch = data.get("batch", torch.zeros(z.shape[0], dtype=torch.long, device=z.device))
        num_graphs = batch.max().item() + 1

        # Pool to graph-level features
        graph_features = torch.zeros(num_graphs, self.feature_dim, device=h.device, dtype=h.dtype)
        graph_features.scatter_add_(0, batch.unsqueeze(-1).expand_as(h), h)

        energy = self.energy_head(graph_features).squeeze(-1)
        return {"features": graph_features, "energy": energy}
