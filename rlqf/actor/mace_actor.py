"""MACE actor: V_theta(R) — the MM neural network potential.

Wraps the MACE equivariant message-passing architecture to produce
energies and forces for molecular configurations. In the RLQF framework
this is the "actor" whose parameters theta are updated in the inner loop.

References
----------
- Batatia et al., "MACE: Higher Order Equivariant Message Passing Neural Networks
  for Fast and Accurate Force Fields", NeurIPS 2022.
- docs/formulation/mdp-formulation.md (V_theta definition)
- docs/losses/energy-force-loss.md (inner-loop training objective)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class MACEActorConfig:
    """Configuration for the MACE actor network.

    Attributes
    ----------
    r_max : float
        Radial cutoff distance in Angstroms.
    num_bessel : int
        Number of Bessel basis functions for radial encoding.
    num_polynomial_cutoff : int
        Order of the polynomial envelope cutoff.
    max_ell : int
        Maximum spherical harmonics order (angular resolution).
    num_interactions : int
        Number of message-passing layers.
    hidden_irreps : str
        Hidden feature irreps in e3nn format, e.g. "128x0e + 128x1o".
    correlation : int
        Body-order correlation (MACE's key innovation; 2 = 3-body, 3 = 4-body).
    num_elements : int
        Number of chemical elements supported.
    avg_num_neighbors : float
        Average number of neighbors per atom (for normalization).
    """

    r_max: float = 5.0
    num_bessel: int = 8
    num_polynomial_cutoff: int = 6
    max_ell: int = 3
    num_interactions: int = 2
    hidden_irreps: str = "128x0e + 128x1o"
    correlation: int = 3
    num_elements: int = 118
    avg_num_neighbors: float = 10.0


class MACEActor(nn.Module):
    """MACE-based molecular mechanics potential (the RLQF actor).

    This module wraps the MACE architecture to provide:
      - V_theta(R): predicted energy for configuration R
      - F_theta(R) = -grad_R V_theta(R): predicted forces

    The actor's parameters theta are updated in the RLQF inner loop
    via the energy-force loss (see rlqf.losses.energy_force).

    Parameters
    ----------
    config : MACEActorConfig
        Architecture hyperparameters.
    pretrained : str, optional
        Path or identifier for a pretrained MACE checkpoint.
        If provided, loads weights and optionally freezes early layers.
    freeze_backbone : bool
        If True and pretrained is set, freeze message-passing layers
        and only fine-tune the readout head.
    """

    def __init__(
        self,
        config: MACEActorConfig | None = None,
        pretrained: str | None = None,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.config = config or MACEActorConfig()
        self._build_model(pretrained, freeze_backbone)

    def _build_model(self, pretrained: str | None, freeze_backbone: bool) -> None:
        """Construct or load the MACE model."""
        try:
            from mace.tools.scripts_utils import get_default_args
            from mace.modules import MACE

            # Build from MACE library
            self.model = MACE(
                r_max=self.config.r_max,
                num_bessel=self.config.num_bessel,
                num_polynomial_cutoff=self.config.num_polynomial_cutoff,
                max_ell=self.config.max_ell,
                interaction_cls_first=None,  # use MACE defaults
                interaction_cls=None,
                num_interactions=self.config.num_interactions,
                hidden_irreps=self._parse_irreps(self.config.hidden_irreps),
                correlation=self.config.correlation,
                num_elements=self.config.num_elements,
                avg_num_neighbors=self.config.avg_num_neighbors,
            )
        except ImportError:
            # Fallback: build a minimal stand-in for development/testing
            # when mace-torch is not installed
            self.model = _MinimalPotential(self.config)

        if pretrained is not None:
            self._load_pretrained(pretrained)

        if freeze_backbone:
            self._freeze_backbone()

    def _parse_irreps(self, irreps_str: str):
        """Parse irreps string, returning e3nn Irreps if available."""
        try:
            from e3nn.o3 import Irreps
            return Irreps(irreps_str)
        except ImportError:
            return irreps_str

    def _load_pretrained(self, path: str) -> None:
        """Load pretrained weights from checkpoint or MACE foundation model."""
        try:
            from mace.calculators import mace_mp

            # Try loading as a MACE foundation model identifier
            calc = mace_mp(model=path, default_dtype="float64")
            self.model.load_state_dict(calc.models[0].state_dict(), strict=False)
        except Exception:
            # Fall back to direct checkpoint loading
            state = torch.load(path, map_location="cpu", weights_only=True)
            if "model" in state:
                state = state["model"]
            self.model.load_state_dict(state, strict=False)

    def _freeze_backbone(self) -> None:
        """Freeze message-passing layers, leaving readout trainable."""
        for name, param in self.model.named_parameters():
            if "readout" not in name and "head" not in name:
                param.requires_grad = False

    def forward(
        self,
        data: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Compute energy and forces for a batch of configurations.

        Parameters
        ----------
        data : dict
            Must contain at minimum:
              - "positions": (N_total, 3) atomic positions
              - "atomic_numbers": (N_total,) integer atomic numbers
              - "edge_index": (2, E) neighbor list
              - "batch": (N_total,) batch assignment indices
            May also contain:
              - "cell": (B, 3, 3) unit cell vectors
              - "shifts": (E, 3) periodic shifts

        Returns
        -------
        dict with keys:
          - "energy": (B,) predicted energies V_theta(R)
          - "forces": (N_total, 3) predicted forces F_theta(R)
        """
        # Ensure positions track gradients for force computation
        positions = data["positions"]
        if not positions.requires_grad:
            positions = positions.detach().requires_grad_(True)
            data["positions"] = positions

        output = self.model(data)

        # Extract energy; compute forces as negative gradient
        energy = output["energy"] if isinstance(output, dict) else output
        forces = -torch.autograd.grad(
            energy.sum(),
            positions,
            create_graph=self.training,
            retain_graph=True,
            allow_unused=True,
        )[0]

        # If positions were unused in graph (e.g. minimal fallback), zero forces
        if forces is None:
            forces = torch.zeros_like(positions)

        return {"energy": energy, "forces": forces}

    @property
    def theta(self) -> list[Tensor]:
        """Convenience accessor for trainable parameters (theta in the docs)."""
        return [p for p in self.parameters() if p.requires_grad]


class _MinimalPotential(nn.Module):
    """Minimal fallback potential for testing without MACE installed.

    NOT physically meaningful — just maintains the correct API shape.
    """

    def __init__(self, config: MACEActorConfig):
        super().__init__()
        self.embed = nn.Embedding(config.num_elements, 64)
        self.mlp = nn.Sequential(
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(self, data: dict[str, Tensor]) -> dict[str, Tensor]:
        z = data["atomic_numbers"].long()
        pos = data["positions"]  # (N_total, 3)
        h = self.embed(z)
        # Mix in positions so autograd can compute forces
        pos_feat = pos.sum(dim=-1, keepdim=True).expand_as(h[:, :1])
        h = h + 0.0 * pos_feat  # Identity contribution, but creates grad path
        per_atom = self.mlp(h).squeeze(-1)  # (N_total,)
        batch = data.get("batch", torch.zeros(z.shape[0], dtype=torch.long, device=z.device))
        energy = torch.zeros(batch.max() + 1, device=z.device, dtype=per_atom.dtype)
        energy.scatter_add_(0, batch, per_atom)
        return {"energy": energy}
