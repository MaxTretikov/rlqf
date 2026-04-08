"""Shared fixtures for the RLQF validation / testing pipeline.

These fixtures provide:
  - Model instantiation (actor, critic, policy) with deterministic seeds
  - OrbNet-proxy QM reference data generation
  - Molecular test case factories (small molecules, distortions, scans)
  - A trained-vs-untrained actor pair for comparative tests
"""

from __future__ import annotations

import math
from typing import Generator

import pytest
import torch
from torch import Tensor

from rlqf.actor import MACEActor
from rlqf.actor.mace_actor import MACEActorConfig
from rlqf.critic import OrbNetCritic
from rlqf.critic.orbnet_critic import OrbNetCriticConfig
from rlqf.exploration.policy import ExplorationPolicy, PolicyConfig
from rlqf.utils.graph import build_neighbor_list


# ---------------------------------------------------------------------------
# Deterministic seeding
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _deterministic_seed():
    """Ensure reproducibility across all tests."""
    torch.manual_seed(42)
    yield


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def actor() -> MACEActor:
    """A freshly initialised MACE actor (uses the minimal fallback)."""
    torch.manual_seed(42)
    cfg = MACEActorConfig(num_interactions=2, num_elements=118)
    return MACEActor(config=cfg).eval()


@pytest.fixture(scope="session")
def critic() -> OrbNetCritic:
    """A freshly initialised OrbNet critic (uses the minimal backbone)."""
    torch.manual_seed(42)
    cfg = OrbNetCriticConfig(device="cpu", use_energy_input=True)
    return OrbNetCritic(config=cfg).eval()


@pytest.fixture(scope="session")
def policy() -> ExplorationPolicy:
    """An exploration policy with default config."""
    torch.manual_seed(42)
    return ExplorationPolicy(PolicyConfig()).eval()


@pytest.fixture(scope="session")
def untrained_actor() -> MACEActor:
    """A second, independently-initialised actor for baseline comparisons.

    Uses a *different* seed so it represents a truly untrained network,
    not a copy of the primary actor.
    """
    torch.manual_seed(999)
    cfg = MACEActorConfig(num_interactions=2, num_elements=118)
    return MACEActor(config=cfg).eval()


# ---------------------------------------------------------------------------
# Molecular test-case factories
# ---------------------------------------------------------------------------

ELEMENT_LIBRARY = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9,
    "Si": 14, "P": 15, "S": 16, "Cl": 17,
}


def _make_graph(
    positions: Tensor,
    atomic_numbers: Tensor,
    cutoff: float = 5.0,
) -> dict[str, Tensor]:
    """Build the data dict expected by actor / critic forward()."""
    edge_index = build_neighbor_list(positions, cutoff=cutoff)
    batch = torch.zeros(positions.shape[0], dtype=torch.long)
    return {
        "positions": positions.requires_grad_(True),
        "atomic_numbers": atomic_numbers,
        "edge_index": edge_index,
        "batch": batch,
    }


@pytest.fixture()
def make_molecule():
    """Factory fixture: create an arbitrary molecule graph dict.

    Usage in tests::

        data = make_molecule(positions_tensor, atomic_numbers_tensor)
    """
    return _make_graph


@pytest.fixture()
def water_equilibrium() -> dict[str, Tensor]:
    """An approximate equilibrium water molecule (H-O-H ~104.5 deg)."""
    positions = torch.tensor([
        [0.000, 0.000, 0.000],   # O
        [0.757, 0.586, 0.000],   # H
        [-0.757, 0.586, 0.000],  # H
    ], dtype=torch.float32)
    z = torch.tensor([8, 1, 1], dtype=torch.long)
    return _make_graph(positions, z)


@pytest.fixture()
def methane_equilibrium() -> dict[str, Tensor]:
    """Approximate tetrahedral CH4."""
    r = 1.089  # C-H bond length in Angstroms
    positions = torch.tensor([
        [0.0, 0.0, 0.0],                                   # C
        [r, r, r],                                          # H
        [r, -r, -r],                                        # H
        [-r, r, -r],                                        # H
        [-r, -r, r],                                        # H
    ], dtype=torch.float32) / math.sqrt(3.0)
    z = torch.tensor([6, 1, 1, 1, 1], dtype=torch.long)
    return _make_graph(positions, z)


@pytest.fixture()
def ethane_equilibrium() -> dict[str, Tensor]:
    """Approximate staggered ethane C2H6."""
    positions = torch.tensor([
        # Carbons along x-axis
        [-0.762, 0.000, 0.000],  # C1
        [0.762, 0.000, 0.000],   # C2
        # H on C1
        [-1.156, 1.019, 0.000],
        [-1.156, -0.510, 0.883],
        [-1.156, -0.510, -0.883],
        # H on C2
        [1.156, -1.019, 0.000],
        [1.156, 0.510, -0.883],
        [1.156, 0.510, 0.883],
    ], dtype=torch.float32)
    z = torch.tensor([6, 6, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    return _make_graph(positions, z)


# ---------------------------------------------------------------------------
# QM reference generator (OrbNet proxy)
# ---------------------------------------------------------------------------

@pytest.fixture()
def qm_reference(critic: OrbNetCritic):
    """Factory: compute OrbNet-proxy QM energies & features for a data dict.

    Returns a callable: ``ref = qm_reference(data)`` → dict with keys
    ``qm_energy``, ``graph_features``, and (if available) ``qm_forces``.
    """

    @torch.no_grad()
    def _compute(data: dict[str, Tensor]) -> dict[str, Tensor]:
        # Use the critic's full forward to get backbone QM energy
        # We pass a dummy energy_pred (zeros) since we only want the
        # backbone outputs, not the error head.
        dummy_energy = torch.zeros(data["batch"].max().item() + 1)
        out = critic(data, dummy_energy)
        return {
            "qm_energy": out["qm_energy"],
            "graph_features": out["graph_features"],
            "qm_forces": out.get("qm_forces"),
        }

    return _compute


# ---------------------------------------------------------------------------
# Scan generators
# ---------------------------------------------------------------------------

@pytest.fixture()
def bond_stretch_scan():
    """Factory: generate a 1-D potential energy scan along a bond stretch.

    Returns a callable::

        configs = bond_stretch_scan(
            base_positions, atomic_numbers,
            atom_i=0, atom_j=1,
            distances=[0.8, 1.0, ..., 3.0],
        )

    Each element of ``configs`` is a data dict ready for actor/critic.
    """

    def _scan(
        base_positions: Tensor,
        atomic_numbers: Tensor,
        atom_i: int,
        atom_j: int,
        distances: list[float],
    ) -> list[dict[str, Tensor]]:
        configs = []
        direction = base_positions[atom_j] - base_positions[atom_i]
        direction = direction / direction.norm()

        for d in distances:
            pos = base_positions.clone()
            midpoint = (pos[atom_i] + pos[atom_j]) / 2
            pos[atom_i] = midpoint - direction * d / 2
            pos[atom_j] = midpoint + direction * d / 2
            configs.append(_make_graph(pos.clone(), atomic_numbers.clone()))

        return configs

    return _scan


@pytest.fixture()
def torsion_scan():
    """Factory: generate a 1-D torsion scan around a dihedral angle.

    Rotates all atoms bonded to atom_j (on the far side of the i-j bond)
    by the specified dihedral angles.

    Returns a callable::

        configs = torsion_scan(
            base_positions, atomic_numbers,
            axis_i=0, axis_j=1,
            rotating_atoms=[4, 5, 6],
            angles_deg=[0, 30, 60, ..., 360],
        )
    """

    def _scan(
        base_positions: Tensor,
        atomic_numbers: Tensor,
        axis_i: int,
        axis_j: int,
        rotating_atoms: list[int],
        angles_deg: list[float],
    ) -> list[dict[str, Tensor]]:
        configs = []
        axis = base_positions[axis_j] - base_positions[axis_i]
        axis = axis / axis.norm()
        origin = base_positions[axis_j]

        for angle in angles_deg:
            pos = base_positions.clone()
            theta = math.radians(angle)
            # Rodrigues rotation
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            for idx in rotating_atoms:
                v = pos[idx] - origin
                v_rot = (
                    v * cos_t
                    + torch.linalg.cross(axis.unsqueeze(0), v.unsqueeze(0)).squeeze(0) * sin_t
                    + axis * (axis @ v) * (1 - cos_t)
                )
                pos[idx] = origin + v_rot
            configs.append(_make_graph(pos.clone(), atomic_numbers.clone()))

        return configs

    return _scan


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------

@pytest.fixture()
def random_molecule_batch():
    """Factory: create a batch of N random molecules for stress testing.

    Returns a callable::

        data_list = random_molecule_batch(n_molecules=50, n_atoms=10)
    """

    def _make(
        n_molecules: int = 50,
        n_atoms: int = 10,
        elements: list[int] | None = None,
    ) -> list[dict[str, Tensor]]:
        if elements is None:
            elements = [1, 6, 7, 8]
        batch = []
        for _ in range(n_molecules):
            pos = torch.randn(n_atoms, 3) * 2.0
            z = torch.tensor(
                [elements[i % len(elements)] for i in range(n_atoms)],
                dtype=torch.long,
            )
            batch.append(_make_graph(pos, z))
        return batch

    return _make
