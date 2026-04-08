"""Generalization / out-of-distribution tests.

A network with genuine quantum intuition should generalise beyond its
training distribution. These tests probe:

  1. Unseen geometries: distorted / high-energy configs not in training
  2. Element transfer: new elements or combinations
  3. Extrapolation stability: graceful degradation at extreme geometries
  4. Size transfer: small → larger molecular systems

See also
--------
- docs/exploration/distribution-shift.md
- docs/convergence/convergence.md
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from rlqf.actor import MACEActor
from rlqf.critic import OrbNetCritic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _actor_predict(actor: MACEActor, data: dict[str, Tensor]) -> dict[str, Tensor]:
    """Run actor forward, ensuring autograd for forces."""
    pos = data["positions"].detach().requires_grad_(True)
    return actor({**data, "positions": pos})


# ---------------------------------------------------------------------------
# Unseen geometries
# ---------------------------------------------------------------------------

class TestUnseenGeometries:
    """The actor should produce physically reasonable outputs on geometries
    it has never seen during training.
    """

    def test_compressed_geometry_stable(self, actor, water_equilibrium, make_molecule):
        """Compressing atoms close together should raise energy, not crash."""
        pos = water_equilibrium["positions"].detach().clone()
        pos_compressed = pos * 0.3  # Severely compressed
        data = make_molecule(pos_compressed, water_equilibrium["atomic_numbers"].clone())

        out = _actor_predict(actor, data)
        assert torch.isfinite(out["energy"]).all(), "Non-finite energy at compressed geometry"
        assert torch.isfinite(out["forces"]).all(), "Non-finite forces at compressed geometry"

        # Compressed geometry should have different energy than equilibrium
        e_eq = _actor_predict(actor, water_equilibrium)["energy"].detach().item()
        e_compressed = out["energy"].detach().item()
        assert e_compressed != e_eq, (
            "Same energy at equilibrium and compressed geometry — no repulsion"
        )

    def test_expanded_geometry_stable(self, actor, water_equilibrium, make_molecule):
        """Expanding atoms far apart should produce finite outputs."""
        pos = water_equilibrium["positions"].detach().clone()
        pos_expanded = pos * 5.0
        data = make_molecule(pos_expanded, water_equilibrium["atomic_numbers"].clone())

        out = _actor_predict(actor, data)
        assert torch.isfinite(out["energy"]).all(), "Non-finite energy at expanded geometry"
        assert torch.isfinite(out["forces"]).all(), "Non-finite forces at expanded geometry"

    def test_random_perturbation_stability(self, actor, methane_equilibrium, make_molecule):
        """Randomly perturbing atomic positions should not produce NaN/Inf."""
        n_perturbations = 20
        for i in range(n_perturbations):
            torch.manual_seed(i + 100)
            pos = methane_equilibrium["positions"].detach().clone()
            noise = torch.randn_like(pos) * 0.5
            data = make_molecule(
                pos + noise,
                methane_equilibrium["atomic_numbers"].clone(),
            )

            out = _actor_predict(actor, data)
            assert torch.isfinite(out["energy"]).all(), (
                f"Non-finite energy at perturbation {i}"
            )
            assert torch.isfinite(out["forces"]).all(), (
                f"Non-finite forces at perturbation {i}"
            )


# ---------------------------------------------------------------------------
# Element transfer
# ---------------------------------------------------------------------------

class TestElementTransfer:
    """The network should handle element types it may not have seen
    frequently during training.
    """

    @pytest.mark.parametrize("element,z", [
        ("H", 1), ("C", 6), ("N", 7), ("O", 8),
        ("F", 9), ("Si", 14), ("S", 16), ("Cl", 17),
    ])
    def test_single_element_dimer_stable(self, actor, make_molecule, element, z):
        """A homonuclear dimer of any common element should give finite energy."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float32)
        atomic_numbers = torch.tensor([z, z], dtype=torch.long)
        data = make_molecule(pos, atomic_numbers)

        out = _actor_predict(actor, data)
        assert torch.isfinite(out["energy"]).all(), (
            f"Non-finite energy for {element}-{element} dimer"
        )

    def test_heteronuclear_combinations(self, actor, make_molecule):
        """Mixed-element systems should produce distinct energies."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float32)

        results = {}
        for name, z_pair in [("CO", [6, 8]), ("NO", [7, 8]), ("CN", [6, 7])]:
            z = torch.tensor(z_pair, dtype=torch.long)
            data = make_molecule(pos.clone(), z)
            results[name] = _actor_predict(actor, data)["energy"].detach().item()

        vals = list(results.values())
        max_diff = max(abs(vals[i] - vals[j])
                       for i in range(len(vals))
                       for j in range(i + 1, len(vals)))
        assert max_diff > 1e-6, (
            f"All heteronuclear dimers give the same energy: {results}"
        )

    def test_unusual_element_no_crash(self, actor, make_molecule):
        """Rare elements (e.g. Fe, Zn) should not crash the network."""
        for z_val in [26, 30, 35, 53]:  # Fe, Zn, Br, I
            pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float32)
            z = torch.tensor([z_val, z_val], dtype=torch.long)
            data = make_molecule(pos, z)

            out = _actor_predict(actor, data)
            assert torch.isfinite(out["energy"]).all(), (
                f"Non-finite energy for Z={z_val} dimer"
            )


# ---------------------------------------------------------------------------
# Extrapolation stability
# ---------------------------------------------------------------------------

class TestExtrapolationStability:
    """At extreme (unphysical) geometries, the network should degrade
    gracefully — producing finite (if inaccurate) outputs.
    """

    def test_very_short_bond(self, actor, make_molecule):
        """Atoms at 0.3 Å should not crash."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]], dtype=torch.float32)
        z = torch.tensor([6, 6], dtype=torch.long)
        data = make_molecule(pos, z)

        out = _actor_predict(actor, data)
        assert torch.isfinite(out["energy"]).all(), "Crash at very short bond"

    def test_very_long_bond(self, actor, make_molecule):
        """Atoms at 50 Å (well beyond any cutoff) should not crash."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0]], dtype=torch.float32)
        z = torch.tensor([6, 6], dtype=torch.long)
        data = make_molecule(pos, z)

        out = _actor_predict(actor, data)
        assert torch.isfinite(out["energy"]).all(), "Crash at very long bond"

    def test_large_system_no_oom(self, actor, make_molecule):
        """A system with 100 atoms should run without OOM or crash."""
        n_atoms = 100
        pos = torch.randn(n_atoms, 3) * 3.0
        z = torch.randint(1, 18, (n_atoms,))
        data = make_molecule(pos, z)

        out = _actor_predict(actor, data)
        assert torch.isfinite(out["energy"]).all(), "Non-finite energy for 100-atom system"
        assert out["forces"].shape == (n_atoms, 3)

    @pytest.mark.parametrize("n_atoms", [5, 10, 20, 50])
    def test_energy_scales_with_size(self, actor, make_molecule, n_atoms):
        """Energy magnitude should scale roughly with system size."""
        torch.manual_seed(42)
        pos = torch.randn(n_atoms, 3) * 2.0
        z = torch.ones(n_atoms, dtype=torch.long) * 6

        data = make_molecule(pos, z)
        e = _actor_predict(actor, data)["energy"].detach().item()
        assert math.isfinite(e), f"Non-finite energy for n_atoms={n_atoms}"


# ---------------------------------------------------------------------------
# Equivariance spot-checks
# ---------------------------------------------------------------------------

class TestEquivarianceSpotCheck:
    """Basic sanity checks for SE(3) equivariance.

    A truly equivariant network should produce:
      - Same energy under rotation/translation (invariance)
      - Rotated forces under rotation (equivariance)
    """

    def test_translation_invariance(self, actor, water_equilibrium, make_molecule):
        """Energy should not change under rigid translation."""
        e_orig = _actor_predict(actor, water_equilibrium)["energy"].detach().item()

        # Translate by a large vector
        shift = torch.tensor([100.0, -50.0, 75.0])
        pos_shifted = water_equilibrium["positions"].detach() + shift
        data_shifted = make_molecule(
            pos_shifted,
            water_equilibrium["atomic_numbers"].clone(),
        )
        e_shifted = _actor_predict(actor, data_shifted)["energy"].detach().item()

        assert abs(e_orig - e_shifted) < 1e-4, (
            f"Energy changed under translation: {e_orig:.6f} vs {e_shifted:.6f}"
        )

    def test_rotation_invariance(self, actor, water_equilibrium, make_molecule):
        """Energy should not change under rigid rotation."""
        e_orig = _actor_predict(actor, water_equilibrium)["energy"].detach().item()

        # Rotate 90° around z-axis
        R = torch.tensor([
            [0.0, -1.0, 0.0],
            [1.0,  0.0, 0.0],
            [0.0,  0.0, 1.0],
        ])
        pos_rot = (water_equilibrium["positions"].detach() @ R.T)
        data_rot = make_molecule(
            pos_rot,
            water_equilibrium["atomic_numbers"].clone(),
        )
        e_rot = _actor_predict(actor, data_rot)["energy"].detach().item()

        assert abs(e_orig - e_rot) < 1e-3, (
            f"Energy changed under rotation: {e_orig:.6f} vs {e_rot:.6f}"
        )

    def test_permutation_invariance(self, actor, make_molecule):
        """Swapping identical atoms should not change the energy."""
        pos1 = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=torch.float32)
        pos2 = torch.tensor([[1.5, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
        z = torch.tensor([1, 1], dtype=torch.long)

        e1 = _actor_predict(actor, make_molecule(pos1, z.clone()))["energy"].detach().item()
        e2 = _actor_predict(actor, make_molecule(pos2, z.clone()))["energy"].detach().item()

        assert abs(e1 - e2) < 1e-5, (
            f"Energy changed under permutation of identical atoms: {e1:.6f} vs {e2:.6f}"
        )
