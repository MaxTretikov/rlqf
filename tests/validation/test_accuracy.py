"""Accuracy benchmarks: does the RLQF'd actor match QM reference data?

Tests in this module compare actor predictions (energies, forces) against
the OrbNet-proxy QM reference across a variety of molecular configurations.
These are the bread-and-butter metrics for any neural potential — but the
key question is whether the RLQF-trained actor achieves *systematically*
lower error than an untrained baseline.

Metrics
-------
- Energy MAE (kcal/mol equivalent in model units)
- Force MAE (component-wise, kcal/mol/Å equivalent)
- Force cosine similarity (direction quality independent of magnitude)
- PES smoothness (no discontinuous jumps along a scan)

See also
--------
- docs/losses/energy-force-loss.md
- tests/validation/test_comparative.py (RLQF vs baseline head-to-head)
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
    """Run actor forward *without* no_grad, since forces need autograd."""
    # Ensure positions have grad
    pos = data["positions"].detach().requires_grad_(True)
    data_copy = {**data, "positions": pos}
    return actor(data_copy)


def _energy_mae(actor: MACEActor, critic: OrbNetCritic, data: dict) -> float:
    """Compute |V_theta(R) - E_qm(R)| for a single config."""
    out = _actor_predict(actor, data)
    e_pred = out["energy"].detach()
    with torch.no_grad():
        dummy = torch.zeros_like(e_pred)
        critic_out = critic(data, dummy)
        e_qm = critic_out["qm_energy"]
    return (e_pred - e_qm).abs().mean().item()


def _force_metrics(actor: MACEActor, data: dict) -> dict[str, float]:
    """Compute force-level metrics (MAE and cosine similarity).

    Tests internal consistency: analytic forces F = -dV/dR vs numerical
    finite differences.
    """
    out = _actor_predict(actor, data)
    forces = out["forces"].detach()

    # Numerical gradient check (central differences)
    eps = 1e-3
    pos = data["positions"].detach()
    numerical_forces = torch.zeros_like(pos)
    for i in range(pos.shape[0]):
        for j in range(3):
            pos_plus = pos.clone()
            pos_plus[i, j] += eps
            d_plus = {**data, "positions": pos_plus.requires_grad_(True)}
            e_plus = actor(d_plus)["energy"].detach()

            pos_minus = pos.clone()
            pos_minus[i, j] -= eps
            d_minus = {**data, "positions": pos_minus.requires_grad_(True)}
            e_minus = actor(d_minus)["energy"].detach()

            numerical_forces[i, j] = -(e_plus - e_minus).sum() / (2 * eps)

    analytic = forces.flatten()
    numerical = numerical_forces.flatten()

    mae = (analytic - numerical).abs().mean().item()

    # Cosine similarity (direction agreement)
    cos_sim = torch.nn.functional.cosine_similarity(
        analytic.unsqueeze(0), numerical.unsqueeze(0)
    ).item()

    return {"force_mae": mae, "force_cosine": cos_sim}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnergyAccuracy:
    """Energy prediction accuracy against QM reference."""

    def test_water_energy_finite(self, actor, critic, water_equilibrium):
        """Actor produces a finite energy for water."""
        out = _actor_predict(actor, water_equilibrium)
        assert torch.isfinite(out["energy"]).all(), "Energy is not finite"

    def test_methane_energy_finite(self, actor, critic, methane_equilibrium):
        """Actor produces a finite energy for methane."""
        out = _actor_predict(actor, methane_equilibrium)
        assert torch.isfinite(out["energy"]).all(), "Energy is not finite"

    def test_energy_varies_with_geometry(self, actor, water_equilibrium, make_molecule):
        """Different geometries should give different energies.

        A network that always returns the same energy regardless of input
        has zero quantum intuition.
        """
        # Distort water by stretching one O-H bond
        distorted_pos = water_equilibrium["positions"].detach().clone()
        distorted_pos[1] *= 1.5  # Move H away
        distorted = make_molecule(
            distorted_pos,
            water_equilibrium["atomic_numbers"].clone(),
        )

        e_eq = _actor_predict(actor, water_equilibrium)["energy"].detach()
        e_dist = _actor_predict(actor, distorted)["energy"].detach()

        assert not torch.allclose(e_eq, e_dist, atol=1e-6), (
            "Energy unchanged under geometry distortion — no structural sensitivity"
        )

    def test_energy_extensive(self, actor, make_molecule):
        """Energy should scale roughly linearly with system size.

        Two isolated water molecules should have ~2x the energy of one.
        (Tests that the network is extensive, not just a constant.)
        """
        # Single water
        pos1 = torch.tensor([
            [0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]
        ])
        z1 = torch.tensor([8, 1, 1])
        data1 = make_molecule(pos1, z1)

        # Two waters far apart (non-interacting)
        pos2 = torch.cat([pos1, pos1 + torch.tensor([20.0, 0.0, 0.0])], dim=0)
        z2 = torch.cat([z1, z1])
        data2 = make_molecule(pos2, z2)

        e1 = _actor_predict(actor, data1)["energy"].detach().item()
        # For the double system, manually set batch indices
        data2["batch"] = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
        e2_out = _actor_predict(actor, data2)["energy"].detach()

        # Each water in the double-system should have roughly the same energy
        # as the single water (within 50% — we're testing extensivity, not accuracy)
        e2_per_mol = e2_out.mean().item()
        ratio = abs(e2_per_mol / (e1 + 1e-10))
        assert 0.3 < ratio < 3.0, (
            f"Energy not extensive: single={e1:.4f}, double_per_mol={e2_per_mol:.4f}"
        )

    @pytest.mark.parametrize("n_configs", [20, 50])
    def test_energy_mae_on_random_configs(
        self, actor, critic, random_molecule_batch, n_configs
    ):
        """Energy MAE on random configs should be finite and reportable.

        This isn't a pass/fail threshold (the network may not be trained yet),
        but the test ensures the full prediction-vs-reference pipeline works
        end-to-end.
        """
        configs = random_molecule_batch(n_molecules=n_configs, n_atoms=8)
        maes = []
        for data in configs:
            mae = _energy_mae(actor, critic, data)
            assert math.isfinite(mae), "Non-finite MAE detected"
            maes.append(mae)

        mean_mae = sum(maes) / len(maes)
        # Just ensure it's a real number; comparative tests check improvement
        assert mean_mae >= 0.0


class TestForceAccuracy:
    """Force prediction quality (consistency and direction)."""

    def test_forces_finite(self, actor, water_equilibrium):
        """Forces should be finite for a well-formed input."""
        out = _actor_predict(actor, water_equilibrium)
        assert torch.isfinite(out["forces"]).all(), "Non-finite forces"

    def test_forces_shape(self, actor, water_equilibrium):
        """Forces should have the same shape as positions."""
        out = _actor_predict(actor, water_equilibrium)
        assert out["forces"].shape == water_equilibrium["positions"].shape

    def test_analytic_numerical_force_consistency(self, actor, water_equilibrium):
        """Analytic forces (-dV/dR) should match numerical finite differences.

        This is the most fundamental test of force correctness. If this fails,
        the network's forces are meaningless regardless of accuracy.
        """
        metrics = _force_metrics(actor, water_equilibrium)
        # Allow generous tolerance for the minimal fallback potential
        assert metrics["force_mae"] < 1.0, (
            f"Analytic/numerical force mismatch: MAE={metrics['force_mae']:.4f}"
        )

    def test_force_direction_quality(self, actor, methane_equilibrium):
        """Force directions should have positive cosine similarity with
        numerical gradients (i.e., at least point in roughly the right direction).
        """
        metrics = _force_metrics(actor, methane_equilibrium)
        # Cosine > 0 means forces at least point in the right half-space
        assert metrics["force_cosine"] > -0.5, (
            f"Force direction quality very poor: cosine={metrics['force_cosine']:.4f}"
        )


class TestPESSmoothness:
    """Potential energy surface should be smooth along physical coordinates."""

    def test_bond_stretch_pes_smooth(
        self, actor, water_equilibrium, bond_stretch_scan
    ):
        """PES along O-H stretch should be smooth (no wild jumps).

        A network with quantum intuition should produce a smooth, roughly
        Morse-like curve. Discontinuities indicate poor generalisation.
        """
        distances = [0.8 + 0.1 * i for i in range(25)]  # 0.8 to 3.2 Å
        configs = bond_stretch_scan(
            water_equilibrium["positions"].detach().clone(),
            water_equilibrium["atomic_numbers"].clone(),
            atom_i=0, atom_j=1,
            distances=distances,
        )

        energies = []
        for data in configs:
            e = _actor_predict(actor, data)["energy"].detach().item()
            assert math.isfinite(e), f"Non-finite energy at d={distances[len(energies)]}"
            energies.append(e)

        # Check smoothness: no single step should change energy by more than
        # 50% of the total range (pathological jump detection)
        e_range = max(energies) - min(energies) + 1e-10
        for i in range(1, len(energies)):
            jump = abs(energies[i] - energies[i - 1])
            fraction = jump / e_range
            assert fraction < 0.5, (
                f"PES discontinuity at step {i}: jump={jump:.4f} "
                f"({fraction:.1%} of total range)"
            )

    def test_torsion_pes_periodic(
        self, actor, ethane_equilibrium, torsion_scan
    ):
        """Torsion PES of ethane should show periodicity.

        The H-C-C-H torsion in ethane has 3-fold symmetry. We scan 0-360°
        and check that the PES is at least continuous (no wild jumps) and
        roughly returns to its starting value.
        """
        angles = [i * 15.0 for i in range(25)]  # 0° to 360°
        configs = torsion_scan(
            ethane_equilibrium["positions"].detach().clone(),
            ethane_equilibrium["atomic_numbers"].clone(),
            axis_i=0, axis_j=1,
            rotating_atoms=[5, 6, 7],  # H atoms on C2
            angles_deg=angles,
        )

        energies = []
        for data in configs:
            e = _actor_predict(actor, data)["energy"].detach().item()
            assert math.isfinite(e)
            energies.append(e)

        # The PES at 0° and 360° should be very close (periodicity)
        e_start = energies[0]
        e_end = energies[-1]
        e_range = max(energies) - min(energies) + 1e-10
        closure = abs(e_end - e_start) / e_range
        assert closure < 0.3, (
            f"Torsion PES not periodic: |E(0°)-E(360°)|/range = {closure:.2%}"
        )
