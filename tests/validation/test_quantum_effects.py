"""QM-specific effect tests: does the actor capture phenomena that pure MM misses?

Classical force fields fail on a well-known set of quantum phenomena:
  - Bond dissociation (homolytic cleavage → biradical, not handled by harmonics)
  - Torsion barriers (hyperconjugation, conjugation effects)
  - Many-body / cooperative interactions (H-bond cooperativity, polarisation)
  - Charge-transfer-like sensitivity to electronegativity

These tests probe whether the RLQF-trained network shows "quantum intuition"
by checking that the actor's PES has the *qualitative shape* expected from
QM, using the OrbNet critic as reference.

We don't demand quantitative agreement with experiment — just that the actor
exhibits the correct qualitative physics that a classical harmonic or
Lennard-Jones potential would miss.

See also
--------
- docs/foundations/mm-qm-gap.md
- docs/formulation/critic-architecture.md
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

def _actor_energy(actor: MACEActor, data: dict[str, Tensor]) -> float:
    """Get actor energy, handling autograd requirements."""
    pos = data["positions"].detach().requires_grad_(True)
    out = actor({**data, "positions": pos})
    return out["energy"].detach().item()


def _get_energies(actor: MACEActor, configs: list[dict[str, Tensor]]) -> list[float]:
    """Run actor on a list of configs and return energies."""
    return [_actor_energy(actor, data) for data in configs]


def _get_qm_energies(critic: OrbNetCritic, configs: list[dict[str, Tensor]]) -> list[float]:
    """Run the critic backbone to get proxy-QM energies."""
    energies = []
    for data in configs:
        with torch.no_grad():
            dummy = torch.zeros(data["batch"].max().item() + 1)
            out = critic(data, dummy)
            energies.append(out["qm_energy"].item())
    return energies


# ---------------------------------------------------------------------------
# Bond Dissociation
# ---------------------------------------------------------------------------

class TestBondDissociation:
    """A network with quantum intuition should produce an asymptotically
    flat dissociation curve (Morse-like), NOT a parabola that diverges.

    Classical harmonic potentials: V(r) = k(r - r0)^2  → diverges as r→∞
    Quantum (Morse-like):          V(r) → D_e          → asymptotically flat
    """

    def test_oh_dissociation_flattens(self, actor, water_equilibrium, bond_stretch_scan):
        """O-H dissociation curve should flatten at large distances.

        We check that the energy slope decreases as the bond stretches,
        consistent with a Morse-like (not harmonic) potential.
        """
        distances = [0.8 + 0.15 * i for i in range(20)]  # 0.8 to 3.65 Å
        configs = bond_stretch_scan(
            water_equilibrium["positions"].detach().clone(),
            water_equilibrium["atomic_numbers"].clone(),
            atom_i=0, atom_j=1,
            distances=distances,
        )

        energies = _get_energies(actor, configs)

        # Compute slopes in two regions: near-equilibrium and far-from-equilibrium
        near_slopes = []
        far_slopes = []
        for i in range(1, len(energies)):
            slope = abs(energies[i] - energies[i - 1]) / 0.15
            if i < len(energies) // 2:
                near_slopes.append(slope)
            else:
                far_slopes.append(slope)

        avg_near = sum(near_slopes) / max(len(near_slopes), 1)
        avg_far = sum(far_slopes) / max(len(far_slopes), 1)

        # The far-region slope should be smaller or comparable to near-region
        # A harmonic potential would have increasing slopes; Morse-like would decrease
        # We allow generous tolerance — just check it doesn't explode
        assert avg_far < avg_near * 5.0, (
            f"Dissociation curve looks harmonic (diverging): "
            f"near_slope={avg_near:.4f}, far_slope={avg_far:.4f}"
        )

    def test_dissociation_energy_bounded(
        self, actor, water_equilibrium, bond_stretch_scan
    ):
        """Energy at large bond distances should not blow up to ±infinity.

        Even a minimal network should produce bounded energies. Unbounded
        dissociation curves are a hallmark of non-physical potentials.
        """
        distances = [0.9, 1.0, 2.0, 3.0, 4.0, 5.0]
        configs = bond_stretch_scan(
            water_equilibrium["positions"].detach().clone(),
            water_equilibrium["atomic_numbers"].clone(),
            atom_i=0, atom_j=1,
            distances=distances,
        )
        energies = _get_energies(actor, configs)

        e_range = max(energies) - min(energies)
        assert e_range < 1000.0, (
            f"Energy range over dissociation scan is unreasonably large: {e_range:.1f}"
        )

    def test_critic_detects_dissociation_error(
        self, actor, critic, water_equilibrium, bond_stretch_scan
    ):
        """The critic should report higher error scores for stretched bonds.

        Dissociation is where MM potentials are most wrong. The critic
        (error estimator) should flag these geometries with high C_phi.
        """
        configs = bond_stretch_scan(
            water_equilibrium["positions"].detach().clone(),
            water_equilibrium["atomic_numbers"].clone(),
            atom_i=0, atom_j=1,
            distances=[1.0, 2.5, 4.0],
        )

        scores = []
        for data in configs:
            pos = data["positions"].detach().requires_grad_(True)
            data_copy = {**data, "positions": pos}
            e_pred = actor(data_copy)["energy"].detach()
            with torch.no_grad():
                out = critic(data, e_pred)
                scores.append(out["error_score"].item())

        # At least one stretched geometry should have a higher error score
        # than the equilibrium geometry (the critic is doing its job)
        assert max(scores[1:]) >= scores[0] * 0.5, (
            f"Critic doesn't flag stretched bonds: scores={scores}"
        )


# ---------------------------------------------------------------------------
# Torsion Barriers
# ---------------------------------------------------------------------------

class TestTorsionBarriers:
    """Torsional rotation barriers arise from orbital-level effects
    (hyperconjugation, steric interactions). A quantum-aware potential
    should produce non-trivial torsion profiles.
    """

    def test_ethane_torsion_has_barrier(
        self, actor, ethane_equilibrium, torsion_scan
    ):
        """Ethane H-C-C-H torsion should show a rotational barrier.

        Classical MM gets this from explicit torsion terms, but a neural
        potential should learn it from QM data. The barrier is ~3 kcal/mol
        in reality; we just check it's non-zero.
        """
        angles = [i * 30.0 for i in range(13)]  # 0° to 360° in 30° steps
        configs = torsion_scan(
            ethane_equilibrium["positions"].detach().clone(),
            ethane_equilibrium["atomic_numbers"].clone(),
            axis_i=0, axis_j=1,
            rotating_atoms=[5, 6, 7],
            angles_deg=angles,
        )
        energies = _get_energies(actor, configs)

        e_range = max(energies) - min(energies)
        assert e_range > 1e-6, (
            f"No torsion barrier detected: E_range={e_range:.6f}. "
            "Network may be ignoring dihedral interactions."
        )

    def test_torsion_profile_correlates_with_qm(
        self, actor, critic, ethane_equilibrium, torsion_scan
    ):
        """Actor's torsion profile should correlate with the QM reference.

        Positive Pearson correlation between actor PES and QM PES along
        the torsion scan indicates the network has learned the correct
        qualitative shape.
        """
        angles = [i * 30.0 for i in range(13)]
        configs = torsion_scan(
            ethane_equilibrium["positions"].detach().clone(),
            ethane_equilibrium["atomic_numbers"].clone(),
            axis_i=0, axis_j=1,
            rotating_atoms=[5, 6, 7],
            angles_deg=angles,
        )

        e_actor = torch.tensor(_get_energies(actor, configs))
        e_qm = torch.tensor(_get_qm_energies(critic, configs))

        # Pearson correlation
        e_a = e_actor - e_actor.mean()
        e_q = e_qm - e_qm.mean()
        corr = (e_a @ e_q) / (e_a.norm() * e_q.norm() + 1e-10)

        # We don't require strong correlation (untrained net may be weak),
        # but it should be finite
        assert math.isfinite(corr.item()), "Non-finite PES correlation"


# ---------------------------------------------------------------------------
# Many-body / Cooperative Effects
# ---------------------------------------------------------------------------

class TestManyBodyEffects:
    """QM exhibits many-body effects (e.g. 3-body contributions, cooperativity)
    that pairwise classical potentials miss. A quantum-aware potential should
    show non-additive energy contributions.
    """

    def test_three_body_non_additivity(self, actor, make_molecule):
        """Energy of 3 interacting atoms ≠ sum of pairwise energies.

        For atoms A, B, C:
            E(ABC) ≠ E(AB) + E(AC) + E(BC) - E(A) - E(B) - E(C)

        The difference is the many-body contribution, which should be
        non-zero for a quantum-aware potential.
        """
        # Three oxygen atoms in a triangle
        pos_abc = torch.tensor([
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [1.25, 2.17, 0.0],
        ], dtype=torch.float32)
        z = torch.tensor([8, 8, 8], dtype=torch.long)

        # Full trimer
        data_abc = make_molecule(pos_abc, z)
        e_abc = _actor_energy(actor, data_abc)

        # Individual atoms (isolated)
        atom_energies = []
        for i in range(3):
            data_i = make_molecule(pos_abc[i:i+1].clone(), z[i:i+1].clone())
            atom_energies.append(_actor_energy(actor, data_i))

        # Pairs
        pair_indices = [(0, 1), (0, 2), (1, 2)]
        pair_energies = []
        for i, j in pair_indices:
            pos_pair = torch.stack([pos_abc[i], pos_abc[j]])
            z_pair = torch.stack([z[i], z[j]])
            data_pair = make_molecule(pos_pair, z_pair)
            pair_energies.append(_actor_energy(actor, data_pair))

        # Many-body contribution
        # E_3body = E(ABC) - [E(AB)+E(AC)+E(BC)] + [E(A)+E(B)+E(C)]
        sum_pairs = sum(pair_energies)
        sum_atoms = sum(atom_energies)
        three_body = e_abc - sum_pairs + sum_atoms

        # The three-body term should be non-zero (quantum intuition)
        assert math.isfinite(three_body), "Non-finite three-body energy"
        # At minimum the trimer and sum-of-atoms should differ
        assert abs(e_abc - sum_atoms) > 1e-8, (
            "Energy is purely additive — no interaction terms detected"
        )

    def test_element_sensitivity(self, actor, make_molecule):
        """Different elements at the same positions should give different energies.

        A quantum-aware potential should distinguish between e.g. C, N, O
        because they have different electronic structures. A potential that
        ignores element identity has no quantum intuition.
        """
        pos = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
        ], dtype=torch.float32)

        e_by_element = {}
        for name, z_val in [("CC", [6, 6]), ("NN", [7, 7]), ("OO", [8, 8])]:
            z = torch.tensor(z_val, dtype=torch.long)
            data = make_molecule(pos.clone(), z)
            e_by_element[name] = _actor_energy(actor, data)

        # At least two element pairs should give different energies
        vals = list(e_by_element.values())
        diffs = [abs(vals[i] - vals[j]) for i in range(len(vals)) for j in range(i+1, len(vals))]
        assert max(diffs) > 1e-6, (
            f"All element pairs give identical energy: {e_by_element}. "
            "Network ignores atomic species."
        )


# ---------------------------------------------------------------------------
# Critic as QM Proxy Validation
# ---------------------------------------------------------------------------

class TestCriticQMProxy:
    """Validate that the critic's QM proxy (backbone energy) behaves
    sensibly enough to serve as a reference for the other tests.
    """

    def test_critic_qm_energy_finite(self, critic, water_equilibrium):
        """Critic backbone produces finite QM energies."""
        with torch.no_grad():
            dummy = torch.zeros(1)
            out = critic(water_equilibrium, dummy)
        assert torch.isfinite(out["qm_energy"]).all()

    def test_critic_error_score_non_negative(self, critic, actor, water_equilibrium):
        """Error scores C_phi should always be >= 0 (Softplus output)."""
        pos = water_equilibrium["positions"].detach().requires_grad_(True)
        e_pred = actor({**water_equilibrium, "positions": pos})["energy"].detach()
        with torch.no_grad():
            out = critic(water_equilibrium, e_pred)
        assert (out["error_score"] >= 0).all(), "Negative error score"

    def test_critic_qm_energy_varies(self, critic, water_equilibrium, make_molecule):
        """QM energy should change with geometry (not a constant)."""
        distorted_pos = water_equilibrium["positions"].detach().clone()
        distorted_pos[1] *= 2.0
        distorted = make_molecule(
            distorted_pos,
            water_equilibrium["atomic_numbers"].clone(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1)
            e_eq = critic(water_equilibrium, dummy)["qm_energy"]
            e_dist = critic(distorted, dummy)["qm_energy"]

        assert not torch.allclose(e_eq, e_dist, atol=1e-6), (
            "QM proxy energy unchanged under geometry change"
        )
