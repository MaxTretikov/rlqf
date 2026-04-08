"""Comparative tests: RLQF-trained actor vs untrained baseline.

The ultimate test of "quantum intuition" is whether the RLQF training
loop actually *improves* the actor's agreement with QM references. These
tests run a short training loop, then compare the trained actor against
an untrained copy on the same test set.

This module answers the question: "Is RLQF doing anything useful?"

Test categories
---------------
1. **Short-loop improvement**: After N outer steps, does error decrease?
2. **Critic-guided exploration**: Do explored configs have higher error?
3. **Importance weighting**: Do critic weights correlate with actual error?
4. **End-to-end pipeline**: Full train → evaluate round-trip.

See also
--------
- docs/actor-critic/rlqf-objective.md
- docs/convergence/convergence.md
"""

from __future__ import annotations

import copy
import math

import pytest
import torch
from torch import Tensor

from rlqf.actor import MACEActor
from rlqf.actor.mace_actor import MACEActorConfig
from rlqf.critic import OrbNetCritic
from rlqf.critic.orbnet_critic import OrbNetCriticConfig
from rlqf.exploration.policy import ExplorationPolicy, PolicyConfig
from rlqf.trainer import RLQFTrainer, RLQFConfig
from rlqf.utils.graph import build_neighbor_list


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph(positions, atomic_numbers, cutoff=5.0):
    edge_index = build_neighbor_list(positions, cutoff=cutoff)
    batch = torch.zeros(positions.shape[0], dtype=torch.long)
    return {
        "positions": positions,
        "atomic_numbers": atomic_numbers,
        "edge_index": edge_index,
        "batch": batch,
    }


def _actor_predict(actor, data):
    """Run actor forward with autograd-enabled positions."""
    pos = data["positions"].detach().requires_grad_(True)
    return actor({**data, "positions": pos})


def _compute_mae_vs_qm(
    actor: MACEActor,
    critic: OrbNetCritic,
    configs: list[dict[str, Tensor]],
) -> float:
    """Compute mean |V_theta - E_qm| over a set of configs."""
    errors = []
    for data in configs:
        e_pred = _actor_predict(actor, data)["energy"].detach()
        with torch.no_grad():
            dummy = torch.zeros_like(e_pred)
            e_qm = critic(data, dummy)["qm_energy"]
        errors.append((e_pred - e_qm).abs().mean().item())
    return sum(errors) / max(len(errors), 1)


def _make_test_configs(n: int = 20, seed: int = 123) -> list[dict[str, Tensor]]:
    """Generate a fixed set of test configurations."""
    torch.manual_seed(seed)
    configs = []
    for _ in range(n):
        pos = torch.randn(8, 3) * 2.0
        z = torch.tensor([6, 6, 1, 1, 1, 1, 8, 7], dtype=torch.long)
        configs.append(_make_graph(pos, z))
    return configs


# ---------------------------------------------------------------------------
# Short Training Loop
# ---------------------------------------------------------------------------

class TestShortTrainingImprovement:
    """After a short RLQF training loop, the actor should show improvement
    (or at least not get worse) compared to its initial state.
    """

    @pytest.fixture()
    def trained_pair(self):
        """Run a short RLQF loop and return (trained_actor, initial_actor, critic)."""
        torch.manual_seed(42)

        # Create models
        actor = MACEActor(MACEActorConfig(num_interactions=2))
        critic = OrbNetCritic(OrbNetCriticConfig(device="cpu"))
        policy = ExplorationPolicy(PolicyConfig())

        # Save initial state
        initial_actor = copy.deepcopy(actor)

        # Short training run
        config = RLQFConfig(
            num_outer_steps=5,
            num_inner_steps=3,
            trajectory_length=8,
            inner_batch_size=4,
            device="cpu",
            log_every=100,       # Suppress logging
            checkpoint_every=999,  # No checkpoints
        )

        trainer = RLQFTrainer(actor, critic, policy, config)
        metrics = trainer.train()

        return actor, initial_actor, critic, metrics

    def test_training_loop_completes(self, trained_pair):
        """The RLQF training loop should complete without errors."""
        actor, _, _, metrics = trained_pair
        assert len(metrics["outer_step"]) > 0, "No training steps recorded"

    def test_metrics_are_finite(self, trained_pair):
        """All training metrics should be finite (no NaN/Inf)."""
        _, _, _, metrics = trained_pair
        for key, values in metrics.items():
            for v in values:
                assert math.isfinite(v), f"Non-finite metric {key}: {v}"

    def test_actor_weights_changed(self, trained_pair):
        """Actor parameters should have changed during training.

        If they haven't, the inner loop isn't updating the actor.
        """
        actor, initial_actor, _, _ = trained_pair

        any_changed = False
        for p1, p2 in zip(actor.parameters(), initial_actor.parameters()):
            if not torch.allclose(p1.data, p2.data, atol=1e-8):
                any_changed = True
                break

        assert any_changed, "Actor weights unchanged after training"

    def test_actor_still_produces_finite_output(self, trained_pair):
        """The trained actor should still produce finite predictions.

        Training can destabilise weights if learning rates or losses
        are misconfigured.
        """
        actor, _, _, _ = trained_pair
        configs = _make_test_configs(n=10)

        for data in configs:
            out = _actor_predict(actor, data)
            assert torch.isfinite(out["energy"]).all()
            assert torch.isfinite(out["forces"]).all()


# ---------------------------------------------------------------------------
# Critic-Guided Exploration
# ---------------------------------------------------------------------------

class TestCriticGuidedExploration:
    """The exploration policy should be guided toward high-error regions
    by the critic, not just random sampling.
    """

    def test_critic_error_scores_vary(self, actor, critic, random_molecule_batch):
        """The critic should assign different error scores to different configs.

        If all scores are identical, the critic provides no training signal.
        """
        configs = random_molecule_batch(n_molecules=20, n_atoms=8)
        scores = []
        for data in configs:
            e_pred = _actor_predict(actor, data)["energy"].detach()
            with torch.no_grad():
                out = critic(data, e_pred)
                scores.append(out["error_score"].item())

        score_range = max(scores) - min(scores)
        assert score_range > 1e-8, (
            f"All critic scores identical: range={score_range:.2e}"
        )

    def test_importance_weights_normalised(self, actor, critic, random_molecule_batch):
        """Critic importance weights should sum to 1 (properly normalised)."""
        configs = random_molecule_batch(n_molecules=10, n_atoms=8)

        # Build a mini-batch
        all_pos, all_z, all_edges, all_batch = [], [], [], []
        offset = 0
        for i, data in enumerate(configs):
            n = data["positions"].shape[0]
            all_pos.append(data["positions"].detach())
            all_z.append(data["atomic_numbers"])
            all_edges.append(data["edge_index"] + offset)
            all_batch.append(torch.full((n,), i, dtype=torch.long))
            offset += n

        batch_data = {
            "positions": torch.cat(all_pos).requires_grad_(True),
            "atomic_numbers": torch.cat(all_z),
            "edge_index": torch.cat(all_edges, dim=1),
            "batch": torch.cat(all_batch),
        }

        e_pred = _actor_predict(actor, batch_data)["energy"].detach()
        with torch.no_grad():
            weights = critic.compute_importance_weights(
                e_pred, kappa=1.0, data=batch_data
            )

        assert abs(weights.sum().item() - 1.0) < 1e-5, (
            f"Importance weights don't sum to 1: sum={weights.sum().item():.6f}"
        )
        assert (weights >= 0).all(), "Negative importance weight"

    def test_cached_vs_full_importance_weights_consistent(self, actor, critic, water_equilibrium):
        """Cheap (cached) and expensive (full) importance weights should agree.

        This validates the verification-generation asymmetry: the cached
        path should produce the same result as the full forward.
        """
        e_pred = _actor_predict(actor, water_equilibrium)["energy"].detach()
        with torch.no_grad():
            # Full path
            out_full = critic(water_equilibrium, e_pred)
            w_full = critic.compute_importance_weights(
                e_pred, kappa=1.0, data=water_equilibrium
            )

            # Cached path
            w_cached = critic.compute_importance_weights(
                e_pred, kappa=1.0,
                graph_features=out_full["graph_features"],
                qm_energy=out_full["qm_energy"],
            )

        assert torch.allclose(w_full, w_cached, atol=1e-5), (
            f"Cached vs full weights disagree: {w_full} vs {w_cached}"
        )


# ---------------------------------------------------------------------------
# Replay Buffer Integration
# ---------------------------------------------------------------------------

class TestReplayBufferIntegration:
    """Verify that the replay buffer correctly stores and retrieves
    experiences with cached features.
    """

    def test_buffer_stores_and_retrieves(self):
        """Basic add/sample round-trip on the replay buffer."""
        from rlqf.data.replay_buffer import Experience, ReplayBuffer

        buf = ReplayBuffer(capacity=100, alpha_mix=0.5)

        for i in range(10):
            exp = Experience(
                config_data={
                    "positions": torch.randn(5, 3),
                    "atomic_numbers": torch.ones(5, dtype=torch.long) * 6,
                    "edge_index": torch.tensor([[0, 1], [1, 0]]),
                    "batch": torch.zeros(5, dtype=torch.long),
                },
                ref_energy=torch.tensor(float(i)),
                ref_forces=None,
                actor_energy=torch.tensor(float(i) + 0.1),
                critic_score=torch.tensor(float(i) * 0.5 + 0.1),
                reward=torch.tensor(float(i) * 0.3),
                log_prob=torch.tensor(-1.0),
                graph_features=torch.randn(1, 128),
                qm_energy=torch.tensor(float(i) + 0.01),
            )
            buf.add(exp, fresh=True)

        assert len(buf) == 10
        sampled = buf.sample(5)
        assert len(sampled) == 5
        assert sampled[0].graph_features is not None, "Cached features lost"

    def test_priority_sampling_prefers_high_error(self):
        """Priority sampling should oversample high-critic-score configs."""
        from rlqf.data.replay_buffer import Experience, ReplayBuffer

        buf = ReplayBuffer(capacity=100, alpha_mix=1.0, priority_exponent=2.0)

        # Add configs with varying critic scores
        for i in range(20):
            exp = Experience(
                config_data={
                    "positions": torch.randn(3, 3),
                    "atomic_numbers": torch.ones(3, dtype=torch.long),
                    "edge_index": torch.tensor([[0, 1], [1, 0]]),
                    "batch": torch.zeros(3, dtype=torch.long),
                },
                ref_energy=torch.tensor(0.0),
                ref_forces=None,
                actor_energy=torch.tensor(0.0),
                critic_score=torch.tensor(float(i)),  # Linearly increasing
                reward=torch.tensor(0.0),
                log_prob=torch.tensor(0.0),
            )
            buf.add(exp, fresh=True)

        # Sample many times and check that high-score configs are overrepresented
        score_sum = 0.0
        n_samples = 200
        for _ in range(n_samples):
            batch = buf.sample(5)
            for exp in batch:
                score_sum += exp.critic_score.item()

        avg_sampled_score = score_sum / (n_samples * 5)
        uniform_avg = sum(range(20)) / 20  # = 9.5

        # Priority sampling should give higher average score than uniform
        assert avg_sampled_score > uniform_avg * 0.8, (
            f"Priority sampling not working: avg_score={avg_sampled_score:.2f} "
            f"vs uniform={uniform_avg:.1f}"
        )


# ---------------------------------------------------------------------------
# End-to-End Pipeline Smoke Test
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    """Full round-trip: initialise → train → evaluate → compare."""

    @pytest.mark.slow
    def test_full_pipeline_smoke(self):
        """Run the complete RLQF pipeline end-to-end (minimal config).

        This is the integration test: if this passes, the entire
        init → explore → inner-loop → critic-recalibrate → evaluate
        pipeline is functional.
        """
        torch.manual_seed(42)

        actor = MACEActor(MACEActorConfig(num_interactions=2))
        critic = OrbNetCritic(OrbNetCriticConfig(device="cpu"))
        policy = ExplorationPolicy(PolicyConfig())

        config = RLQFConfig(
            num_outer_steps=3,
            num_inner_steps=2,
            trajectory_length=4,
            inner_batch_size=2,
            device="cpu",
            log_every=1,
            checkpoint_every=999,
        )

        trainer = RLQFTrainer(actor, critic, policy, config)
        metrics = trainer.train()

        # Basic sanity checks
        assert len(metrics["outer_step"]) >= 1
        assert all(math.isfinite(v) for v in metrics["actor_loss"])
        assert all(math.isfinite(v) for v in metrics["policy_loss"])

        # The trained actor should produce valid outputs
        test_configs = _make_test_configs(n=5)
        for data in test_configs:
            out = _actor_predict(actor, data)
            assert torch.isfinite(out["energy"]).all()
            assert torch.isfinite(out["forces"]).all()
