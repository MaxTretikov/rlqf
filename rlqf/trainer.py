"""RLQF Trainer: the bilevel optimization training loop.

Implements the core RLQF algorithm from docs/actor-critic/rlqf-objective.md:

    Outer loop (exploration): max_psi J(psi) — find configurations where
        the MM net is most wrong.
    Inner loop (MM training): min_theta L_MM(theta; D_tau) — train the
        MM net on the collected configurations.

The critic is periodically recalibrated (docs/losses/critic-loss.md).

Algorithm sketch (from docs/rlqf.md):
    1. Initialize actor V_theta, critic C_phi, policy pi_psi
    2. For each outer step:
       a. Collect trajectory tau using pi_psi (exploration)
       b. For each configuration in tau:
          - Compute actor prediction V_theta(R)
          - Compute critic reward C_phi(R, V_theta(R))
       c. Update policy pi_psi via policy gradient with critic baseline
       d. For each inner step:
          - Sample batch from replay buffer
          - Compute importance weights w(R) from critic
          - Update actor theta via energy-force loss
       e. Periodically recalibrate critic phi
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam, Optimizer
from tqdm import tqdm

from rlqf.actor import MACEActor
from rlqf.critic import OrbNetCritic
from rlqf.data.molecular_batch import MolecularBatch
from rlqf.data.replay_buffer import Experience, ReplayBuffer
from rlqf.exploration.policy import ExplorationPolicy
from rlqf.losses.critic_loss import CriticLoss
from rlqf.losses.energy_force import EnergyForceLoss
from rlqf.losses.kl_divergence import KLDivergenceLoss
from rlqf.utils.graph import build_neighbor_list, encode_state

logger = logging.getLogger(__name__)


@dataclass
class RLQFConfig:
    """Configuration for the RLQF training loop.

    Notation follows docs/meta/notation.md throughout.
    """

    # === Outer loop (exploration) ===
    num_outer_steps: int = 1000          # Total outer-loop iterations
    trajectory_length: int = 32          # Steps per trajectory (T in the MDP)
    gamma: float = 0.99                  # Discount factor
    lambda_reg: float = 0.01             # Configuration jump penalty (lambda)
    clip_epsilon: float = 0.2            # PPO clip parameter

    # === Inner loop (MM training) ===
    num_inner_steps: int = 10            # Gradient steps per outer step
    inner_batch_size: int = 32           # Batch size for inner-loop updates
    mu: float = 100.0                    # Energy/force loss balance
    kappa: float = 1.0                   # Importance weight exponent
    nu: float = 0.1                      # KL/energy-force mixing coefficient

    # === Critic ===
    critic_recalibrate_every: int = 50   # Recalibrate critic every N outer steps
    critic_recalibrate_steps: int = 100  # Gradient steps per recalibration

    # === Optimization ===
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    policy_lr: float = 3e-4
    max_grad_norm: float = 1.0

    # === Replay buffer ===
    buffer_capacity: int = 100_000
    alpha_mix: float = 0.5               # Fresh vs historical mixing ratio

    # === Entropy (Soft RLQF) ===
    beta: float = 0.1                    # Entropy temperature

    # === Infrastructure ===
    device: str = "cpu"
    log_every: int = 10
    checkpoint_every: int = 100
    checkpoint_dir: str = "checkpoints"
    seed: int = 42


class RLQFTrainer:
    """Bilevel optimization trainer for RLQF.

    Orchestrates the outer-loop exploration policy, inner-loop MM training,
    and periodic critic recalibration.

    Parameters
    ----------
    actor : MACEActor
        The MM neural network potential (V_theta).
    critic : OrbNetCritic
        The QM error estimator (C_phi).
    policy : ExplorationPolicy
        The exploration policy (pi_psi).
    config : RLQFConfig
        Training hyperparameters.
    calibration_data : list[MolecularBatch], optional
        QM calibration data for critic training.
    """

    def __init__(
        self,
        actor: MACEActor,
        critic: OrbNetCritic,
        policy: ExplorationPolicy,
        config: RLQFConfig | None = None,
        calibration_data: list[MolecularBatch] | None = None,
    ):
        self.actor = actor
        self.critic = critic
        self.policy = policy
        self.config = config or RLQFConfig()
        self.calibration_data = calibration_data or []

        self.device = torch.device(self.config.device)
        self._setup_components()

    def _setup_components(self) -> None:
        """Initialize optimizers, losses, buffer, and move to device."""
        # Move models to device
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.policy.to(self.device)

        # Optimizers
        self.actor_optim = Adam(
            [p for p in self.actor.parameters() if p.requires_grad],
            lr=self.config.actor_lr,
        )
        self.critic_optim = Adam(
            [p for p in self.critic.parameters() if p.requires_grad],
            lr=self.config.critic_lr,
        )
        self.policy_optim = Adam(
            self.policy.parameters(),
            lr=self.config.policy_lr,
        )

        # Loss functions
        self.ef_loss_fn = EnergyForceLoss(mu=self.config.mu, kappa=self.config.kappa)
        self.critic_loss_fn = CriticLoss()
        self.kl_loss_fn = KLDivergenceLoss()

        # Replay buffer
        self.buffer = ReplayBuffer(
            capacity=self.config.buffer_capacity,
            alpha_mix=self.config.alpha_mix,
        )

        # Metrics tracking
        self.metrics: dict[str, list[float]] = {
            "outer_step": [],
            "policy_loss": [],
            "actor_loss": [],
            "critic_loss": [],
            "mean_reward": [],
            "mean_error": [],
            "policy_entropy": [],
        }

    def train(self) -> dict[str, list[float]]:
        """Run the full RLQF training loop.

        Returns
        -------
        metrics : dict of training metrics over time.
        """
        logger.info("Starting RLQF training: %d outer steps", self.config.num_outer_steps)
        torch.manual_seed(self.config.seed)

        for outer_step in tqdm(range(self.config.num_outer_steps), desc="RLQF"):
            # === OUTER LOOP: Exploration ===
            trajectory = self._collect_trajectory()

            # Update exploration policy via policy gradient
            policy_metrics = self._update_policy(trajectory)

            # Add experiences to replay buffer
            self.buffer.add_batch(trajectory, fresh=True)

            # === INNER LOOP: MM Training ===
            actor_metrics = self._inner_loop()

            # Clear fresh buffer after inner loop uses it
            self.buffer.clear_fresh()

            # === CRITIC RECALIBRATION ===
            critic_metrics = {}
            if (
                outer_step > 0
                and outer_step % self.config.critic_recalibrate_every == 0
                and self.calibration_data
            ):
                critic_metrics = self._recalibrate_critic()

            # === LOGGING ===
            if outer_step % self.config.log_every == 0:
                self._log_step(outer_step, policy_metrics, actor_metrics, critic_metrics)

            # === CHECKPOINTING ===
            if outer_step % self.config.checkpoint_every == 0:
                self._save_checkpoint(outer_step)

        logger.info("RLQF training complete.")
        return self.metrics

    def _collect_trajectory(self) -> list[Experience]:
        """Collect a trajectory using the exploration policy.

        Implements the outer-loop rollout: at each step, the policy
        proposes a configuration perturbation, the actor evaluates it,
        and the critic scores the error.
        """
        self.actor.eval()
        self.critic.eval()
        self.policy.eval()

        trajectory: list[Experience] = []

        # Initialize with a random configuration or last known state
        positions = torch.randn(10, 3, device=self.device) * 2.0  # Placeholder init
        atomic_numbers = torch.ones(10, dtype=torch.long, device=self.device) * 6  # Carbon
        batch_idx = torch.zeros(10, dtype=torch.long, device=self.device)

        for t in range(self.config.trajectory_length):
            # Build graph from current configuration R_t
            edge_index = build_neighbor_list(positions, cutoff=5.0)
            data = {
                "positions": positions.requires_grad_(True),
                "atomic_numbers": atomic_numbers,
                "edge_index": edge_index,
                "batch": batch_idx,
            }

            # Encode state s_t = (R_t, theta_t) from graph features
            # Actor prediction at current config (no grad for rollout)
            self.actor.eval()
            actor_out = self.actor(data)
            energy_pred_t = actor_out["energy"].detach()

            # Critic at current config (FULL forward = C_QM)
            with torch.no_grad():
                critic_out_t = self.critic(data, energy_pred_t)

            # Policy: propose next configuration R_{t+1} = R_t + displacement
            graph_feat = critic_out_t["graph_features"]  # (1, D)
            state = encode_state(graph_feat, target_dim=self.policy.config.state_dim)
            policy_out = self.policy.sample(state)
            value_t = policy_out["value"].detach()

            # Action = perturbation applied to positions
            action = policy_out["action"]  # (1, action_dim)
            n_atoms = positions.shape[0]
            if action.shape[-1] >= n_atoms * 3:
                displacement = action[0, : n_atoms * 3].reshape(n_atoms, 3)
            else:
                displacement = torch.zeros(n_atoms, 3, device=self.device)
                displacement.view(-1)[: action.shape[-1]] = action[0]

            # Transition to R_{t+1}
            next_positions = (positions.detach() + displacement.detach())

            # Evaluate critic at R_{t+1} — the reward depends on the NEW
            # config per docs/formulation/mdp-formulation.md:
            #   R(s_t, a_t) = C_phi(R_{t+1}, V_theta(R_{t+1})) - lambda * d(R_{t+1}, R_t)
            next_edge_index = build_neighbor_list(next_positions, cutoff=5.0)
            next_data = {
                "positions": next_positions.requires_grad_(True),
                "atomic_numbers": atomic_numbers,
                "edge_index": next_edge_index,
                "batch": batch_idx,
            }
            # Actor needs autograd for force computation even in eval mode,
            # so don't wrap it in no_grad. Critic can use no_grad.
            next_actor_out = self.actor(next_data)
            next_energy = next_actor_out["energy"].detach()
            with torch.no_grad():
                next_critic_out = self.critic(next_data, next_energy)

            # Reward per MDP formulation (evaluated at R_{t+1}, not R_t)
            config_distance = displacement.detach().norm()
            reward = (
                next_critic_out["error_score"]
                - self.config.lambda_reg * config_distance
            )

            # Store experience with cached features for R_{t+1}
            # (these are what the inner loop trains on)
            exp = Experience(
                config_data={k: v.detach() for k, v in next_data.items()},
                ref_energy=next_critic_out["qm_energy"].detach(),
                ref_forces=next_critic_out.get("qm_forces"),
                actor_energy=next_energy.detach(),
                critic_score=next_critic_out["error_score"].detach(),
                reward=reward.detach(),
                log_prob=policy_out["log_prob"].detach(),
                graph_features=next_critic_out["graph_features"].detach(),
                qm_energy=next_critic_out["qm_energy"].detach(),
                state=state.detach(),           # for policy re-evaluation
                action=action.detach(),         # for policy re-evaluation
                value=value_t,                  # V^pi(s_t) for advantage
            )
            trajectory.append(exp)

            # Advance to R_{t+1}
            positions = next_positions

        self.actor.train()
        self.critic.train()
        self.policy.train()

        return trajectory

    def _update_policy(self, trajectory: list[Experience]) -> dict[str, float]:
        """Update the exploration policy via policy gradient.

        Implements the advantage-based policy gradient from
        docs/actor-critic/policy-gradient.md:

            grad_psi J = E[ sum_t grad log pi(a_t|s_t) * A(s_t, a_t) ]

        where A(s_t, a_t) = G_t - V^pi(s_t) uses the value baseline from
        the policy network's value head (not normalized returns).

        Uses PPO clipping (docs/actor-critic/soft-rlqf.md) with entropy
        regularization for diverse exploration.
        """
        if not trajectory:
            return {}

        rewards = torch.stack([exp.reward for exp in trajectory])

        # --- Discounted returns G_t = sum_{k=0}^{T-t} gamma^k r_{t+k} ---
        returns = torch.zeros_like(rewards)
        running_return = 0.0
        for t in reversed(range(len(rewards))):
            running_return = rewards[t] + self.config.gamma * running_return
            returns[t] = running_return

        # --- Advantage: A_t = G_t - V^pi(s_t) per policy-gradient.md ---
        # Uses the value baseline recorded during trajectory collection,
        # NOT normalized returns (which would lose the critic signal).
        values = torch.stack([exp.value for exp in trajectory])
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # --- Re-evaluate trajectory under CURRENT policy parameters ---
        # Uses the ACTUAL states and actions from the trajectory (stored
        # in Experience), not random tensors.
        old_log_probs = torch.stack([exp.log_prob for exp in trajectory])
        states = torch.stack([exp.state.squeeze(0) for exp in trajectory])
        actions = torch.stack([exp.action.squeeze(0) for exp in trajectory])

        eval_out = self.policy.evaluate(states, actions)
        new_log_probs = eval_out["log_prob"]
        new_values = eval_out["value"]
        entropies = eval_out["entropy"]

        # --- Policy loss (PPO + entropy from soft-rlqf.md) ---
        loss_out = self.policy.compute_policy_loss(
            log_probs=new_log_probs,
            advantages=advantages,
            entropies=entropies,
            old_log_probs=old_log_probs,
            clip_epsilon=self.config.clip_epsilon,
        )

        # --- Value loss: train V^pi(s) to predict returns ---
        value_loss = (new_values - returns).pow(2).mean()

        total_loss = loss_out["loss"] + 0.5 * value_loss

        self.policy_optim.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
        self.policy_optim.step()

        return {
            "policy_loss": loss_out["loss"].item(),
            "value_loss": value_loss.item(),
            "entropy": loss_out["entropy"].item(),
            "mean_reward": rewards.mean().item(),
        }

    def _inner_loop(self) -> dict[str, float]:
        """Inner loop: train the MM actor on collected data.

        Performs num_inner_steps gradient updates on the energy-force loss
        with critic-derived importance weights.

        Returns
        -------
        dict of averaged inner-loop metrics.
        """
        if len(self.buffer) == 0:
            return {}

        total_loss = 0.0
        total_e_loss = 0.0
        total_f_loss = 0.0

        for step in range(self.config.num_inner_steps):
            # Sample from replay buffer (mixed fresh + historical)
            experiences = self.buffer.sample(self.config.inner_batch_size)
            if not experiences:
                continue

            # Build batch from experiences (includes cached backbone features)
            batch_data, ref_energies, cached_features, cached_qm_e = (
                self._experiences_to_batch(experiences)
            )

            # Compute actor predictions (with gradients) — cost: C_MM
            actor_out = self.actor(batch_data)
            energy_pred = actor_out["energy"]
            forces_pred = actor_out["forces"]

            # Compute importance weights via CHEAP verification oracle.
            # Uses cached backbone features from exploration — cost: ~C_error_head
            # This is where the verification-generation asymmetry pays off:
            # we do NOT re-run the expensive OrbNet backbone here.
            if cached_features is not None and cached_qm_e is not None:
                # CHEAP PATH: verification oracle only (~C_MM)
                importance_weights = self.critic.compute_importance_weights(
                    energy_pred.detach(),
                    kappa=self.config.kappa,
                    graph_features=cached_features,
                    qm_energy=cached_qm_e,
                )
            else:
                # EXPENSIVE FALLBACK: full forward pass (~C_QM)
                # Only used for very old buffer entries where cache was evicted
                importance_weights = self.critic.compute_importance_weights(
                    energy_pred.detach(),
                    kappa=self.config.kappa,
                    data=batch_data,
                )

            # Energy-force loss — uses QM labels obtained during exploration
            ef_out = self.ef_loss_fn(
                energy_pred=energy_pred,
                forces_pred=forces_pred,
                energy_ref=ref_energies,
                forces_ref=torch.zeros_like(forces_pred),  # Use QM forces when available
                batch=batch_data["batch"],
                importance_weights=importance_weights,
            )

            # Optional KL divergence loss (docs/losses/kl-divergence-loss.md).
            # Uses cached QM energies as E_0 (no extra backbone cost).
            # Currently single-sample estimator (QM term only); full two-sample
            # estimator requires MM MD sampling (see KLDivergenceLoss docstring).
            kl_loss = torch.tensor(0.0, device=self.device)
            if self.config.nu > 0 and cached_qm_e is not None:
                kl_out = self.kl_loss_fn(
                    energy_mm_at_qm_samples=energy_pred,
                    energy_qm_at_qm_samples=cached_qm_e,
                    energy_mm_at_mm_samples=None,  # TODO: add MM MD sampling
                )
                kl_loss = kl_out["loss"]

            # Combined inner-loop loss
            loss = ef_out["loss"] + self.config.nu * kl_loss

            self.actor_optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in self.actor.parameters() if p.requires_grad],
                self.config.max_grad_norm,
            )
            self.actor_optim.step()

            total_loss += loss.item()
            total_e_loss += ef_out["energy_loss"].item()
            total_f_loss += ef_out["force_loss"].item()

        n = max(self.config.num_inner_steps, 1)
        return {
            "actor_loss": total_loss / n,
            "energy_loss": total_e_loss / n,
            "force_loss": total_f_loss / n,
        }

    def _recalibrate_critic(self) -> dict[str, float]:
        """Recalibrate the critic on fresh calibration data.

        From docs/losses/critic-loss.md: the critic must be periodically
        recalibrated as the actor improves, since the error distribution shifts.
        """
        logger.info("Recalibrating critic...")
        total_loss = 0.0

        for step in range(self.config.critic_recalibrate_steps):
            for cal_batch in self.calibration_data:
                data = cal_batch.to_dict()

                # Actor prediction (detached — no gradients through actor here)
                with torch.no_grad():
                    actor_out = self.actor(data)
                energy_pred = actor_out["energy"]

                # Critic prediction
                critic_out = self.critic(data, energy_pred)

                # Critic loss: predict |V_theta(R) - E_0(R)|
                loss_out = self.critic_loss_fn(
                    error_score=critic_out["error_score"],
                    energy_pred=energy_pred,
                    energy_ref=cal_batch.ref_energy.to(self.device),
                )

                self.critic_optim.zero_grad()
                loss_out["loss"].backward()
                nn.utils.clip_grad_norm_(
                    [p for p in self.critic.parameters() if p.requires_grad],
                    self.config.max_grad_norm,
                )
                self.critic_optim.step()

                total_loss += loss_out["loss"].item()

        n = max(self.config.critic_recalibrate_steps * len(self.calibration_data), 1)
        logger.info("Critic recalibration complete. Mean loss: %.6f", total_loss / n)
        return {"critic_loss": total_loss / n}

    def _experiences_to_batch(
        self, experiences: list[Experience]
    ) -> tuple[dict[str, Tensor], Tensor, Tensor | None, Tensor | None]:
        """Convert a list of experiences into batched tensors.

        Returns cached backbone features alongside the molecular data,
        enabling the cheap verification path in the inner loop.

        Returns
        -------
        batch_data : dict of batched molecular configuration tensors.
        ref_energies : (B,) QM reference energies (from generation oracle).
        cached_features : (B, D) cached backbone features, or None.
        cached_qm_energies : (B,) cached QM energies, or None.
        """
        all_positions = []
        all_atomic_numbers = []
        all_edges = []
        all_batch = []
        ref_energies = []
        cached_features = []
        cached_qm_energies = []
        has_cache = True
        offset = 0

        for i, exp in enumerate(experiences):
            pos = exp.config_data["positions"]
            z = exp.config_data["atomic_numbers"]
            edges = exp.config_data["edge_index"]
            n = pos.shape[0]

            all_positions.append(pos)
            all_atomic_numbers.append(z)
            all_edges.append(edges + offset)
            all_batch.append(torch.full((n,), i, dtype=torch.long, device=pos.device))
            ref_energies.append(exp.ref_energy)

            if exp.graph_features is not None and exp.qm_energy is not None:
                cached_features.append(exp.graph_features)
                cached_qm_energies.append(exp.qm_energy)
            else:
                has_cache = False

            offset += n

        batch_data = {
            "positions": torch.cat(all_positions, dim=0),
            "atomic_numbers": torch.cat(all_atomic_numbers, dim=0),
            "edge_index": torch.cat(all_edges, dim=1),
            "batch": torch.cat(all_batch, dim=0),
        }

        ref_e = torch.stack(ref_energies)

        # Return cached features if ALL experiences in the batch have them
        if has_cache and cached_features:
            feat = torch.cat(cached_features, dim=0)   # (B, D) — cat along batch dim
            qm_e = torch.cat(cached_qm_energies, dim=0)  # (B,)
            return batch_data, ref_e, feat, qm_e
        else:
            return batch_data, ref_e, None, None

    def _log_step(
        self,
        step: int,
        policy_metrics: dict[str, float],
        actor_metrics: dict[str, float],
        critic_metrics: dict[str, float],
    ) -> None:
        """Log metrics for the current step."""
        self.metrics["outer_step"].append(step)
        self.metrics["policy_loss"].append(policy_metrics.get("policy_loss", 0.0))
        self.metrics["actor_loss"].append(actor_metrics.get("actor_loss", 0.0))
        self.metrics["critic_loss"].append(critic_metrics.get("critic_loss", 0.0))
        self.metrics["mean_reward"].append(policy_metrics.get("mean_reward", 0.0))
        self.metrics["mean_error"].append(actor_metrics.get("energy_loss", 0.0))
        self.metrics["policy_entropy"].append(policy_metrics.get("entropy", 0.0))

        logger.info(
            "Step %d | policy_loss=%.4f actor_loss=%.4f reward=%.4f entropy=%.4f",
            step,
            policy_metrics.get("policy_loss", 0.0),
            actor_metrics.get("actor_loss", 0.0),
            policy_metrics.get("mean_reward", 0.0),
            policy_metrics.get("entropy", 0.0),
        )

    def _save_checkpoint(self, step: int) -> None:
        """Save a training checkpoint."""
        import os

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.config.checkpoint_dir, f"rlqf_step_{step}.pt")
        torch.save(
            {
                "step": step,
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "policy": self.policy.state_dict(),
                "actor_optim": self.actor_optim.state_dict(),
                "critic_optim": self.critic_optim.state_dict(),
                "policy_optim": self.policy_optim.state_dict(),
                "config": self.config,
                "metrics": self.metrics,
            },
            path,
        )
        logger.info("Checkpoint saved: %s", path)
