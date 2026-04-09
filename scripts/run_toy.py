#!/usr/bin/env python3
"""Run RLQF end-to-end on a small molecule using fallback MLPs.

Usage:
    python -m scripts.run_toy
"""

from __future__ import annotations

import logging
import time

import torch

from rlqf.actor import MACEActor
from rlqf.actor.mace_actor import MACEActorConfig
from rlqf.critic import OrbNetCritic
from rlqf.critic.orbnet_critic import OrbNetCriticConfig
from rlqf.exploration.policy import ExplorationPolicy, PolicyConfig
from rlqf.trainer import RLQFConfig, RLQFTrainer
from rlqf.utils.graph import build_neighbor_list

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def make_water():
    positions = torch.tensor([
        [0.000, 0.000, 0.000],   # O
        [0.757, 0.586, 0.000],   # H
        [-0.757, 0.586, 0.000],  # H
    ], dtype=torch.float32)
    return positions, torch.tensor([8, 1, 1], dtype=torch.long)


def make_methane():
    import math
    r = 1.089 / math.sqrt(3.0)
    positions = torch.tensor([
        [0.0, 0.0, 0.0],
        [r, r, r],
        [r, -r, -r],
        [-r, r, -r],
        [-r, -r, r],
    ], dtype=torch.float32)
    return positions, torch.tensor([6, 1, 1, 1, 1], dtype=torch.long)


def evaluate_actor(actor, critic, label):
    molecules = [make_water(), make_methane()]
    errors = []
    for positions, atomic_numbers in molecules:
        edge_index = build_neighbor_list(positions, cutoff=5.0)
        data = {
            "positions": positions.requires_grad_(True),
            "atomic_numbers": atomic_numbers,
            "edge_index": edge_index,
            "batch": torch.zeros(positions.shape[0], dtype=torch.long),
        }
        e_actor = actor(data)["energy"].detach()
        with torch.no_grad():
            e_critic = critic(data, torch.zeros_like(e_actor))["qm_energy"]
        errors.append((e_actor - e_critic).abs().mean().item())
    mae = sum(errors) / len(errors)
    logger.info("%s — MAE = %.6f", label, mae)
    return mae


def main():
    torch.manual_seed(42)

    actor = MACEActor(MACEActorConfig(num_interactions=2))
    critic = OrbNetCritic(OrbNetCriticConfig(device="cpu"))
    policy = ExplorationPolicy(PolicyConfig(state_dim=256, action_dim=64))

    logger.info(
        "Params — actor: %d, critic: %d, policy: %d",
        sum(p.numel() for p in actor.parameters() if p.requires_grad),
        sum(p.numel() for p in critic.parameters() if p.requires_grad),
        sum(p.numel() for p in policy.parameters()),
    )

    mae_before = evaluate_actor(actor, critic, "Before")

    config = RLQFConfig(
        num_outer_steps=30,
        trajectory_length=8,
        num_inner_steps=5,
        inner_batch_size=4,
        mu=10.0,
        nu=0.0,
        actor_lr=1e-3,
        log_every=5,
        checkpoint_every=9999,
    )

    trainer = RLQFTrainer(actor, critic, policy, config)
    positions, atomic_numbers = make_water()
    trainer.set_initial_molecule(positions, atomic_numbers)

    t0 = time.time()
    metrics = trainer.train()
    logger.info("Done in %.1fs", time.time() - t0)

    mae_after = evaluate_actor(actor, critic, "After")

    if mae_after < mae_before:
        logger.info("Improved by %.1f%%", 100 * (1 - mae_after / mae_before))

    for step, loss in zip(metrics["outer_step"], metrics["actor_loss"]):
        bar = "#" * max(1, min(50, int(loss * 100)))
        logger.info("  Step %3d | loss=%.6f | %s", step, loss, bar)


if __name__ == "__main__":
    main()
