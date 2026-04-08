"""RLQF training entry point.

Usage:
    rlqf-train --config configs/default.yaml
    python -m rlqf.train --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys

import yaml
import torch

from rlqf.actor import MACEActor
from rlqf.actor.mace_actor import MACEActorConfig
from rlqf.critic import OrbNetCritic
from rlqf.critic.orbnet_critic import OrbNetCriticConfig
from rlqf.exploration.policy import ExplorationPolicy, PolicyConfig
from rlqf.trainer import RLQFConfig, RLQFTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_from_config(cfg: dict) -> RLQFTrainer:
    """Construct the full RLQF pipeline from a config dict."""

    # Actor (MACE)
    actor_cfg = MACEActorConfig(**cfg.get("actor", {}))
    actor = MACEActor(
        config=actor_cfg,
        pretrained=cfg.get("actor", {}).get("pretrained"),
        freeze_backbone=cfg.get("actor", {}).get("freeze_backbone", False),
    )
    logger.info(
        "Actor: MACE with %d trainable params",
        sum(p.numel() for p in actor.parameters() if p.requires_grad),
    )

    # Critic (OrbNet Denali)
    critic_cfg = OrbNetCriticConfig(**cfg.get("critic", {}))
    critic = OrbNetCritic(config=critic_cfg)
    logger.info(
        "Critic: OrbNet with %d trainable params",
        sum(p.numel() for p in critic.parameters() if p.requires_grad),
    )

    # Exploration policy
    policy_cfg = PolicyConfig(**cfg.get("policy", {}))
    policy = ExplorationPolicy(config=policy_cfg)
    logger.info(
        "Policy: Soft RLQF with %d trainable params",
        sum(p.numel() for p in policy.parameters()),
    )

    # Training config
    train_cfg = RLQFConfig(**cfg.get("training", {}))

    # Build trainer
    trainer = RLQFTrainer(
        actor=actor,
        critic=critic,
        policy=policy,
        config=train_cfg,
    )

    return trainer


def main():
    parser = argparse.ArgumentParser(description="RLQF Training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cpu, cuda, mps)",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.device:
        cfg.setdefault("training", {})["device"] = args.device

    logger.info("RLQF Training — config: %s", args.config)
    logger.info("Device: %s", cfg.get("training", {}).get("device", "cpu"))

    trainer = build_from_config(cfg)
    metrics = trainer.train()

    logger.info("Training complete. Final metrics saved.")


if __name__ == "__main__":
    main()
