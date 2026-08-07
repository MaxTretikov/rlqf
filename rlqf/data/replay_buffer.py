"""Replay buffer with fresh/historical mixing and priority sampling."""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Experience:
    """Single trajectory step stored for training and policy updates."""

    config_data: dict[str, Tensor]
    ref_energy: Tensor
    ref_forces: Tensor | None = None
    actor_energy: Tensor | None = None
    critic_score: Tensor | None = None
    reward: Tensor | None = None
    log_prob: Tensor | None = None
    graph_features: Tensor | None = None
    qm_energy: Tensor | None = None
    state: Tensor | None = None
    action: Tensor | None = None
    value: Tensor | None = None


class ReplayBuffer:
    """Two-pool replay buffer (fresh + historical) with optional priority sampling.

    Fresh experiences come from the current outer step and are cleared after
    each inner loop. Historical experiences persist up to capacity.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha_mix: float = 0.5,
        priority_exponent: float = 0.0,
    ):
        self.capacity = capacity
        self.alpha_mix = alpha_mix
        self.priority_exponent = priority_exponent
        self._historical: list[Experience] = []
        self._fresh: list[Experience] = []

    def __len__(self) -> int:
        return len(self._historical)

    def add(self, experience: Experience, fresh: bool = False) -> None:
        self._historical.append(experience)
        if fresh:
            self._fresh.append(experience)
        if len(self._historical) > self.capacity:
            self._historical = self._historical[-self.capacity:]

    def add_batch(self, experiences: list[Experience], fresh: bool = False) -> None:
        for exp in experiences:
            self.add(exp, fresh=fresh)

    def sample(self, batch_size: int) -> list[Experience]:
        if len(self) == 0:
            return []

        n_fresh = 0
        if self._fresh and self.alpha_mix > 0:
            n_fresh = min(int(batch_size * self.alpha_mix), len(self._fresh))
        n_hist = min(batch_size - n_fresh, len(self._historical))

        if n_hist < batch_size - n_fresh:
            n_fresh = min(batch_size - n_hist, len(self._fresh))
        if n_fresh < int(batch_size * self.alpha_mix) and len(self._historical) > n_hist:
            n_hist = min(batch_size - n_fresh, len(self._historical))

        result = []
        if n_fresh > 0:
            result.extend(random.choices(self._fresh, k=n_fresh))
        if n_hist > 0:
            if self.priority_exponent > 0:
                result.extend(self._priority_sample(self._historical, n_hist))
            else:
                result.extend(random.choices(self._historical, k=n_hist))
        return result

    def _priority_sample(self, pool: list[Experience], n: int) -> list[Experience]:
        scores = []
        for exp in pool:
            s = exp.critic_score.item() if exp.critic_score is not None else 0.0
            scores.append(max(s, 1e-8))
        weights = [s ** self.priority_exponent for s in scores]
        return random.choices(pool, weights=weights, k=n)

    def clear_fresh(self) -> None:
        self._fresh.clear()
