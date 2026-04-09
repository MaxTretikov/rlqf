"""Batched molecular data container."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class MolecularBatch:
    """Batch of molecular configurations with optional reference labels."""

    positions: Tensor            # (N_total, 3)
    atomic_numbers: Tensor       # (N_total,)
    edge_index: Tensor           # (2, E)
    batch: Tensor                # (N_total,)
    ref_energy: Tensor | None = None   # (B,)
    ref_forces: Tensor | None = None   # (N_total, 3)

    def to_dict(self) -> dict[str, Tensor]:
        return {
            "positions": self.positions.requires_grad_(True),
            "atomic_numbers": self.atomic_numbers,
            "edge_index": self.edge_index,
            "batch": self.batch,
        }

    def to(self, device: torch.device | str) -> MolecularBatch:
        return MolecularBatch(
            positions=self.positions.to(device),
            atomic_numbers=self.atomic_numbers.to(device),
            edge_index=self.edge_index.to(device),
            batch=self.batch.to(device),
            ref_energy=self.ref_energy.to(device) if self.ref_energy is not None else None,
            ref_forces=self.ref_forces.to(device) if self.ref_forces is not None else None,
        )

    def __len__(self) -> int:
        return int(self.batch.max().item()) + 1 if self.batch.numel() > 0 else 0
