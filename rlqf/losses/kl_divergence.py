"""KL divergence loss between QM and MM Boltzmann distributions.

From docs/losses/kl-divergence-loss.md:

    L_KL = KL(p_QM || p_MM)

The gradient decomposes into a two-sample estimator (Section "Practical
Estimation" in the doc):

    grad_theta L_KL ≈ (1/kBT) * (
        E_{p_QM}[grad V_theta(R)]  -  E_{p_MM}[grad V_theta(R)]
    )

This requires samples from BOTH distributions:
  - p_QM samples: from QM MD or the exploration trajectory (labeled with E_0)
  - p_MM samples: from cheap MM MD simulations with V_theta

The partition function ratio Z_MM/Z_QM drops out of the gradient,
making this tractable.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class KLDivergenceLoss(nn.Module):
    """KL divergence between QM and MM Boltzmann distributions.

    Implements the two-sample gradient estimator from
    docs/losses/kl-divergence-loss.md. Unlike the energy-force loss,
    this targets distributional fidelity — correct thermodynamic
    ensembles — rather than pointwise energy accuracy.

    Parameters
    ----------
    beta : float
        Inverse temperature 1/(kB*T) in consistent energy units.
    """

    def __init__(self, beta: float = 1.0):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        energy_mm_at_qm_samples: Tensor,
        energy_qm_at_qm_samples: Tensor,
        energy_mm_at_mm_samples: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute the KL divergence loss.

        From docs/losses/kl-divergence-loss.md, the tractable decomposition:

            KL(p_QM || p_MM) = (1/kBT) E_{p_QM}[V_theta - E_0]
                              + log(Z_MM) - log(Z_QM)

        The gradient (Section "Practical Estimation") requires two sample sets:

            grad L_KL ≈ (1/kBT) ( <grad V_theta>_{D_QM} - <grad V_theta>_{D_MM} )

        When MM samples are not yet available (early training), falls back
        to the single-sample estimator using only the QM term.

        Parameters
        ----------
        energy_mm_at_qm_samples : (B_qm,)
            V_theta(R) evaluated at configs R ~ p_QM.
        energy_qm_at_qm_samples : (B_qm,)
            E_0(R) for the same QM-distributed configs.
        energy_mm_at_mm_samples : (B_mm,), optional
            V_theta(R') evaluated at configs R' ~ p_MM (from MM MD).
            If None, uses single-sample estimator (QM term only).

        Returns
        -------
        dict with:
          - "loss": scalar KL divergence loss (differentiable w.r.t. theta)
          - "mean_energy_diff": mean V_theta - E_0 under QM samples (detached)
          - "has_mm_samples": whether the full two-sample estimator was used
        """
        # QM term: (1/kBT) * <V_theta(R)>_{p_QM}
        # Only V_theta carries gradients; E_0 is detached reference data
        qm_term = self.beta * energy_mm_at_qm_samples.mean()

        if energy_mm_at_mm_samples is not None:
            # MM term: -(1/kBT) * <V_theta(R')>_{p_MM}
            # These are configs from MM MD — cheap to generate.
            # The gradient through this term provides the p_MM correction
            # that makes the KL gradient unbiased.
            mm_term = self.beta * energy_mm_at_mm_samples.mean()
            loss = qm_term - mm_term
            has_mm = True
        else:
            # Fallback: single-sample estimator (biased but usable early on).
            # Equivalent to just minimizing <V_theta - E_0>_{p_QM}, which
            # is the dominant term when V_theta is far from E_0.
            loss = self.beta * (energy_mm_at_qm_samples - energy_qm_at_qm_samples.detach()).mean()
            has_mm = False

        mean_diff = (energy_mm_at_qm_samples - energy_qm_at_qm_samples).mean()

        return {
            "loss": loss,
            "mean_energy_diff": mean_diff.detach(),
            "has_mm_samples": has_mm,
        }
