# Distribution Shift Mitigation

Preventing catastrophic forgetting on physically important configurations when the exploration policy focuses on adversarially-selected hard cases.

**Parent:** [[index]]
**See also:** [[adversarial-generation]], [[curiosity-reward]], [[kl-divergence-loss]], [[energy-force-loss]]

---

## The Problem

Pure adversarial sampling (via [[adversarial-generation]] or aggressive [[curiosity-reward]]-driven exploration) leads to distribution shift: the MM network may degrade on "easy" configurations while improving on hard ones. This is particularly problematic because the "easy" configurations — equilibrium geometries, common conformations — are often the ones that matter most for practical molecular dynamics simulations.

## Mixed Replay Buffer

We address this via a mixed replay buffer:

$$
\mathcal{D}_{\text{train}} = (1 - \alpha) \cdot \mathcal{D}_{\text{Boltzmann}} + \alpha \cdot \mathcal{D}_{\text{adversarial}}
$$

where $\mathcal{D}_{\text{Boltzmann}}$ contains configurations sampled from a physical Boltzmann distribution at temperature $T_{\text{phys}}$ and $\mathcal{D}_{\text{adversarial}}$ contains configurations selected by the exploration policy. The mixing ratio $\alpha \in [0, 1]$ trades off accuracy on physically relevant configurations against worst-case robustness.

## Connection to KL Divergence Loss

The [[kl-divergence-loss]] provides a principled alternative to ad-hoc mixing. Because the KL divergence objective is defined over the Boltzmann distribution $p_{\text{QM}}$, it naturally weights configurations by their thermodynamic importance. Combining the KL loss with [[energy-force-loss|energy-force loss]] on adversarial samples achieves both distributional fidelity and worst-case robustness without requiring a hand-tuned mixing ratio.

## Practical Guidance

- $\alpha \approx 0.7$ (70% adversarial, 30% Boltzmann) is a reasonable starting point
- The mixing ratio can be adapted during training based on validation loss on a held-out Boltzmann-sampled set
- The Boltzmann samples can be generated cheaply via MM molecular dynamics, since they don't require QM labels for the mixing step (only for computing the [[energy-force-loss]] when they enter the training batch)
