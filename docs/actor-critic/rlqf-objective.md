# The RLQF Objective

The bilevel optimization at the heart of RLQF: an outer loop that explores configuration space for maximally informative samples, and an inner loop that trains the MM network on those samples.

**Parent:** [[index]]
**See also:** [[mdp-formulation]], [[policy-gradient]], [[soft-rlqf]], [[energy-force-loss]], [[kl-divergence-loss]]

---

## Bilevel Structure

The RLQF objective is to find a policy $\pi_\psi^*$ that maximizes the expected cumulative reward over a trajectory $\tau = (s_0, a_0, s_1, a_1, \ldots)$:

$$
J(\psi) = \mathbb{E}_{\tau \sim \pi_\psi} \left[ \sum_{t=0}^{T} \gamma^t \mathcal{R}(s_t, a_t) \right]
$$

subject to the MM parameter updates induced by the trajectory. This is a *bilevel optimization*:

- **Outer loop** (exploration policy): $\max_\psi J(\psi)$ — find configurations where the MM net is most wrong.
- **Inner loop** (MM training): $\min_\theta \mathcal{L}_{\text{MM}}(\theta; \mathcal{D}_\tau)$ — train the MM net on the collected configurations.

where $\mathcal{D}_\tau = \{(\mathbf{R}_t, E_0(\mathbf{R}_t))\}_{t=0}^{T}$ is the dataset induced by the trajectory.

## Choice of Inner-Loop Loss

The inner-loop loss $\mathcal{L}_{\text{MM}}$ admits several formulations, each capturing a different aspect of the [[mm-qm-gap]]:

- **[[energy-force-loss]]** — Pointwise MSE on energies and forces, weighted by critic-derived importance. Minimizes the sup-norm gap $\Delta^*(\theta)$ with emphasis on hard configurations.
- **[[kl-divergence-loss]]** — KL divergence between QM and MM Boltzmann distributions. Minimizes distributional error, ensuring the MM surface produces correct thermodynamic ensembles.
- **Combined objective** — A weighted blend $\mathcal{L}_{\text{MM}} = \mathcal{L}_{\text{EF}} + \nu \mathcal{L}_{\text{KL}}$ that targets both pointwise accuracy and distributional fidelity.

The choice of loss function affects what the outer loop optimizes *for*: the exploration policy finds configurations that are maximally informative under the chosen loss.

## Optimization Methods

The outer loop is solved via:

- [[policy-gradient]] — Standard REINFORCE with critic baseline
- [[soft-rlqf]] — Entropy-regularized (SAC-style) formulation for diverse exploration

The inner loop uses standard gradient descent on whichever loss (or combination of losses) is selected.
