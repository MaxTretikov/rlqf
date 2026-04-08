# Soft RLQF (Entropy-Regularized Formulation)

Maximum-entropy reinforcement learning applied to RLQF, ensuring the exploration policy discovers *diverse* failure modes of the MM network rather than collapsing to a single high-error region.

**Parent:** [[index]]
**See also:** [[policy-gradient]], [[rlqf-objective]], [[convergence]]

---

## Motivation

Standard [[policy-gradient]] methods can collapse to repeatedly sampling a single high-error region. In RLQF, we need the exploration policy to discover *all* regions where the MM network is wrong, not just the worst one. Entropy regularization prevents this mode collapse.

## The Soft RLQF Objective

We augment the [[rlqf-objective]] with an entropy bonus:

$$
J_{\text{soft}}(\psi) = \mathbb{E}_{\tau \sim \pi_\psi} \left[ \sum_{t=0}^{T} \gamma^t \Big( \mathcal{R}(s_t, a_t) + \beta \mathcal{H}(\pi_\psi(\cdot | s_t)) \Big) \right]
$$

where $\mathcal{H}(\pi_\psi(\cdot | s_t)) = -\mathbb{E}_{a \sim \pi_\psi}[\log \pi_\psi(a | s_t)]$ is the entropy of the policy and $\beta > 0$ is the temperature parameter.

## Diversity Guarantee

**Proposition** (Diversity Guarantee). *Under the soft RLQF objective, the optimal policy at state $s$ satisfies:*

$$
\pi_\psi^*(a | s) \propto \exp\left(\frac{1}{\beta} Q_{\text{soft}}^*(s, a)\right)
$$

*In particular, $\pi_\psi^*$ assigns non-zero probability to every action in $\mathcal{A}$, preventing collapse to a single failure mode of $V_\theta$.*

This is the Boltzmann policy form familiar from Soft Actor-Critic (SAC). The temperature $\beta$ controls the exploration-exploitation tradeoff: high $\beta$ favors uniform exploration; low $\beta$ concentrates on the highest-error configurations.

## Connection to KL Divergence Loss

The entropy-regularized policy has a natural connection to the [[kl-divergence-loss]]. If the reward is reframed in distributional terms — rewarding the policy for finding configurations where the MM and QM Boltzmann distributions diverge — then the soft objective directly targets regions of high KL divergence, with the entropy term ensuring coverage.

## Role in Convergence

The [[convergence]] analysis relies on the soft formulation to establish the ergodicity assumption (Assumption 4.4): the entropy bonus ensures the policy's stationary distribution has full support over $\mathcal{X}$, which is required for the coverage guarantee in the convergence proof.
