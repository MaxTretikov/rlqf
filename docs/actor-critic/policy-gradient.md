# Policy Gradient with Critic Baseline

Using policy gradient methods to optimize the exploration policy, with the QM critic providing both the reward signal and value baseline.

**Parent:** [[index]]
**See also:** [[rlqf-objective]], [[soft-rlqf]], [[critic-architecture]], [[mdp-formulation]]

---

## Policy Gradient Theorem for RLQF

The gradient of the [[rlqf-objective|RLQF objective]] with respect to the exploration policy parameters $\psi$ follows the policy gradient theorem:

$$
\nabla_\psi J(\psi) = \mathbb{E}_{\tau \sim \pi_\psi} \left[ \sum_{t=0}^{T} \nabla_\psi \log \pi_\psi(a_t | s_t) \cdot A^{\pi_\psi}(s_t, a_t) \right]
$$

where the advantage function is:

$$
A^{\pi_\psi}(s_t, a_t) = Q^{\pi_\psi}(s_t, a_t) - V^{\pi_\psi}(s_t)
$$

## The Critic's Dual Role

In RLQF, the QM [[critic-architecture|critic]] serves double duty: it provides both the reward signal $\mathcal{R}$ (through error estimation) and the value baseline $V^{\pi_\psi}$ (through generalization over states).

This is possible because $C_\phi(\mathbf{R}, V_\theta(\mathbf{R}))$ is a learned function that generalizes across configuration space. Its output at a given state $s_t = (\mathbf{R}_t, \theta_t)$ naturally estimates the expected future reward, since future rewards also come from the critic.

## Limitations

The standard policy gradient formulation does not guarantee diverse exploration. If the critic identifies a single high-error region early, the policy can collapse to repeatedly sampling that region, missing other failure modes of $V_\theta$. This motivates the [[soft-rlqf]] formulation, which adds an entropy bonus to ensure broad coverage.

The [[convergence]] guarantees additionally require [[ensemble-ucb|ensemble-based uncertainty augmentation]] to provide reliable exploration incentives.
