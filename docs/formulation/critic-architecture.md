# The Critic Architecture

The QM critic network: how it scores MM predictions, how it is trained, and why it generalizes beyond its calibration set.

**Parent:** [[index]]
**See also:** [[verification-generation-asymmetry]], [[critic-loss]], [[mm-qm-gap]], [[convergence]]

---

## Definition

The QM critic $C_\phi: \mathcal{X} \times \mathbb{R} \to \mathbb{R}_{\geq 0}$ maps a molecular configuration and a candidate energy to a non-negative error score. It must satisfy the [[equivariance]] constraints of molecular systems.

The critic is trained via supervised regression — see [[critic-loss]] for the training objective.

## Critic Sufficiency

The following result establishes that an approximately correct critic is sufficient for the RLQF optimization to work.

**Proposition** (Critic Sufficiency). *If $C_\phi$ satisfies $|C_\phi(\mathbf{R}, \tilde{E}) - |\tilde{E} - E_0(\mathbf{R})|| \leq \epsilon_C$ uniformly over $\mathcal{X}$, then optimizing $V_\theta$ to minimize $C_\phi$ is equivalent to minimizing $\Delta(\mathbf{R}; \theta)$ up to additive error $\epsilon_C$.*

*Proof.* For any $\mathbf{R} \in \mathcal{X}$:

$$
\Delta(\mathbf{R}; \theta) - \epsilon_C \leq C_\phi(\mathbf{R}, V_\theta(\mathbf{R})) \leq \Delta(\mathbf{R}; \theta) + \epsilon_C
$$

Thus $\arg\min_\theta \mathbb{E}_\mathbf{R}[C_\phi(\mathbf{R}, V_\theta(\mathbf{R}))]$ and $\arg\min_\theta \mathbb{E}_\mathbf{R}[\Delta(\mathbf{R}; \theta)]$ coincide when $\epsilon_C \to 0$, and the optima differ by at most $\epsilon_C$ in objective value for finite $\epsilon_C$. $\square$

## The Generalization Economy

The critic's value comes from generalization: after training on a modest calibration set of QM calculations, it can score MM predictions on novel configurations without additional QM computation. This is the neural network instantiation of the [[verification-generation-asymmetry]].

The amortized cost of the critic — calibration versus inference — is analyzed in [[computational-complexity]].

## Dual Role in RLQF

In the RLQF framework, the QM critic serves double duty:

1. **Reward signal.** It provides the reward $\mathcal{R}$ in the [[mdp-formulation]] through error estimation.
2. **Value baseline.** It provides the value baseline $V^{\pi_\psi}$ in the [[policy-gradient]] through generalization over states.
3. **Importance weighting.** It derives the importance weights $w(\mathbf{R})$ in the [[energy-force-loss]].
4. **Exploration signal.** It provides the [[curiosity-reward]] and drives [[adversarial-generation]].

## Critic Drift

As the MM actor improves, the distribution of errors it produces changes, potentially invalidating the critic's training distribution. See [[open-questions]] for discussion of this non-stationarity.
