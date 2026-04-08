# Convergence Properties and Error Bounds

Main convergence theorem for RLQF, proof sketch, sample complexity, and comparison with passive learning.

**Parent:** [[index]]
**See also:** [[rlqf-objective]], [[soft-rlqf]], [[computational-complexity]], [[equivariance]]

---

## Assumptions

We require the following regularity conditions.

**Assumption 4.1** (Lipschitz Continuity). *The true PES $E_0$ is $L_E$-Lipschitz on $\mathcal{X}$:*

$$
|E_0(\mathbf{R}) - E_0(\mathbf{R}')| \leq L_E \|\mathbf{R} - \mathbf{R}'\| \quad \forall\, \mathbf{R}, \mathbf{R}' \in \mathcal{X}
$$

**Assumption 4.2** (Universal Approximation). *The MM network class $\{V_\theta : \theta \in \Theta\}$ is dense in the space of $\mathrm{SE}(3) \times S_N$-invariant continuous functions on $\mathcal{X}$ under the sup-norm.* (See [[equivariance]] for the symmetry group.)

**Assumption 4.3** (Critic Calibration). *There exists a finite calibration set size $n_C$ such that, after training on $n_C$ QM evaluations (see [[critic-loss]]), the critic satisfies $\|C_\phi - \Delta(\cdot; \theta)\|_\infty \leq \epsilon_C$ with probability at least $1 - \delta_C$.*

**Assumption 4.4** (Ergodic Exploration). *The entropy-regularized policy $\pi_\psi$ (see [[soft-rlqf]]) induces an ergodic distribution over $\mathcal{X}$: for any open set $U \subset \mathcal{X}$, the stationary distribution $\mu_\psi$ satisfies $\mu_\psi(U) > 0$.*

## Main Convergence Result

**Theorem** (RLQF Convergence). *Under Assumptions 4.1–4.4, the RLQF algorithm with soft policy optimization and prioritized MM training produces a sequence of MM parameters $\{\theta_k\}_{k=0}^{\infty}$ such that:*

$$
\mathbb{E}\left[\Delta^*(\theta_k)\right] \leq \epsilon_C + \mathcal{O}\left(\frac{L_E}{\sqrt{k}} \cdot \sqrt{\frac{\dim(\Theta)}{n_k}}\right)
$$

*where $n_k = \sum_{j=0}^{k} |\mathcal{D}_{\tau_j}|$ is the cumulative number of QM evaluations through iteration $k$.*

### Proof Sketch

The proof proceeds in three stages.

**Stage 1: Critic reliability.** By Assumption 4.3, the [[critic-architecture|critic]] provides an $\epsilon_C$-accurate proxy for the true error. All subsequent bounds carry this additive offset.

**Stage 2: Coverage.** By Assumption 4.4 (ergodicity) and the entropy regularization ([[soft-rlqf|Diversity Guarantee]]), the exploration policy $\pi_\psi$ visits every region of configuration space with positive frequency. Combined with the adversarial reward structure (which directs the policy toward high-error regions via [[curiosity-reward]]), the empirical distribution of sampled configurations converges to a distribution concentrated on the support of $\Delta(\cdot; \theta)$.

**Stage 3: Generalization.** By Assumption 4.2, $V_\theta$ can approximate $E_0$ arbitrarily well. Standard results in neural network generalization theory (Rademacher complexity bounds for equivariant architectures) give:

$$
\mathbb{E}\left[\sup_{\mathbf{R} \in \mathcal{X}} |V_{\theta_k}(\mathbf{R}) - E_0(\mathbf{R})|\right] \leq \inf_{\theta \in \Theta} \sup_{\mathbf{R}} |V_\theta(\mathbf{R}) - E_0(\mathbf{R})| + \mathcal{O}\left(\sqrt{\frac{\dim(\Theta)}{n_k}}\right)
$$

The Lipschitz condition (Assumption 4.1) controls interpolation error between sampled points. Combining stages yields the stated bound. $\square$

## Sample Complexity

**Corollary** (Sample Complexity). *To achieve $\mathbb{E}[\Delta^*(\theta_k)] \leq \epsilon$ (for $\epsilon > \epsilon_C$), it suffices to collect:*

$$
n_k = \mathcal{O}\left(\frac{L_E^2 \cdot \dim(\Theta)}{(\epsilon - \epsilon_C)^2}\right)
$$

*QM evaluations. The critic reduces the constant factor by directing samples to high-error regions, but does not change the asymptotic scaling.*

## Comparison with Passive Learning

In passive (i.i.d.) learning, one samples configurations from a fixed distribution $\rho$ over $\mathcal{X}$. The resulting uniform convergence bound is:

$$
\mathbb{E}\left[\Delta^*(\theta)\right] \leq \mathcal{O}\left(\sqrt{\frac{\dim(\Theta)}{n}} \cdot \frac{1}{\inf_{\mathbf{R} \in \mathcal{X}_{\text{hard}}} \rho(\mathbf{R})}\right)
$$

where $\mathcal{X}_{\text{hard}} = \{\mathbf{R} : \Delta(\mathbf{R}; \theta) > \epsilon\}$. If $\rho$ has poor coverage of $\mathcal{X}_{\text{hard}}$ (typical, since hard configurations are rare in equilibrium sampling), the passive bound degrades by a factor of $1/\rho_{\min}$, which can be exponentially large.

**Proposition** (Active Learning Advantage). *Let $n_{\text{active}}$ and $n_{\text{passive}}$ denote the number of QM evaluations required by RLQF and passive learning, respectively, to achieve $\mathbb{E}[\Delta^*(\theta)] \leq \epsilon$. Then:*

$$
\frac{n_{\text{passive}}}{n_{\text{active}}} \geq \frac{1}{\inf_{\mathbf{R} \in \mathcal{X}_{\text{hard}}} \rho(\mathbf{R})}
$$

*which can be exponentially large in the system dimension $3N$.*

This advantage is the formal justification for the active [[curiosity-reward|exploration]] strategies in RLQF versus passive dataset construction.
