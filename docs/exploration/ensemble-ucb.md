# Ensemble Uncertainty and UCB Exploration

Using ensemble disagreement as a proxy for epistemic uncertainty, combined with the critic signal via the Upper Confidence Bound principle.

**Parent:** [[index]]
**See also:** [[curiosity-reward]], [[adversarial-generation]], [[convergence]]

---

## Ensemble Construction

To account for epistemic uncertainty in both the actor and critic, we maintain ensembles $\{V_{\theta^{(m)}}\}_{m=1}^{M}$ and $\{C_{\phi^{(m)}}\}_{m=1}^{M}$.

## Augmented Exploration Reward

The augmented exploration reward combines the [[curiosity-reward]] with ensemble disagreement:

$$
r_{\text{explore}}(\mathbf{R}) = \underbrace{C_\phi(\mathbf{R}, V_\theta(\mathbf{R}))}_{\text{critic error estimate}} + \underbrace{\beta_1 \cdot \text{Var}_m\left[V_{\theta^{(m)}}(\mathbf{R})\right]^{1/2}}_{\text{actor disagreement}} + \underbrace{\beta_2 \cdot \text{Var}_m\left[C_{\phi^{(m)}}(\mathbf{R}, V_\theta(\mathbf{R}))\right]^{1/2}}_{\text{critic uncertainty}}
$$

where $\beta_1, \beta_2 > 0$ are exploration coefficients. This follows the UCB (Upper Confidence Bound) principle: configurations with high predicted error *or* high model disagreement are prioritized.

## Exploration Completeness

**Proposition** (Exploration Completeness). *Under the augmented reward $r_{\text{explore}}$ and the ergodicity assumption (see [[soft-rlqf]]), the RLQF policy satisfies: for any region $U \subset \mathcal{X}$ where $\Delta(\cdot; \theta) > \epsilon$, the expected number of visits to $U$ in $T$ steps is:*

$$
\mathbb{E}\left[\sum_{t=0}^{T} \mathbb{1}[\mathbf{R}_t \in U]\right] \geq \frac{T \cdot \mu_\psi(U) \cdot r_{\min}(U)}{Z}
$$

*where $r_{\min}(U) = \inf_{\mathbf{R} \in U} r_{\text{explore}}(\mathbf{R})$ and $Z$ is a normalizing constant. Since $r_{\min}(U) \geq \epsilon - \epsilon_C > 0$ for $\epsilon > \epsilon_C$, all high-error regions receive positive visitation.*

## Why Both Actor and Critic Ensembles

- **Actor disagreement** ($\beta_1$ term) catches configurations where the MM networks are uncertain about the energy — they might all be wrong, or they might disagree about which way.
- **Critic disagreement** ($\beta_2$ term) catches configurations where the critic itself is unreliable — regions far from the [[critic-loss|calibration set]] where the error estimate may be inaccurate.

Together, they ensure that the exploration policy is neither overconfident about the actor's predictions nor overconfident about the critic's evaluation of those predictions.
