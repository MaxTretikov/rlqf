# Open Questions

Unresolved theoretical and practical questions in the RLQF framework.

**Parent:** [[index]]

---

## Critic Drift

As the MM actor improves, the distribution of errors it produces changes, potentially invalidating the [[critic-architecture|critic's]] training distribution. The RLQF framework addresses this through periodic recalibration (see [[computational-complexity]]), but a rigorous analysis of the non-stationary dynamics of the coupled actor-critic system remains open.

Key questions: How fast does the critic's accuracy degrade as $\theta$ evolves? Can we bound the recalibration frequency as a function of the actor's learning rate? Is there a joint training schedule for actor and critic that avoids drift entirely?

## The Curse of Dimensionality in Configuration Space

For large molecular systems ($N \gg 100$), the configuration space $\mathcal{X} \subset \mathbb{R}^{3N}$ becomes extremely high-dimensional. While the [[equivariance]] constraints reduce the effective dimensionality, the exploration problem remains challenging. Hierarchical RLQF — operating at multiple scales from local atomic environments to global molecular conformations — is a promising direction.

## Transferability

A trained RLQF critic $C_\phi$ may transfer across related chemical systems (e.g., from small organic molecules to larger drug-like compounds), further amortizing the QM cost (see [[computational-complexity]]). The conditions under which such transfer is reliable, and the associated generalization bounds, are an important open question.

## Connections to Game Theory

The [[adversarial-generation|adversarial dynamics]] between the exploration policy (maximizing error) and the MM actor (minimizing error) can be formalized as a two-player zero-sum game. The existence and uniqueness of Nash equilibria in this setting, and the convergence rate of gradient-based dynamics to such equilibria, connect RLQF to the broader literature on generative adversarial networks and minimax optimization.

## Unifying the Loss Functions

The [[energy-force-loss]], [[kl-divergence-loss]], and [[critic-loss]] each target different aspects of the [[mm-qm-gap]]. Is there a single variational principle from which all three can be derived? The connection between the [[soft-rlqf]] entropy temperature $\beta$ and the physical temperature $k_B T$ in the KL loss (noted in [[kl-divergence-loss]]) hints at a unified framework, but this remains to be formalized.

## Finite-Temperature Convergence

The [[convergence]] analysis targets the sup-norm gap $\Delta^*(\theta)$. When using the [[kl-divergence-loss]], the relevant convergence criterion is instead the KL divergence $D_{\text{KL}}(p_{\text{QM}} \| p_{\text{MM}})$, which depends on temperature $T$. Convergence guarantees for the distributional objective, and their relationship to the pointwise bounds, have not been established.
