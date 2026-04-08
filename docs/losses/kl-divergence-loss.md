# KL Divergence Loss

A distributional loss that measures how the MM surface distorts the thermodynamically relevant Boltzmann distribution, rather than targeting pointwise energy errors.

**Parent:** [[index]]
**See also:** [[energy-force-loss]], [[critic-loss]], [[mm-qm-gap]], [[soft-rlqf]]

---

## Motivation

The [[energy-force-loss]] minimizes pointwise energy and force errors. But what we often *actually* care about is whether the MM surface produces correct statistical mechanics: correct populations, free energies, and ensemble averages. A configuration with small energy error $\Delta(\mathbf{R}; \theta)$ can still produce large errors in thermodynamic observables if the error is concentrated near the barriers or minima that determine Boltzmann weights.

The KL divergence loss directly targets distributional fidelity.

## Boltzmann Distributions

The QM potential energy surface $E_0$ defines a Boltzmann distribution over configurations at temperature $T$:

$$
p_{\text{QM}}(\mathbf{R}) = \frac{1}{Z_{\text{QM}}} \exp\left(-\frac{E_0(\mathbf{R})}{k_B T}\right), \qquad Z_{\text{QM}} = \int_{\mathcal{X}} \exp\left(-\frac{E_0(\mathbf{R})}{k_B T}\right) d\mathbf{R}
$$

The MM network $V_\theta$ defines an analogous distribution:

$$
p_{\text{MM}}(\mathbf{R}; \theta) = \frac{1}{Z_{\text{MM}}(\theta)} \exp\left(-\frac{V_\theta(\mathbf{R})}{k_B T}\right), \qquad Z_{\text{MM}}(\theta) = \int_{\mathcal{X}} \exp\left(-\frac{V_\theta(\mathbf{R})}{k_B T}\right) d\mathbf{R}
$$

## The KL Divergence Objective

The KL divergence from the QM distribution to the MM distribution measures how much information is lost when using $p_{\text{MM}}$ in place of $p_{\text{QM}}$:

$$
\mathcal{L}_{\text{KL}}(\theta) = D_{\text{KL}}(p_{\text{QM}} \| p_{\text{MM}}) = \int_{\mathcal{X}} p_{\text{QM}}(\mathbf{R}) \log \frac{p_{\text{QM}}(\mathbf{R})}{p_{\text{MM}}(\mathbf{R}; \theta)} \, d\mathbf{R}
$$

## Tractable Decomposition

Expanding the KL divergence yields a form amenable to estimation from QM samples:

$$
D_{\text{KL}}(p_{\text{QM}} \| p_{\text{MM}}) = \frac{1}{k_B T} \mathbb{E}_{p_{\text{QM}}}\left[V_\theta(\mathbf{R}) - E_0(\mathbf{R})\right] + \log Z_{\text{MM}}(\theta) - \log Z_{\text{QM}}
$$

Since $Z_{\text{QM}}$ is constant with respect to $\theta$, the gradient simplifies to:

$$
\nabla_\theta \mathcal{L}_{\text{KL}}(\theta) = \frac{1}{k_B T} \mathbb{E}_{p_{\text{QM}}}\left[\nabla_\theta V_\theta(\mathbf{R})\right] - \mathbb{E}_{p_{\text{MM}}}\left[\frac{\nabla_\theta V_\theta(\mathbf{R})}{k_B T}\right]
$$

$$
= \frac{1}{k_B T} \left( \mathbb{E}_{p_{\text{QM}}}\left[\nabla_\theta V_\theta(\mathbf{R})\right] - \mathbb{E}_{p_{\text{MM}}}\left[\nabla_\theta V_\theta(\mathbf{R})\right] \right)
$$

The first expectation is estimated from QM-sampled configurations (available from the training set). The second is estimated from MM-sampled configurations (cheap to generate via molecular dynamics with $V_\theta$).

## Practical Estimation

In practice, the gradient is estimated as:

$$
\nabla_\theta \mathcal{L}_{\text{KL}}(\theta) \approx \frac{1}{k_B T} \left( \frac{1}{|\mathcal{D}_{\text{QM}}|} \sum_{\mathbf{R} \in \mathcal{D}_{\text{QM}}} \nabla_\theta V_\theta(\mathbf{R}) - \frac{1}{|\mathcal{D}_{\text{MM}}|} \sum_{\mathbf{R} \in \mathcal{D}_{\text{MM}}} \nabla_\theta V_\theta(\mathbf{R}) \right)
$$

where $\mathcal{D}_{\text{QM}}$ are configurations sampled from $p_{\text{QM}}$ (or approximated by short QM MD trajectories) and $\mathcal{D}_{\text{MM}}$ are configurations sampled from $p_{\text{MM}}$ (via cheap MM MD simulations).

The partition function ratio $Z_{\text{MM}} / Z_{\text{QM}}$ drops out of the gradient, making this tractable without computing either partition function directly.

## Relationship to the Critic

The [[critic-architecture|critic]] can be adapted to score distributional error rather than pointwise error. Instead of predicting $|V_\theta(\mathbf{R}) - E_0(\mathbf{R})|$, a distributional critic predicts the local contribution to KL divergence:

$$
C_\phi^{\text{KL}}(\mathbf{R}) \approx p_{\text{QM}}(\mathbf{R}) \log \frac{p_{\text{QM}}(\mathbf{R})}{p_{\text{MM}}(\mathbf{R}; \theta)}
$$

This gives the exploration policy (via [[curiosity-reward]]) a distributional signal: it seeks configurations where the MM surface most distorts the Boltzmann distribution, not just where pointwise energy is worst.

## Complementarity with Energy-Force Loss

The KL divergence loss and [[energy-force-loss]] are complementary:

| Property | Energy-Force Loss | KL Divergence Loss |
|----------|------------------|--------------------|
| Targets | Pointwise energy/force error | Distributional (thermodynamic) error |
| Sensitive to constant offset | Yes | No |
| Requires QM forces | Yes | No (energies suffice) |
| Requires MM sampling | No | Yes |
| Best for | Reaction barriers, geometries | Free energies, populations, ensemble averages |

A combined objective $\mathcal{L}_{\text{MM}} = \mathcal{L}_{\text{EF}} + \nu \mathcal{L}_{\text{KL}}$ with mixing coefficient $\nu > 0$ targets both pointwise accuracy and distributional fidelity, as discussed in [[rlqf-objective]].

## Connection to Soft RLQF

The [[soft-rlqf]] entropy-regularized policy has a natural connection to the KL divergence loss. The optimal soft policy takes the Boltzmann form $\pi^*(a|s) \propto \exp(Q^*/\beta)$, which is structurally identical to the Boltzmann distributions $p_{\text{QM}}$ and $p_{\text{MM}}$ defined here. This suggests a unified variational framework in which both the exploration policy and the MM training objective are derived from the same KL minimization principle, with temperature $\beta$ (policy) and $k_B T$ (physics) playing analogous roles.
