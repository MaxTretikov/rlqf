# Energy-Force Loss

The primary supervised loss for training the MM actor: weighted mean squared error on energies and forces, with critic-derived importance weights that focus training on the hardest configurations.

**Parent:** [[index]]
**See also:** [[critic-architecture]], [[kl-divergence-loss]], [[critic-loss]], [[rlqf-objective]]

---

## Definition

At each step of the outer loop in the [[rlqf-objective]], the inner-loop MM update minimizes:

$$
\mathcal{L}_{\text{EF}}(\theta; \mathcal{D}_\tau) = \underbrace{\frac{1}{|\mathcal{D}_\tau|} \sum_{(\mathbf{R}, E_0) \in \mathcal{D}_\tau} w(\mathbf{R}) \left( V_\theta(\mathbf{R}) - E_0(\mathbf{R}) \right)^2}_{\text{energy loss}} + \underbrace{\frac{\mu}{|\mathcal{D}_\tau|} \sum_{(\mathbf{R}, E_0) \in \mathcal{D}_\tau} w(\mathbf{R}) \left\| \mathbf{F}_\theta(\mathbf{R}) - \mathbf{F}_0(\mathbf{R}) \right\|^2}_{\text{force loss}}
$$

where $\mathbf{F}_0 = -\nabla_\mathbf{R} E_0$ are reference QM forces, $\mathbf{F}_\theta = -\nabla_\mathbf{R} V_\theta$ are MM forces (which transform covariantly under the [[equivariance]] constraints), and $\mu$ balances energy and force terms.

## Critic-Derived Importance Weights

The importance weights $w(\mathbf{R})$ are derived from the [[critic-architecture|critic]]:

$$
w(\mathbf{R}) = \frac{C_\phi(\mathbf{R}, V_\theta(\mathbf{R}))^\kappa}{\sum_{\mathbf{R}' \in \mathcal{D}_\tau} C_\phi(\mathbf{R}', V_\theta(\mathbf{R}'))^\kappa}
$$

with temperature exponent $\kappa \geq 0$ controlling the degree of prioritization:

- $\kappa = 0$: Uniform weighting (standard MSE loss)
- $\kappa = 1$: Linear prioritization proportional to critic error
- $\kappa \to \infty$: Trains exclusively on the single hardest sample

This is analogous to prioritized experience replay in standard RL.

## Strengths

- Directly targets the pointwise [[mm-qm-gap]] $\Delta(\mathbf{R}; \theta)$
- Force matching provides gradient information, improving sample efficiency
- Importance weighting focuses compute on the configurations that matter most

## Limitations

- Pointwise energy errors can be dominated by an irrelevant constant offset (the absolute energy reference)
- Does not directly optimize for correct thermodynamic ensembles — a configuration with small energy error can still produce large errors in free energy or population ratios
- These limitations motivate the [[kl-divergence-loss]] as a complementary objective
