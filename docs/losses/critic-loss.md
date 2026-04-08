# Critic Loss

The supervised regression loss for training the QM critic network on calibration data.

**Parent:** [[index]]
**See also:** [[critic-architecture]], [[energy-force-loss]], [[kl-divergence-loss]], [[computational-complexity]]

---

## Definition

The [[critic-architecture|QM critic]] $C_\phi$ is trained via supervised regression on a calibration set $\mathcal{D}_{\text{cal}}$ of configurations for which both MM predictions and QM ground truth are available:

$$
\mathcal{L}_{\text{critic}}(\phi) = \frac{1}{|\mathcal{D}_{\text{cal}}|} \sum_{(\mathbf{R}, E_0) \in \mathcal{D}_{\text{cal}}} \left( C_\phi(\mathbf{R}, V_\theta(\mathbf{R})) - |V_\theta(\mathbf{R}) - E_0(\mathbf{R})| \right)^2
$$

The critic learns to predict the absolute error $\Delta(\mathbf{R}; \theta) = |V_\theta(\mathbf{R}) - E_0(\mathbf{R})|$ given only the configuration $\mathbf{R}$ and the MM prediction $V_\theta(\mathbf{R})$.

## Calibration Requirements

The [[convergence]] analysis requires the critic to achieve uniform approximation error $\epsilon_C$ after training on $n_C$ QM evaluations (Assumption 4.3). The calibration set $\mathcal{D}_{\text{cal}}$ should cover the relevant regions of configuration space, particularly the high-error regions that the exploration policy will target.

## Recalibration

As the MM actor improves through the RLQF loop, the distribution of errors it produces changes. The critic must be periodically recalibrated to maintain accuracy. The cost of recalibration is analyzed in [[computational-complexity]], where it is shown that the amortization ratio improves with each cycle.

This non-stationarity — "critic drift" — is discussed further in [[open-questions]].

## Relationship to Other Losses

The critic loss trains the *evaluator*, while the [[energy-force-loss]] and [[kl-divergence-loss]] train the *actor*. The critic must be trained first (or concurrently) so that it can provide reliable signals for both the actor's loss weighting and the exploration policy's reward.
