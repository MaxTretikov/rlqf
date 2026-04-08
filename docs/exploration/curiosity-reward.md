# Curiosity Reward

Intrinsic motivation from the critic's error signal: the exploration policy is rewarded for finding configurations where the MM network is most wrong.

**Parent:** [[index]]
**See also:** [[critic-architecture]], [[ensemble-ucb]], [[adversarial-generation]], [[mdp-formulation]]

---

## Definition

The [[critic-architecture|critic]] $C_\phi$ provides a natural *intrinsic motivation* signal. Define the curiosity reward:

$$
r_{\text{curiosity}}(\mathbf{R}) = C_\phi(\mathbf{R}, V_\theta(\mathbf{R}))
$$

This is large precisely where the MM network is uncertain or wrong, directing the exploration policy toward the most informative configurations.

## Role in the MDP

The curiosity reward is the primary component of the [[mdp-formulation|reward function]] $\mathcal{R}(s_t, a_t)$. Combined with the distance regularization term $-\lambda \cdot d(\mathbf{R}_{t+1}, \mathbf{R}_t)$, it defines the signal that the [[policy-gradient]] or [[soft-rlqf]] optimizer follows.

## Relationship to ICM

This is directly analogous to the Intrinsic Curiosity Module (ICM) framework from Pathak et al., except here the "prediction error" has real physical meaning — it is the [[mm-qm-gap]]. As the MM network improves on previously-hard regions, the curiosity signal naturally shifts to unexplored territory.

## Limitations

The curiosity reward alone may not account for epistemic uncertainty in the critic itself. If $C_\phi$ is confidently wrong about its error estimate, the curiosity signal can mislead the exploration policy. This motivates the [[ensemble-ucb]] augmentation, which adds a second signal based on model disagreement.

## Distributional Variant

When using the [[kl-divergence-loss]], the curiosity reward can be reformulated in distributional terms: the critic scores how much each configuration contributes to the KL divergence between QM and MM Boltzmann distributions, rather than how large the pointwise energy error is. This directs exploration toward thermodynamically important failure modes.
