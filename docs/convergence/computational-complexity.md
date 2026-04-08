# Computational Complexity Analysis

Cost model for RLQF, speedup over passive QM training, and the amortized cost of the critic.

**Parent:** [[index]]
**See also:** [[verification-generation-asymmetry]], [[convergence]], [[critic-architecture]], [[critic-loss]]

---

## Cost Model

Let $C_{\text{QM}}$, $C_{\text{MM}}$, and $C_{\text{critic}}$ denote the per-evaluation costs of QM calculation, MM inference, and critic inference, respectively. By the [[verification-generation-asymmetry|verification advantage]] and neural network amortization:

$$
C_{\text{MM}} \approx C_{\text{critic}} \ll C_{\text{QM}}
$$

## Total Computational Budget

The total cost of $K$ RLQF iterations, each collecting $B$ configurations, is:

$$
C_{\text{total}} = \underbrace{n_C \cdot C_{\text{QM}}}_{\text{critic calibration}} + \sum_{k=1}^{K} \left( \underbrace{B \cdot C_{\text{critic}}}_{\text{exploration}} + \underbrace{B \cdot C_{\text{QM}}}_{\text{QM labeling}} + \underbrace{n_{\text{inner}} \cdot B \cdot C_{\text{MM}}}_{\text{MM training}} \right)
$$

where $n_{\text{inner}}$ is the number of inner-loop gradient steps. When using the [[kl-divergence-loss]], an additional term for MM molecular dynamics sampling is included in $C_{\text{MM}}$.

## Speedup over Direct QM Training

**Theorem** (Computational Advantage). *Let $n_{\text{direct}}$ be the number of QM evaluations required to train $V_\theta$ to accuracy $\epsilon$ by passive sampling, and $n_{\text{RLQF}}$ the number required by RLQF. Then the effective speedup is:*

$$
S = \frac{n_{\text{direct}} \cdot C_{\text{QM}}}{n_C \cdot C_{\text{QM}} + n_{\text{RLQF}} \cdot (C_{\text{QM}} + C_{\text{critic}}) + n_{\text{overhead}}}
$$

*By the [[convergence|Active Learning Advantage]], $n_{\text{RLQF}} \ll n_{\text{direct}}$ when the hard region $\mathcal{X}_{\text{hard}}$ has small measure under the passive sampling distribution. Since $C_{\text{critic}} \ll C_{\text{QM}}$, the speedup is approximately:*

$$
S \approx \frac{n_{\text{direct}}}{n_C + n_{\text{RLQF}}}
$$

*The factor $n_{\text{direct}} / n_{\text{RLQF}}$ can be exponentially large in the system dimension.*

## Amortized Critic Cost

A critical feature of the RLQF framework is that the [[critic-loss|critic]] cost is *amortized*. The initial calibration requires $n_C$ QM evaluations, but subsequent critic queries cost only $C_{\text{critic}} \approx C_{\text{MM}}$. As the actor improves and the error landscape shifts, the critic requires periodic recalibration (see [[open-questions|critic drift]]), but the amortization ratio improves with each cycle:

$$
\text{Amortization ratio at cycle } k = \frac{\sum_{j=1}^{k} B_j \cdot C_{\text{critic}}}{\sum_{j=1}^{k} n_{C,j} \cdot C_{\text{QM}}} \to 0 \text{ as } k \to \infty
$$

provided the recalibration sets $n_{C,j}$ grow sublinearly in $j$.
