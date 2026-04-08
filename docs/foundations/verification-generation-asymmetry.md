# The Verification–Generation Asymmetry

The core insight behind RLQF: scoring how wrong an MM prediction is costs far less than computing the correct QM answer from scratch.

**Parent:** [[index]]
**See also:** [[mm-qm-gap]], [[critic-architecture]], [[computational-complexity]]

---

## The Cost Landscape of Quantum Chemistry

The computational cost of *generating* $E_0(\mathbf{R})$ from first principles scales steeply with system size:

| Method | Scaling | Accuracy |
|--------|---------|----------|
| DFT | $\mathcal{O}(N^3)$ | Moderate |
| MP2 | $\mathcal{O}(N^5)$ | Good |
| CCSD(T) | $\mathcal{O}(N^7)$ | High |
| FCI | $\mathcal{O}(e^N)$ | Exact |

In contrast, *verifying* whether a candidate energy $\tilde{E}$ is close to $E_0(\mathbf{R})$ can be performed at reduced cost — via variational bounds, density comparisons, or force residuals.

## Oracle Formalization

**Definition** (QM Generation Oracle). *A QM generation oracle $\mathcal{O}_{\text{gen}}$ takes a configuration $\mathbf{R} \in \mathcal{X}$ and returns $E_0(\mathbf{R})$ to precision $\epsilon$. Its cost is $C_{\text{gen}}(N, \epsilon)$.*

**Definition** (QM Verification Oracle). *A QM verification oracle $\mathcal{O}_{\text{ver}}$ takes a configuration $\mathbf{R} \in \mathcal{X}$ and a candidate energy $\tilde{E}$, and returns a score $s(\mathbf{R}, \tilde{E}) \in \mathbb{R}$ that is monotonically related to $|\tilde{E} - E_0(\mathbf{R})|$. Its cost is $C_{\text{ver}}(N, \epsilon)$.*

## The Verification Advantage

**Assumption** (Verification Advantage). *There exists a constant $\alpha > 1$ such that for all $N$ and $\epsilon$:*

$$
C_{\text{ver}}(N, \epsilon) \leq C_{\text{gen}}(N, \epsilon)^{1/\alpha}
$$

This assumption is grounded in the structure of quantum chemistry: checking whether a proposed wavefunction yields the correct energy (via variational bounds, density comparisons, or force residuals) is cheaper than solving for that wavefunction from scratch. The verification oracle need not produce $E_0$ itself — it only needs to produce a reliable *error signal*. The exponent $\alpha$ captures the degree of this asymmetry; for instance, if generation scales as $\mathcal{O}(N^7)$ (CCSD(T)) while verification via density residuals scales as $\mathcal{O}(N^3)$, then $\alpha \approx 7/3$.

## Neural Network Amortization

In the neural network setting, $\mathcal{O}_{\text{ver}}$ is instantiated by a QM-trained [[critic-architecture|critic network]] $C_\phi$ that predicts the error $\Delta(\mathbf{R}; \theta)$ directly, having been trained on a dataset of $(V_\theta(\mathbf{R}), E_0(\mathbf{R}))$ pairs. The key economy arises because $C_\phi$ generalizes: after training on a modest set of QM calculations, it can score MM predictions on novel configurations without additional QM computation.

This amortization is analyzed quantitatively in [[computational-complexity]].

## Role in RLQF

RLQF exploits this asymmetry by casting the [[mm-qm-gap]] approximation problem as a reinforcement learning problem (see [[rlqf-objective]]) in which:

1. The **actor** (policy) is the MM neural network $V_\theta$, proposing energy predictions.
2. The **critic** is the QM neural network $C_\phi$, scoring how wrong those predictions are.
3. The **environment** is molecular configuration space $\mathcal{X}$.
4. The **reward** is derived from the negative of the critic's error estimate.

The RL loop drives $V_\theta$ toward regions of configuration space where it performs poorly, trains on those regions, and iterates — achieving a form of adversarial active learning mediated by the cheap critic. The [[adversarial-generation]] strategy leverages this asymmetry most directly.
