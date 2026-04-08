# The MM/QM Gap

The fundamental problem motivating RLQF: the gap between cheap-but-inaccurate molecular mechanics and expensive-but-accurate quantum mechanics.

**Parent:** [[index]]
**See also:** [[verification-generation-asymmetry]], [[critic-architecture]], [[kl-divergence-loss]]

---

## Configuration Space

Let $\mathcal{X} \subset \mathbb{R}^{3N}$ denote the configuration space of an $N$-atom molecular system, with atomic numbers $\mathbf{Z} = (Z_1, \ldots, Z_N)$. The quantum-mechanical potential energy surface (PES) is defined by the ground-state solution to the electronic Schrödinger equation:

$$
\hat{H}\Psi_0(\mathbf{r}; \mathbf{R}) = E_0(\mathbf{R})\Psi_0(\mathbf{r}; \mathbf{R})
$$

where $\mathbf{R} \in \mathcal{X}$ are nuclear coordinates, $\mathbf{r}$ are electronic coordinates, and $\hat{H}$ is the molecular Hamiltonian. The exact PES $E_0: \mathcal{X} \to \mathbb{R}$ is the object we seek to approximate.

## Neural Network Potentials

A molecular mechanics potential $V_{\text{MM}}: \mathcal{X} \to \mathbb{R}$ approximates $E_0$ via parametric functional forms (bonds, angles, torsions, non-bonded interactions). Modern neural network potentials (NNPs) replace these hand-crafted terms with learned representations:

$$
V_{\theta}(\mathbf{R}) = \sum_{i=1}^{N} \varepsilon_{\theta}(\mathbf{R}_i, \{\mathbf{R}_j : j \in \mathcal{N}(i)\})
$$

where $\varepsilon_{\theta}$ is a neural network parameterized by $\theta$ operating on local atomic environments, and $\mathcal{N}(i)$ denotes the neighborhood of atom $i$ within a cutoff radius $r_c$. These must satisfy the [[equivariance]] constraints of molecular systems.

## Formal Definition of the Gap

**Definition** (MM/QM Gap). *The MM/QM gap for a configuration $\mathbf{R}$ is:*

$$
\Delta(\mathbf{R}; \theta) \coloneqq |V_{\theta}(\mathbf{R}) - E_0(\mathbf{R})|
$$

*The global gap is $\Delta^*(\theta) \coloneqq \sup_{\mathbf{R} \in \mathcal{X}} \Delta(\mathbf{R}; \theta)$.*

Note that this pointwise definition captures energy error at individual configurations. For a distributional perspective on the gap — measuring how the MM surface distorts thermodynamic behavior rather than pointwise energy — see [[kl-divergence-loss]].

## Why the Gap Matters

The gap $\Delta$ is what RLQF seeks to minimize. The [[rlqf-objective]] frames this as a bilevel optimization, and the various [[energy-force-loss|loss functions]] provide different ways to drive $\Delta$ toward zero. The [[critic-architecture]] provides a cheap proxy for $\Delta$ that avoids recomputing $E_0$ at every configuration.
