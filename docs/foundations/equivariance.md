# Equivariance Constraints

Symmetry constraints that both the MM actor and QM critic must satisfy to produce physically meaningful predictions.

**Parent:** [[index]]
**See also:** [[mm-qm-gap]], [[critic-architecture]], [[convergence]]

---

## The Symmetry Group of Molecular Systems

Molecular potential energy surfaces are invariant under rigid motions and permutations of identical atoms. Both the actor $V_\theta$ and critic $C_\phi$ must respect these symmetries.

**Definition** (Equivariance Constraint). *For any rotation $g \in \mathrm{SO}(3)$, translation $\mathbf{t} \in \mathbb{R}^3$, and permutation $\sigma \in S_N$ of identical atoms:*

$$
V_\theta(g \cdot \mathbf{R} + \mathbf{t}) = V_\theta(\mathbf{R}), \qquad V_\theta(\sigma \cdot \mathbf{R}) = V_\theta(\mathbf{R})
$$

*and analogously for $C_\phi$. Forces $\mathbf{F}_\theta = -\nabla_{\mathbf{R}} V_\theta$ transform covariantly:*

$$
\mathbf{F}_\theta(g \cdot \mathbf{R}) = g \cdot \mathbf{F}_\theta(\mathbf{R})
$$

## Architectural Enforcement

These constraints are enforced architecturally through equivariant graph neural networks operating on the atomic graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with nodes $\mathcal{V} = \{1, \ldots, N\}$ and edges $\mathcal{E} = \{(i,j) : \|\mathbf{R}_i - \mathbf{R}_j\| < r_c\}$.

Suitable architectures include MACE (higher-order equivariant message passing), NequIP ($E(3)$-equivariant graph neural networks), and PaiNN (polarizable atom interaction neural networks).

## Role in Convergence

The equivariance constraints reduce the effective dimensionality of the function classes $\{V_\theta\}$ and $\{C_\phi\}$, which appears in the [[convergence]] bounds via $\dim(\Theta)$. The universal approximation assumption (Assumption 4.2 in [[convergence]]) is stated over $\mathrm{SE}(3) \times S_N$-invariant continuous functions specifically because of these constraints.

## Role in Force Computation

The covariance of forces under rotation is critical for the [[energy-force-loss]], which trains on both energies and forces. If equivariance is only approximate (e.g., due to discretization of rotation groups), the force loss can exhibit systematic bias in certain orientations.
