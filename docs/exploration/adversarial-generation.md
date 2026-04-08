# Adversarial Configuration Generation

Directly optimizing in configuration space to find where the MM network is most wrong, using Langevin dynamics in the critic's error landscape.

**Parent:** [[index]]
**See also:** [[curiosity-reward]], [[ensemble-ucb]], [[verification-generation-asymmetry]], [[distribution-shift]]

---

## Direct Optimization Approach

An alternative to the [[policy-gradient]] approach is to directly solve for the maximally wrong configuration:

$$
\mathbf{R}^* = \arg\max_{\mathbf{R} \in \mathcal{X}} C_\phi(\mathbf{R}, V_\theta(\mathbf{R}))
$$

subject to physical validity constraints (bond lengths, steric constraints, etc.).

## Langevin Dynamics in the Error Landscape

This can be approximated via gradient ascent through the differentiable [[critic-architecture|critic]]:

$$
\mathbf{R}_{k+1} = \mathbf{R}_k + \eta_R \nabla_\mathbf{R} C_\phi(\mathbf{R}_k, V_\theta(\mathbf{R}_k)) + \sigma \xi_k, \quad \xi_k \sim \mathcal{N}(0, I)
$$

which is a Langevin dynamics in the "error landscape" with noise $\sigma$ ensuring exploration. The [[verification-generation-asymmetry]] is critical here: each gradient step through the critic is cheap ($\mathcal{O}(C_{\text{ver}})$ per step), while the configurations discovered may correspond to regions where full QM solution would be expensive — precisely the regime where a cheap critic provides the greatest leverage.

## Physical Validity Constraints

Not all configurations in $\mathcal{X} = \mathbb{R}^{3N}$ are physically meaningful. The Langevin dynamics can be constrained to remain in the physically valid region by:

- Projecting onto bond-length constraints after each step
- Adding penalty terms for steric clashes
- Restricting the step size $\eta_R$ to prevent unphysical jumps

## Adversarial Game Interpretation

The adversarial dynamics between the exploration policy (maximizing error) and the MM actor (minimizing error) can be formalized as a two-player zero-sum game. See [[open-questions]] for discussion of Nash equilibria and connections to GAN theory.

## Distribution Shift Risk

Pure adversarial generation can lead to pathological training distributions. If the MM network only trains on adversarially-selected hard cases, it can lose accuracy on "easy" but physically important configurations. This is addressed by [[distribution-shift|mixed replay buffers]].
