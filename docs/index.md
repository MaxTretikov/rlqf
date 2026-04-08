# Reinforcement Learning with Quantum Feedback

A novel framework for training molecular mechanics (MM) neural network potentials using a quantum mechanics (QM) neural network as a critic. The central insight exploits a fundamental asymmetry in computational cost: verifying the accuracy of an MM prediction against QM ground truth is substantially cheaper than computing the QM solution *ab initio*. We formalize this as an actor-critic reinforcement learning problem, establish convergence guarantees, and derive an active learning protocol that directs computational effort toward maximally informative molecular configurations.

---

## Foundations

- [[mm-qm-gap]] — The core problem: approximating quantum potential energy surfaces with classical neural network potentials
- [[verification-generation-asymmetry]] — Why checking an answer is cheaper than computing it, and how RLQF exploits this
- [[equivariance]] — Symmetry constraints ($\mathrm{SE}(3) \times S_N$ invariance) that both actor and critic must satisfy

## Formulation

- [[mdp-formulation]] — The RLQF Markov Decision Process: states, actions, transitions, and reward
- [[critic-architecture]] — The QM critic network: how it scores MM predictions and why it generalizes
- [[rlqf-objective]] — The bilevel optimization: outer-loop exploration + inner-loop MM training

## Actor-Critic Methods

- [[policy-gradient]] — Policy gradient with critic baseline for the exploration policy
- [[soft-rlqf]] — Entropy-regularized (maximum-entropy) formulation for diverse exploration

## Loss Functions

- [[energy-force-loss]] — Weighted MSE loss on energies and forces with critic-derived importance weights
- [[critic-loss]] — Supervised regression loss for training the QM critic
- [[kl-divergence-loss]] — Distributional loss: minimizing KL divergence between QM and MM Boltzmann distributions

## Convergence and Complexity

- [[convergence]] — Main convergence theorem, proof sketch, sample complexity, and comparison with passive learning
- [[computational-complexity]] — Cost model, speedup analysis, and amortized critic cost

## Exploration Strategies

- [[curiosity-reward]] — Intrinsic motivation from the critic's error signal
- [[ensemble-ucb]] — Ensemble disagreement + UCB-style exploration bonuses
- [[adversarial-generation]] — Langevin dynamics in the error landscape to find maximally wrong configurations
- [[distribution-shift]] — Mixed replay buffers to prevent catastrophic forgetting on easy configurations

## Open Questions

- [[open-questions]] — Critic drift, curse of dimensionality, transferability, and game-theoretic connections

## Reference

- [[notation]] — Symbol table
- [[references]] — Bibliography
