# Notation Summary

**Parent:** [[index]]

---

| Symbol                                               | Definition                              | Introduced in                         |
| ---------------------------------------------------- | --------------------------------------- | ------------------------------------- |
| $\mathcal{X} \subset \mathbb{R}^{3N}$                | Molecular configuration space           | [[mm-qm-gap]]                         |
| $E_0(\mathbf{R})$                                    | Exact QM potential energy surface       | [[mm-qm-gap]]                         |
| $V_\theta(\mathbf{R})$                               | MM neural network potential (actor)     | [[mm-qm-gap]]                         |
| $C_\phi(\mathbf{R}, \tilde{E})$                      | QM neural network critic                | [[critic-architecture]]               |
| $\Delta(\mathbf{R}; \theta)$                         | MM/QM gap at configuration $\mathbf{R}$ | [[mm-qm-gap]]                         |
| $\Delta^*(\theta)$                                   | Global (sup-norm) MM/QM gap             | [[mm-qm-gap]]                         |
| $\mathcal{O}_{\text{gen}}, \mathcal{O}_{\text{ver}}$ | Generation and verification oracles     | [[verification-generation-asymmetry]] |
| $C_{\text{gen}}, C_{\text{ver}}$                     | Oracle costs                            | [[verification-generation-asymmetry]] |
| $\alpha$ (Assumption 1.1)                            | Verification advantage exponent         | [[verification-generation-asymmetry]] |
| $\pi_\psi$                                           | Exploration policy                      | [[mdp-formulation]]                   |
| $\mathcal{R}(s, a)$                                  | Reward function                         | [[mdp-formulation]]                   |
| $\gamma$                                             | Discount factor                         | [[mdp-formulation]]                   |
| $\beta$                                              | Entropy temperature (Soft RLQF)         | [[soft-rlqf]]                         |
| $\beta_1, \beta_2$                                   | Ensemble exploration coefficients       | [[ensemble-ucb]]                      |
| $\mu$                                                | Energy/force loss balance               | [[energy-force-loss]]                 |
| $\kappa$                                             | Prioritization exponent                 | [[energy-force-loss]]                 |
| $w(\mathbf{R})$                                      | Critic-derived importance weights       | [[energy-force-loss]]                 |
| $\alpha$ (Section 5.4)                               | Adversarial mixing ratio                | [[distribution-shift]]                |
| $\nu$                                                | KL/energy-force loss mixing coefficient | [[rlqf-objective]]                    |
| $\epsilon_C$                                         | Critic approximation error              | [[convergence]]                       |
| $\delta_C$                                           | Critic calibration failure probability  | [[convergence]]                       |
| $L_E$                                                | Lipschitz constant of $E_0$             | [[convergence]]                       |
| $p_{\text{QM}}, p_{\text{MM}}$                       | Boltzmann distributions                 | [[kl-divergence-loss]]                |
| $Z_{\text{QM}}, Z_{\text{MM}}$                       | Partition functions                     | [[kl-divergence-loss]]                |
| $C_{\text{QM}}, C_{\text{MM}}, C_{\text{critic}}$    | Per-evaluation computational costs      | [[computational-complexity]]          |
