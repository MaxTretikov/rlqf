# Reinforcement Learning with Quantum Feedback: A Mathematical Thesis

---

## Abstract

We present **Reinforcement Learning with Quantum Feedback (RLQF)**, a novel framework for training molecular mechanics (MM) neural network potentials using a quantum mechanics (QM) neural network as a critic. The central insight exploits a fundamental asymmetry in computational complexity: verifying the accuracy of an MM prediction against QM ground truth is substantially cheaper than computing the QM solution *ab initio*. We formalize this as an actor-critic reinforcement learning problem, establish convergence guarantees under stated assumptions, and derive an active learning protocol that directs computational effort toward maximally informative molecular configurations. The framework achieves near-QM accuracy at MM computational cost by leveraging the intrinsic cost asymmetry between QM error verification and QM solution generation.

---

## 1. Introduction and the Verification–Generation Asymmetry

### 1.1 The MM/QM Gap

Let $\mathcal{X} \subset \mathbb{R}^{3N}$ denote the configuration space of an $N$-atom molecular system, with atomic numbers $\mathbf{Z} = (Z_1, \ldots, Z_N)$. The quantum-mechanical potential energy surface (PES) is defined by the ground-state solution to the electronic Schrödinger equation:

$$
\hat{H}\Psi_0(\mathbf{r}; \mathbf{R}) = E_0(\mathbf{R})\Psi_0(\mathbf{r}; \mathbf{R})
$$

where $\mathbf{R} \in \mathcal{X}$ are nuclear coordinates, $\mathbf{r}$ are electronic coordinates, and $\hat{H}$ is the molecular Hamiltonian. The exact PES $E_0: \mathcal{X} \to \mathbb{R}$ is the object we seek to approximate.

A molecular mechanics potential $V_{\text{MM}}: \mathcal{X} \to \mathbb{R}$ approximates $E_0$ via parametric functional forms (bonds, angles, torsions, non-bonded interactions). Modern neural network potentials (NNPs) replace these hand-crafted terms with learned representations:

$$
V_{\theta}(\mathbf{R}) = \sum_{i=1}^{N} \varepsilon_{\theta}(\mathbf{R}_i, \{\mathbf{R}_j : j \in \mathcal{N}(i)\})
$$

where $\varepsilon_{\theta}$ is a neural network parameterized by $\theta$ operating on local atomic environments, and $\mathcal{N}(i)$ denotes the neighborhood of atom $i$ within a cutoff radius $r_c$.

**Definition 1.1** (MM/QM Gap). *The MM/QM gap for a configuration $\mathbf{R}$ is:*

$$
\Delta(\mathbf{R}; \theta) \coloneqq |V_{\theta}(\mathbf{R}) - E_0(\mathbf{R})|
$$

*The global gap is $\Delta^*(\theta) \coloneqq \sup_{\mathbf{R} \in \mathcal{X}} \Delta(\mathbf{R}; \theta)$.*

### 1.2 The Verification–Generation Asymmetry

The computational cost of *generating* $E_0(\mathbf{R})$ from first principles scales steeply. Density functional theory (DFT) scales as $\mathcal{O}(N^3)$, coupled cluster with singles, doubles, and perturbative triples (CCSD(T)) scales as $\mathcal{O}(N^7)$, and full configuration interaction (FCI) scales exponentially. In contrast, *verifying* whether a candidate energy $\tilde{E}$ is close to $E_0(\mathbf{R})$ can be performed at reduced cost.

We formalize this via oracle complexity.

**Definition 1.2** (QM Generation Oracle). *A QM generation oracle $\mathcal{O}_{\text{gen}}$ takes a configuration $\mathbf{R} \in \mathcal{X}$ and returns $E_0(\mathbf{R})$ to precision $\epsilon$. Its cost is $C_{\text{gen}}(N, \epsilon)$.*

**Definition 1.3** (QM Verification Oracle). *A QM verification oracle $\mathcal{O}_{\text{ver}}$ takes a configuration $\mathbf{R} \in \mathcal{X}$ and a candidate energy $\tilde{E}$, and returns a score $s(\mathbf{R}, \tilde{E}) \in \mathbb{R}$ that is monotonically related to $|\tilde{E} - E_0(\mathbf{R})|$. Its cost is $C_{\text{ver}}(N, \epsilon)$.*

**Assumption 1.1** (Verification Advantage). *There exists a constant $\alpha > 1$ such that for all $N$ and $\epsilon$:*

$$
C_{\text{ver}}(N, \epsilon) \leq C_{\text{gen}}(N, \epsilon)^{1/\alpha}
$$

This assumption is grounded in the structure of quantum chemistry: checking whether a proposed wavefunction yields the correct energy (via variational bounds, density comparisons, or force residuals) is cheaper than solving for that wavefunction from scratch. The verification oracle need not produce $E_0$ itself — it only needs to produce a reliable *error signal*. The exponent $\alpha$ captures the degree of this asymmetry; for instance, if generation scales as $\mathcal{O}(N^7)$ (CCSD(T)) while verification via density residuals scales as $\mathcal{O}(N^3)$, then $\alpha \approx 7/3$.

**Remark 1.1.** In the neural network setting, $\mathcal{O}_{\text{ver}}$ is instantiated by a QM-trained critic network $C_\phi$ that predicts the error $\Delta(\mathbf{R}; \theta)$ directly, having been trained on a dataset of $(V_\theta(\mathbf{R}), E_0(\mathbf{R}))$ pairs. The key economy arises because $C_\phi$ generalizes: after training on a modest set of QM calculations, it can score MM predictions on novel configurations without additional QM computation.

### 1.3 The RLQF Principle

RLQF exploits this asymmetry by casting the MM/QM approximation problem as a reinforcement learning problem in which:

1. The **actor** (policy) is the MM neural network $V_\theta$, proposing energy predictions.
2. The **critic** is the QM neural network $C_\phi$, scoring how wrong those predictions are.
3. The **environment** is molecular configuration space $\mathcal{X}$.
4. The **reward** is derived from the negative of the critic's error estimate.

The RL loop drives $V_\theta$ toward regions of configuration space where it performs poorly, trains on those regions, and iterates — achieving a form of *adversarial active learning* mediated by the cheap critic.

---

## 2. Formal Problem Setup

### 2.1 The RLQF Markov Decision Process

We define a Markov Decision Process $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma)$ as follows.

**State space.** $\mathcal{S} = \mathcal{X} \times \Theta$, where $\mathcal{X} \subset \mathbb{R}^{3N}$ is the molecular configuration space and $\Theta$ is the parameter space of the MM network. A state $s_t = (\mathbf{R}_t, \theta_t)$ encodes the current molecular configuration and the current MM parameters.

**Action space.** $\mathcal{A} = \mathcal{X}$. An action $a_t = \mathbf{R}_{t+1}$ is the selection of the next molecular configuration to evaluate. The policy $\pi_\psi: \mathcal{S} \to \mathcal{P}(\mathcal{A})$ is a stochastic mapping parameterized by $\psi$ that proposes configurations.

**Transition dynamics.** The transition function $\mathcal{T}$ is factored:

$$
\mathcal{T}(s_{t+1} | s_t, a_t) = \delta(\mathbf{R}_{t+1} = a_t) \cdot p(\theta_{t+1} | \theta_t, \mathbf{R}_{t+1})
$$

where $\delta$ is the Dirac delta (the configuration transitions deterministically to the chosen action) and $p(\theta_{t+1} | \theta_t, \mathbf{R}_{t+1})$ encodes the MM parameter update rule (e.g., a gradient descent step on the loss at $\mathbf{R}_{t+1}$):

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t, \mathbf{R}_{t+1})
$$

**Reward function.** The reward at step $t$ is:

$$
\mathcal{R}(s_t, a_t) = C_\phi(\mathbf{R}_{t+1}, V_{\theta_t}(\mathbf{R}_{t+1})) - \lambda \cdot d(\mathbf{R}_{t+1}, \mathbf{R}_t)
$$

where $C_\phi$ is the critic's error estimate and $\lambda \cdot d(\cdot, \cdot)$ is a regularization term penalizing large jumps in configuration space (encouraging smooth exploration). The first term rewards the agent for finding configurations where the MM network is maximally wrong; the second term prevents degenerate solutions.

**Discount factor.** $\gamma \in [0, 1)$ controls the planning horizon.

### 2.2 Energy Functionals and Equivariance

Both the actor $V_\theta$ and critic $C_\phi$ must respect the symmetries of molecular systems.

**Definition 2.1** (Equivariance Constraint). *For any rotation $g \in \mathrm{SO}(3)$, translation $\mathbf{t} \in \mathbb{R}^3$, and permutation $\sigma \in S_N$ of identical atoms:*

$$
V_\theta(g \cdot \mathbf{R} + \mathbf{t}) = V_\theta(\mathbf{R}), \qquad V_\theta(\sigma \cdot \mathbf{R}) = V_\theta(\mathbf{R})
$$

*and analogously for $C_\phi$. Forces $\mathbf{F}_\theta = -\nabla_{\mathbf{R}} V_\theta$ transform covariantly:*

$$
\mathbf{F}_\theta(g \cdot \mathbf{R}) = g \cdot \mathbf{F}_\theta(\mathbf{R})
$$

These constraints are enforced architecturally through equivariant graph neural networks (e.g., MACE, NequIP) operating on the atomic graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ with nodes $\mathcal{V} = \{1, \ldots, N\}$ and edges $\mathcal{E} = \{(i,j) : \|\mathbf{R}_i - \mathbf{R}_j\| < r_c\}$.

### 2.3 The Critic Architecture

The QM critic $C_\phi: \mathcal{X} \times \mathbb{R} \to \mathbb{R}_{\geq 0}$ maps a molecular configuration and a candidate energy to a non-negative error score. It is trained via supervised regression on a calibration set $\mathcal{D}_{\text{cal}}$:

$$
\mathcal{L}_{\text{critic}}(\phi) = \frac{1}{|\mathcal{D}_{\text{cal}}|} \sum_{(\mathbf{R}, E_0) \in \mathcal{D}_{\text{cal}}} \left( C_\phi(\mathbf{R}, V_\theta(\mathbf{R})) - |V_\theta(\mathbf{R}) - E_0(\mathbf{R})| \right)^2
$$

**Proposition 2.1** (Critic Sufficiency). *If $C_\phi$ satisfies $|C_\phi(\mathbf{R}, \tilde{E}) - |\tilde{E} - E_0(\mathbf{R})|| \leq \epsilon_C$ uniformly over $\mathcal{X}$, then optimizing $V_\theta$ to minimize $C_\phi$ is equivalent to minimizing $\Delta(\mathbf{R}; \theta)$ up to additive error $\epsilon_C$.*

*Proof.* For any $\mathbf{R} \in \mathcal{X}$:

$$
\Delta(\mathbf{R}; \theta) - \epsilon_C \leq C_\phi(\mathbf{R}, V_\theta(\mathbf{R})) \leq \Delta(\mathbf{R}; \theta) + \epsilon_C
$$

Thus $\arg\min_\theta \mathbb{E}_\mathbf{R}[C_\phi(\mathbf{R}, V_\theta(\mathbf{R}))]$ and $\arg\min_\theta \mathbb{E}_\mathbf{R}[\Delta(\mathbf{R}; \theta)]$ coincide when $\epsilon_C \to 0$, and the optima differ by at most $\epsilon_C$ in objective value for finite $\epsilon_C$. $\square$

---

## 3. Actor-Critic Formulation

### 3.1 The RLQF Objective

The RLQF objective is to find a policy $\pi_\psi^*$ that maximizes the expected cumulative reward over a trajectory $\tau = (s_0, a_0, s_1, a_1, \ldots)$:

$$
J(\psi) = \mathbb{E}_{\tau \sim \pi_\psi} \left[ \sum_{t=0}^{T} \gamma^t \mathcal{R}(s_t, a_t) \right]
$$

subject to the MM parameter updates induced by the trajectory. This is a *bilevel optimization*:

- **Outer loop** (exploration policy): $\max_\psi J(\psi)$ — find configurations where the MM net is most wrong.
- **Inner loop** (MM training): $\min_\theta \mathcal{L}_{\text{MM}}(\theta; \mathcal{D}_\tau)$ — train the MM net on the collected configurations.

where $\mathcal{D}_\tau = \{(\mathbf{R}_t, E_0(\mathbf{R}_t))\}_{t=0}^{T}$ is the dataset induced by the trajectory.

### 3.2 Policy Gradient with Critic Baseline

The gradient of the RLQF objective with respect to the exploration policy parameters $\psi$ follows the policy gradient theorem:

$$
\nabla_\psi J(\psi) = \mathbb{E}_{\tau \sim \pi_\psi} \left[ \sum_{t=0}^{T} \nabla_\psi \log \pi_\psi(a_t | s_t) \cdot A^{\pi_\psi}(s_t, a_t) \right]
$$

where the advantage function is:

$$
A^{\pi_\psi}(s_t, a_t) = Q^{\pi_\psi}(s_t, a_t) - V^{\pi_\psi}(s_t)
$$

In RLQF, the QM critic serves double duty: it provides both the reward signal $\mathcal{R}$ (through error estimation) and the value baseline $V^{\pi_\psi}$ (through generalization over states).

### 3.3 Entropy-Regularized Formulation (Soft RLQF)

To ensure diverse exploration of configuration space, we adopt the maximum-entropy framework:

$$
J_{\text{soft}}(\psi) = \mathbb{E}_{\tau \sim \pi_\psi} \left[ \sum_{t=0}^{T} \gamma^t \Big( \mathcal{R}(s_t, a_t) + \beta \mathcal{H}(\pi_\psi(\cdot | s_t)) \Big) \right]
$$

where $\mathcal{H}(\pi_\psi(\cdot | s_t)) = -\mathbb{E}_{a \sim \pi_\psi}[\log \pi_\psi(a | s_t)]$ is the entropy of the policy and $\beta > 0$ is the temperature parameter.

**Proposition 3.1** (Diversity Guarantee). *Under the soft RLQF objective, the optimal policy at state $s$ satisfies:*

$$
\pi_\psi^*(a | s) \propto \exp\left(\frac{1}{\beta} Q_{\text{soft}}^*(s, a)\right)
$$

*In particular, $\pi_\psi^*$ assigns non-zero probability to every action in $\mathcal{A}$, preventing collapse to a single failure mode of $V_\theta$.*

### 3.4 The MM Parameter Update

At each step of the outer loop, the inner-loop MM update proceeds by minimizing a loss that blends supervised and critic-informed components:

$$
\mathcal{L}_{\text{MM}}(\theta; \mathcal{D}_\tau) = \underbrace{\frac{1}{|\mathcal{D}_\tau|} \sum_{(\mathbf{R}, E_0) \in \mathcal{D}_\tau} w(\mathbf{R}) \left( V_\theta(\mathbf{R}) - E_0(\mathbf{R}) \right)^2}_{\text{energy loss}} + \underbrace{\frac{\mu}{|\mathcal{D}_\tau|} \sum_{(\mathbf{R}, E_0) \in \mathcal{D}_\tau} w(\mathbf{R}) \left\| \mathbf{F}_\theta(\mathbf{R}) - \mathbf{F}_0(\mathbf{R}) \right\|^2}_{\text{force loss}}
$$

where $\mathbf{F}_0 = -\nabla_\mathbf{R} E_0$ are reference QM forces, $\mu$ balances energy and force terms, and the importance weights $w(\mathbf{R})$ are derived from the critic:

$$
w(\mathbf{R}) = \frac{C_\phi(\mathbf{R}, V_\theta(\mathbf{R}))^\kappa}{\sum_{\mathbf{R}' \in \mathcal{D}_\tau} C_\phi(\mathbf{R}', V_\theta(\mathbf{R}'))^\kappa}
$$

with temperature exponent $\kappa \geq 0$ controlling the degree of prioritization. Setting $\kappa = 0$ recovers uniform weighting; $\kappa \to \infty$ trains exclusively on the hardest sample.

---

## 4. Convergence Properties and Error Bounds

### 4.1 Assumptions

We require the following regularity conditions.

**Assumption 4.1** (Lipschitz Continuity). *The true PES $E_0$ is $L_E$-Lipschitz on $\mathcal{X}$:*

$$
|E_0(\mathbf{R}) - E_0(\mathbf{R}')| \leq L_E \|\mathbf{R} - \mathbf{R}'\| \quad \forall\, \mathbf{R}, \mathbf{R}' \in \mathcal{X}
$$

**Assumption 4.2** (Universal Approximation). *The MM network class $\{V_\theta : \theta \in \Theta\}$ is dense in the space of $\mathrm{SE}(3) \times S_N$-invariant continuous functions on $\mathcal{X}$ under the sup-norm.*

**Assumption 4.3** (Critic Calibration). *There exists a finite calibration set size $n_C$ such that, after training on $n_C$ QM evaluations, the critic satisfies $\|C_\phi - \Delta(\cdot; \theta)\|_\infty \leq \epsilon_C$ with probability at least $1 - \delta_C$.*

**Assumption 4.4** (Ergodic Exploration). *The entropy-regularized policy $\pi_\psi$ induces an ergodic distribution over $\mathcal{X}$: for any open set $U \subset \mathcal{X}$, the stationary distribution $\mu_\psi$ satisfies $\mu_\psi(U) > 0$.*

### 4.2 Main Convergence Result

**Theorem 4.1** (RLQF Convergence). *Under Assumptions 4.1–4.4, the RLQF algorithm with soft policy optimization and prioritized MM training produces a sequence of MM parameters $\{\theta_k\}_{k=0}^{\infty}$ such that:*

$$
\mathbb{E}\left[\Delta^*(\theta_k)\right] \leq \epsilon_C + \mathcal{O}\left(\frac{L_E}{\sqrt{k}} \cdot \sqrt{\frac{\dim(\Theta)}{n_k}}\right)
$$

*where $n_k = \sum_{j=0}^{k} |\mathcal{D}_{\tau_j}|$ is the cumulative number of QM evaluations through iteration $k$.*

*Proof sketch.* The proof proceeds in three stages.

**Stage 1: Critic reliability.** By Assumption 4.3, the critic provides an $\epsilon_C$-accurate proxy for the true error. All subsequent bounds carry this additive offset.

**Stage 2: Coverage.** By Assumption 4.4 (ergodicity) and the entropy regularization (Proposition 3.1), the exploration policy $\pi_\psi$ visits every region of configuration space with positive frequency. Combined with the adversarial reward structure (which directs the policy toward high-error regions), the empirical distribution of sampled configurations converges to a distribution that is concentrated on the support of $\Delta(\cdot; \theta)$.

**Stage 3: Generalization.** By Assumption 4.2, $V_\theta$ can approximate $E_0$ arbitrarily well. Standard results in neural network generalization theory (Rademacher complexity bounds for equivariant architectures) give:

$$
\mathbb{E}\left[\sup_{\mathbf{R} \in \mathcal{X}} |V_{\theta_k}(\mathbf{R}) - E_0(\mathbf{R})|\right] \leq \inf_{\theta \in \Theta} \sup_{\mathbf{R}} |V_\theta(\mathbf{R}) - E_0(\mathbf{R})| + \mathcal{O}\left(\sqrt{\frac{\dim(\Theta)}{n_k}}\right)
$$

The Lipschitz condition (Assumption 4.1) controls interpolation error between sampled points. Combining stages yields the stated bound. $\square$

**Corollary 4.1** (Sample Complexity). *To achieve $\mathbb{E}[\Delta^*(\theta_k)] \leq \epsilon$ (for $\epsilon > \epsilon_C$), it suffices to collect:*

$$
n_k = \mathcal{O}\left(\frac{L_E^2 \cdot \dim(\Theta)}{(\epsilon - \epsilon_C)^2}\right)
$$

*QM evaluations. The critic reduces the constant factor by directing samples to high-error regions, but does not change the asymptotic scaling.*

### 4.3 Comparison with Passive Learning

In passive (i.i.d.) learning, one samples configurations from a fixed distribution $\rho$ over $\mathcal{X}$. The resulting uniform convergence bound is:

$$
\mathbb{E}\left[\Delta^*(\theta)\right] \leq \mathcal{O}\left(\sqrt{\frac{\dim(\Theta)}{n}} \cdot \frac{1}{\inf_{\mathbf{R} \in \mathcal{X}_{\text{hard}}} \rho(\mathbf{R})}\right)
$$

where $\mathcal{X}_{\text{hard}} = \{\mathbf{R} : \Delta(\mathbf{R}; \theta) > \epsilon\}$. If $\rho$ has poor coverage of $\mathcal{X}_{\text{hard}}$ (which is typical, since hard configurations are often rare in equilibrium sampling), the passive bound degrades by a factor of $1/\rho_{\min}$, which can be exponentially large.

**Proposition 4.1** (Active Learning Advantage). *Let $n_{\text{active}}$ and $n_{\text{passive}}$ denote the number of QM evaluations required by RLQF and passive learning, respectively, to achieve $\mathbb{E}[\Delta^*(\theta)] \leq \epsilon$. Then:*

$$
\frac{n_{\text{passive}}}{n_{\text{active}}} \geq \frac{1}{\inf_{\mathbf{R} \in \mathcal{X}_{\text{hard}}} \rho(\mathbf{R})}
$$

*which can be exponentially large in the system dimension $3N$.*

---

## 5. Active Learning and Curiosity-Driven Exploration

### 5.1 Intrinsic Motivation from the Critic

The critic $C_\phi$ provides a natural *intrinsic motivation* signal. Define the curiosity reward:

$$
r_{\text{curiosity}}(\mathbf{R}) = C_\phi(\mathbf{R}, V_\theta(\mathbf{R}))
$$

This is large precisely where the MM network is uncertain or wrong, directing the exploration policy toward the most informative configurations.

### 5.2 Ensemble Uncertainty Augmentation

To account for epistemic uncertainty in both the actor and critic, we maintain ensembles $\{V_{\theta^{(m)}}\}_{m=1}^{M}$ and $\{C_{\phi^{(m)}}\}_{m=1}^{M}$. The augmented exploration reward is:

$$
r_{\text{explore}}(\mathbf{R}) = \underbrace{C_\phi(\mathbf{R}, V_\theta(\mathbf{R}))}_{\text{critic error estimate}} + \underbrace{\beta_1 \cdot \text{Var}_m\left[V_{\theta^{(m)}}(\mathbf{R})\right]^{1/2}}_{\text{actor disagreement}} + \underbrace{\beta_2 \cdot \text{Var}_m\left[C_{\phi^{(m)}}(\mathbf{R}, V_\theta(\mathbf{R}))\right]^{1/2}}_{\text{critic uncertainty}}
$$

where $\beta_1, \beta_2 > 0$ are exploration coefficients. This follows the UCB (Upper Confidence Bound) principle: configurations with high predicted error *or* high model disagreement are prioritized.

**Proposition 5.1** (Exploration Completeness). *Under the augmented reward $r_{\text{explore}}$ and the ergodicity assumption, the RLQF policy satisfies: for any region $U \subset \mathcal{X}$ where $\Delta(\cdot; \theta) > \epsilon$, the expected number of visits to $U$ in $T$ steps is:*

$$
\mathbb{E}\left[\sum_{t=0}^{T} \mathbb{1}[\mathbf{R}_t \in U]\right] \geq \frac{T \cdot \mu_\psi(U) \cdot r_{\min}(U)}{Z}
$$

*where $r_{\min}(U) = \inf_{\mathbf{R} \in U} r_{\text{explore}}(\mathbf{R})$ and $Z$ is a normalizing constant. Since $r_{\min}(U) \geq \epsilon - \epsilon_C > 0$ for $\epsilon > \epsilon_C$, all high-error regions receive positive visitation.*

### 5.3 Adversarial Configuration Generation

An alternative to the policy-gradient approach is to directly solve:

$$
\mathbf{R}^* = \arg\max_{\mathbf{R} \in \mathcal{X}} C_\phi(\mathbf{R}, V_\theta(\mathbf{R}))
$$

subject to physical validity constraints (bond lengths, steric constraints, etc.). This can be approximated via gradient ascent through the differentiable critic:

$$
\mathbf{R}_{k+1} = \mathbf{R}_k + \eta_R \nabla_\mathbf{R} C_\phi(\mathbf{R}_k, V_\theta(\mathbf{R}_k)) + \sigma \xi_k, \quad \xi_k \sim \mathcal{N}(0, I)
$$

which is a Langevin dynamics in the "error landscape" with noise $\sigma$ ensuring exploration. The verification–generation asymmetry (Assumption 1.1) is critical here: each gradient step through the critic is cheap ($\mathcal{O}(C_{\text{ver}})$ per step), while the configurations discovered may correspond to regions where full QM solution would be expensive — precisely the regime where a cheap critic provides the greatest leverage.

### 5.4 Distribution Shift Mitigation

Pure adversarial sampling leads to distribution shift: the MM network may degrade on "easy" configurations while improving on hard ones. We address this via a mixed replay buffer:

$$
\mathcal{D}_{\text{train}} = (1 - \alpha) \cdot \mathcal{D}_{\text{Boltzmann}} + \alpha \cdot \mathcal{D}_{\text{adversarial}}
$$

where $\mathcal{D}_{\text{Boltzmann}}$ contains configurations sampled from a physical Boltzmann distribution at temperature $T_{\text{phys}}$ and $\mathcal{D}_{\text{adversarial}}$ contains configurations selected by the exploration policy. The mixing ratio $\alpha \in [0, 1]$ trades off accuracy on physically relevant configurations against worst-case robustness.

---

## 6. Computational Complexity Analysis

### 6.1 Cost Model

Let $C_{\text{QM}}$, $C_{\text{MM}}$, and $C_{\text{critic}}$ denote the per-evaluation costs of QM calculation, MM inference, and critic inference, respectively. By the verification advantage (Assumption 1.1) and the neural network amortization:

$$
C_{\text{MM}} \approx C_{\text{critic}} \ll C_{\text{QM}}
$$

### 6.2 Total Computational Budget

The total cost of $K$ RLQF iterations, each collecting $B$ configurations, is:

$$
C_{\text{total}} = \underbrace{n_C \cdot C_{\text{QM}}}_{\text{critic calibration}} + \sum_{k=1}^{K} \left( \underbrace{B \cdot C_{\text{critic}}}_{\text{exploration}} + \underbrace{B \cdot C_{\text{QM}}}_{\text{QM labeling}} + \underbrace{n_{\text{inner}} \cdot B \cdot C_{\text{MM}}}_{\text{MM training}} \right)
$$

where $n_{\text{inner}}$ is the number of inner-loop gradient steps.

### 6.3 Speedup over Direct QM Training

**Theorem 6.1** (Computational Advantage). *Let $n_{\text{direct}}$ be the number of QM evaluations required to train $V_\theta$ to accuracy $\epsilon$ by passive sampling, and $n_{\text{RLQF}}$ the number required by RLQF. Then the effective speedup is:*

$$
S = \frac{n_{\text{direct}} \cdot C_{\text{QM}}}{n_C \cdot C_{\text{QM}} + n_{\text{RLQF}} \cdot (C_{\text{QM}} + C_{\text{critic}}) + n_{\text{overhead}}}
$$

*By Proposition 4.1, $n_{\text{RLQF}} \ll n_{\text{direct}}$ when the hard region $\mathcal{X}_{\text{hard}}$ has small measure under the passive sampling distribution. Since $C_{\text{critic}} \ll C_{\text{QM}}$, the speedup is approximately:*

$$
S \approx \frac{n_{\text{direct}}}{n_C + n_{\text{RLQF}}}
$$

*The factor $n_{\text{direct}} / n_{\text{RLQF}}$ can be exponentially large in the system dimension.*

### 6.4 Amortized Critic Cost

A critical feature of the RLQF framework is that the critic cost is *amortized*. The initial calibration requires $n_C$ QM evaluations, but subsequent critic queries cost only $C_{\text{critic}} \approx C_{\text{MM}}$. As the actor improves and the error landscape shifts, the critic requires periodic recalibration, but the amortization ratio improves with each cycle:

$$
\text{Amortization ratio at cycle } k = \frac{\sum_{j=1}^{k} B_j \cdot C_{\text{critic}}}{\sum_{j=1}^{k} n_{C,j} \cdot C_{\text{QM}}} \to 0 \text{ as } k \to \infty
$$

provided the recalibration sets $n_{C,j}$ grow sublinearly in $j$.

---

## 7. Discussion and Open Questions

### 7.1 Critic Drift

As the MM actor improves, the distribution of errors it produces changes, potentially invalidating the critic's training distribution. The RLQF framework addresses this through periodic recalibration (Section 6.4), but a rigorous analysis of the non-stationary dynamics of the coupled actor-critic system remains an open problem.

### 7.2 The Curse of Dimensionality in Configuration Space

For large molecular systems ($N \gg 100$), the configuration space $\mathcal{X} \subset \mathbb{R}^{3N}$ becomes extremely high-dimensional. While the equivariance constraints (Definition 2.1) reduce the effective dimensionality, the exploration problem remains challenging. Hierarchical RLQF — operating at multiple scales from local atomic environments to global molecular conformations — is a promising direction.

### 7.3 Transferability

A trained RLQF critic $C_\phi$ may transfer across related chemical systems (e.g., from small organic molecules to larger drug-like compounds), further amortizing the QM cost. The conditions under which such transfer is reliable, and the associated generalization bounds, are an important open question.

### 7.4 Connections to Game Theory

The adversarial dynamics between the exploration policy (maximizing error) and the MM actor (minimizing error) can be formalized as a two-player zero-sum game. The existence and uniqueness of Nash equilibria in this setting, and the convergence rate of gradient-based dynamics to such equilibria, connect RLQF to the broader literature on generative adversarial networks and minimax optimization.

---

## Notation Summary

| Symbol | Definition |
|--------|-----------|
| $\mathcal{X} \subset \mathbb{R}^{3N}$ | Molecular configuration space |
| $E_0(\mathbf{R})$ | Exact QM potential energy surface |
| $V_\theta(\mathbf{R})$ | MM neural network potential (actor) |
| $C_\phi(\mathbf{R}, \tilde{E})$ | QM neural network critic |
| $\Delta(\mathbf{R}; \theta)$ | MM/QM gap at configuration $\mathbf{R}$ |
| $\pi_\psi$ | Exploration policy |
| $\mathcal{R}(s, a)$ | Reward function |
| $\gamma$ | Discount factor |
| $\beta$ | Entropy temperature |
| $\kappa$ | Prioritization exponent |
| $\alpha$ | Adversarial mixing ratio |
| $\epsilon_C$ | Critic approximation error |
| $L_E$ | Lipschitz constant of $E_0$ |

---

## References

1. Schütt, K. T., et al. "SchNet: A continuous-filter convolutional neural network for modeling quantum interactions." *NeurIPS*, 2017.
2. Batzner, S., et al. "E(3)-equivariant graph neural networks for data-efficient and accurate interatomic potentials." *Nature Communications*, 2022.
3. Batatia, I., et al. "MACE: Higher order equivariant message passing neural networks for fast and accurate force fields." *NeurIPS*, 2022.
4. Haarnoja, T., et al. "Soft Actor-Critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor." *ICML*, 2018.
5. Pathak, D., et al. "Curiosity-driven exploration by self-supervised prediction." *ICML*, 2017.
6. Simm, G. N. C., et al. "Reinforcement learning for molecular design guided by quantum mechanics." *ICML*, 2020.
7. Schaul, T., et al. "Prioritized experience replay." *ICLR*, 2016.
8. Smith, J. S., et al. "ANI-1: An extensible neural network potential with DFT accuracy at force field computational cost." *Chemical Science*, 2017.
9. Qiao, Z., et al. "OrbNet Denali: A machine learning potential for biological and chemical applications." *Journal of Chemical Physics*, 2022.
