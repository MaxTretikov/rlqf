# The RLQF Markov Decision Process

The formal MDP that structures the RLQF training loop: how configuration selection, MM evaluation, and parameter updates interact as a sequential decision problem.

**Parent:** [[index]]
**See also:** [[rlqf-objective]], [[critic-architecture]], [[mm-qm-gap]]

---

## MDP Definition

We define a Markov Decision Process $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \gamma)$ as follows.

### State Space

$\mathcal{S} = \mathcal{X} \times \Theta$, where $\mathcal{X} \subset \mathbb{R}^{3N}$ is the molecular configuration space (see [[mm-qm-gap]]) and $\Theta$ is the parameter space of the MM network. A state $s_t = (\mathbf{R}_t, \theta_t)$ encodes the current molecular configuration and the current MM parameters.

### Action Space

$\mathcal{A} = \mathcal{X}$. An action $a_t = \mathbf{R}_{t+1}$ is the selection of the next molecular configuration to evaluate. The policy $\pi_\psi: \mathcal{S} \to \mathcal{P}(\mathcal{A})$ is a stochastic mapping parameterized by $\psi$ that proposes configurations.

### Transition Dynamics

The transition function $\mathcal{T}$ is factored:

$$
\mathcal{T}(s_{t+1} | s_t, a_t) = \delta(\mathbf{R}_{t+1} = a_t) \cdot p(\theta_{t+1} | \theta_t, \mathbf{R}_{t+1})
$$

where $\delta$ is the Dirac delta (the configuration transitions deterministically to the chosen action) and $p(\theta_{t+1} | \theta_t, \mathbf{R}_{t+1})$ encodes the MM parameter update rule (e.g., a gradient descent step on the loss at $\mathbf{R}_{t+1}$):

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t, \mathbf{R}_{t+1})
$$

The specific form of $\mathcal{L}$ is detailed in [[energy-force-loss]] and [[kl-divergence-loss]].

### Reward Function

The reward at step $t$ is:

$$
\mathcal{R}(s_t, a_t) = C_\phi(\mathbf{R}_{t+1}, V_{\theta_t}(\mathbf{R}_{t+1})) - \lambda \cdot d(\mathbf{R}_{t+1}, \mathbf{R}_t)
$$

where $C_\phi$ is the [[critic-architecture|critic's]] error estimate and $\lambda \cdot d(\cdot, \cdot)$ is a regularization term penalizing large jumps in configuration space (encouraging smooth exploration). The first term rewards the agent for finding configurations where the MM network is maximally wrong; the second term prevents degenerate solutions.

### Discount Factor

$\gamma \in [0, 1)$ controls the planning horizon.

## Key Properties

The MDP has several unusual features compared to standard RL settings:

1. **Non-stationary transitions.** The parameter component $\theta_t$ evolves via gradient descent, making the transition dynamics non-stationary. This couples exploration and learning in a way that the [[convergence]] analysis must account for.

2. **Continuous, high-dimensional action space.** The action space $\mathcal{A} = \mathcal{X} \subset \mathbb{R}^{3N}$ is continuous, motivating the use of [[policy-gradient]] methods and [[soft-rlqf|entropy regularization]].

3. **Reward depends on actor parameters.** The reward $\mathcal{R}$ depends on $\theta_t$ through the critic, meaning the reward landscape shifts as the MM network improves. This is the mechanism by which RLQF naturally curricularizes training.
