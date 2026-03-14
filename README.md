# Multi-Armed Bandits: Algorithm Comparison

This repository implements several classic **Multi-Armed Bandit (MAB)** algorithms and compares their performance through simulation experiments. The project reproduces classical bandit experiments described in *Reinforcement Learning: An Introduction* (Sutton & Barto).

---

## Project Overview

The multi-armed bandit problem models a situation where an agent must repeatedly choose among several actions (arms), each producing a stochastic reward. The goal is to maximize the cumulative reward by balancing the **Exploration–Exploitation tradeoff**:

* **Exploration:** Trying new actions to gather information about their reward distributions.
* **Exploitation:** Choosing the best-known action to maximize immediate reward.

---

## Algorithms Implemented

### 1. $\epsilon$-Greedy
The $\epsilon$-greedy strategy selects:
* A **random action** with probability $\epsilon$.
* The **best estimated action** (greedy) with probability $1 - \epsilon$.

**Update rule for action-value estimates:**
$$Q(a) \leftarrow Q(a) + \frac{1}{N(a)}(R - Q(a))$$

Two configurations are tested: $\epsilon = 0.1$ and $\epsilon = 0.01$.

### 2. Upper Confidence Bound (UCB)
UCB balances exploration and exploitation using confidence intervals, favoring actions that have high uncertainty.

**Action selection rule:**
$$A_t = \arg\max_a \left[ Q(a) + c \sqrt{\frac{\ln t}{N(a)}} \right]$$

Where:
* $Q(a)$: Estimated value of action $a$.
* $N(a)$: Number of times action $a$ was chosen.
* $t$: Current timestep.
* $c$: Exploration parameter (controls the degree of exploration).

### 3. Thompson Sampling
A Bayesian approach where each arm maintains a probability distribution over its success rate. For Bernoulli rewards, we use the **Beta distribution**:

$$p_a \sim \text{Beta}(\alpha_a, \beta_a)$$

**Procedure:**
1. Sample a probability for each arm from its respective Beta distribution.
2. Choose the arm with the highest sampled value.
3. Update the posterior ($\alpha$ or $\beta$) based on the observed success or failure.

---

## Environments

The simulation includes two distinct bandit environments:

### Gaussian Bandit
Used primarily for **$\epsilon$-greedy** and **UCB** tests.
* **True action values:** $q^*(a) \sim \mathcal{N}(0, 1)$
* **Actual rewards:** $R_t \sim \mathcal{N}(q^*(a), 1)$

### Bernoulli Bandit
Used primarily for **Thompson Sampling**.
* **Success probability:** Each arm has a probability $p_a \sim \text{Uniform}(0, 1)$.
* **Rewards:** Binary ($1$ with probability $p_a$, else $0$).

---

## Experiment Setup

The simulations are conducted using the following standard parameters:

| Parameter | Value |
| :--- | :--- |
| **Number of Arms** | 10 |
| **Number of Runs** | 2000 |
| **Steps per Run** | 1000 |

### Evaluation Metrics
During each run, the agent's performance is tracked via:
1.  **Average Reward:** The mean reward obtained at each timestep across all runs.
2.  **% Optimal Action:** The frequency with which the agent selects the arm with the true highest mean.
