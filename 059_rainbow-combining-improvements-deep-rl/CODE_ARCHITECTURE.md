---
title: "Rainbow: Code architecture"
parent: "059 Rainbow"
nav_order: 1
---
# Code Architecture — Rainbow: Combining Improvements in Deep Reinforcement Learning

## Overview

The notebook implements a simplified Rainbow agent in PyTorch that combines three of the paper's six DQN extensions: **Double DQN**, **dueling networks** and **proportional prioritized replay**. It trains five agents on CartPole-v1 with the same seed and settings (vanilla DQN, Double DQN, Dueling DQN, PER DQN and the combination) and compares their learning curves. A separate section trains Double DQN with 3-step returns. The notebook is self-contained.

## Section-by-Section Breakdown

### 1. Setup & Imports
- Installs `gymnasium` if missing; imports PyTorch, NumPy and matplotlib
- Seeds `random`, NumPy and PyTorch with `SEED = 42`; one CPU thread and `torch.use_deterministic_algorithms(True)` so reruns give the same numbers
- Figures as JPEG at 80 dpi; on Kaggle an `atexit` hook copies the executed notebook to `executed_solution.ipynb`

### 2. Hyperparameters
- CartPole-v1: state dimension 4, 2 actions
- Hidden size 128, Adam learning rate 1e-3, γ = 0.99, buffer 20,000, batch 64, hard target update every 200 steps
- ε from 1.0, ×0.995 per episode, floor 0.05; 250 episodes of at most 500 steps
- Prioritized replay: α = 0.6, β from 0.4 to 1.0 over the first 60% of training, priority constant 1e-6
- `N_STEP = 3` for the multi-step section

### 3. Environment Setup
- Creates CartPole-v1, prints the spaces and runs one seeded random episode

### 4. Q-Network Architectures
- `QNetwork`: `Linear(4, 128)` → ReLU → `Linear(128, 128)` → ReLU → `Linear(128, 2)`
- `DuelingQNetwork`: the same two-layer trunk, then a value head `Linear(128, 1)` and an advantage head `Linear(128, 2)`, combined as `Q = V + A − mean_a A`

### 5. Replay Buffers
- `ReplayBuffer`: a `deque` with uniform sampling
- `SumTree`: leaves hold priorities and internal nodes hold sums, for O(log n) updates and sampling
- `PrioritizedReplayBuffer`: new transitions get the maximum priority so far; `sample` splits the total priority into equal segments and draws one transition from each; importance-sampling weights `(N · P(i)) ** −β`, divided by their maximum; `update_priorities` sets `(\|δ\| + 1e-6) ** α`; `anneal_beta` moves β towards 1

### 6. Action Selection
- `select_action`: ε-greedy over the policy network's Q-values

### 7. Optimization Steps
- `optimize_vanilla_dqn`: target `r + γ · max_a' Q_target(s', a')`
- `optimize_double_dqn`: the online network picks `a*`, the target network evaluates it
- `optimize_dueling_dqn`: the Double DQN update (used with the dueling network)
- `optimize_per_dqn`: Double DQN target, Huber loss weighted by the importance-sampling weights, then priority updates
- `optimize_rainbow`: the same update as `optimize_per_dqn`; the combination differs from PER DQN only by its network
- All use the Huber (smooth L1) loss

### 8. Training Loop (`train_agent`)
- Reseeds, then picks the network and buffer from the agent type: dueling network for `'dueling'` and `'rainbow'`, prioritized buffer for `'per'` and `'rainbow'`
- One optimization step per environment step; β annealed by episode progress for prioritized agents; target network copied every 200 steps
- Returns the reward of every episode and prints the 50-episode average every 50 episodes

### 9. Train All Agents
- Trains the five agents in order and prints each one's final 50-episode average and best episode

### 10. Comparison & Visualization
- Raw and 20-episode moving-average learning curves for the five agents

### 11. Summary Statistics
- Table of the final 50-episode mean and standard deviation, best and last episode for each agent, the agent with the highest final average, and each agent's difference from vanilla DQN

### 12. Optional Extension: Multi-step Returns
- `NStepReplayBuffer`: keeps the last n transitions and stores `(s_t, a_t, Σ γ^k r_{t+k}, s_{t+n}, done)`, cutting the sum at episode ends
- `train_nstep_double_dqn`: Double DQN with target `R_n + γ^n · Q_target(s_{t+n}, a*)`

### 13. Final Comparison (Including Multi-step)
- Smoothed curves for all six agents and a final table

### 14. Summary
- `### What the run above shows`: the saved run's real numbers, with a note on the PER/Rainbow bug fixed in this version

## Key Functions/Classes

| Name | Purpose |
|---|---|
| `QNetwork` | Standard MLP Q-network |
| `DuelingQNetwork` | Value and advantage streams combined into Q-values |
| `ReplayBuffer` | Uniform replay |
| `SumTree` | Priority sums for O(log n) sampling |
| `PrioritizedReplayBuffer` | Proportional prioritized replay with importance-sampling weights |
| `select_action` | ε-greedy action selection |
| `optimize_*` | One update step per agent type |
| `train_agent` | Trains one of the five agents and returns its episode rewards |
| `NStepReplayBuffer`, `train_nstep_double_dqn` | Double DQN with 3-step returns |

## Data Flow / Shapes

```
state (4,) → network → Q-values (2,) → ε-greedy action
(s, a, r, s', terminated) → buffer.push
buffer.sample(64) → states (64, 4), actions (64,), rewards (64,), next_states (64, 4), dones (64,)
                    [+ indices (64,), weights (64,) for prioritized replay]
a* = argmax Q_online(s')  →  target = r + 0.99 · Q_target(s', a*) · (1 − done)
loss = mean(w · Huber(Q(s, a), target))  →  Adam step  →  update_priorities(indices, δ)
```

## Deliberate Simplifications vs. the Paper

1. **Components:** three of six; no distributional RL and no noisy nets. Multi-step returns appear only in the separate Double DQN section, not in the combined agent.
2. **Exploration:** ε-greedy for every agent instead of noisy nets.
3. **Priorities:** absolute TD error with α = 0.6; the paper uses the distributional KL loss with ω = 0.5.
4. **Environment and network:** CartPole-v1 and a small MLP instead of 57 Atari games and a convolutional network.
5. **Scale:** 250 episodes and one seed instead of 200M frames per game.
6. **Optimizer:** Adam with learning rate 1e-3; the paper used Adam with learning rate 0.0000625 and ε = 1.5 × 10⁻⁴.
