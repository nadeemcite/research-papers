---
title: "Dueling DQN: Code architecture"
parent: "053 Dueling DQN"
nav_order: 1
---
# CODE_ARCHITECTURE.md — Dueling Network Architectures for Deep Reinforcement Learning

## Overview

The notebook implements a **standard DQN** and a **Dueling DQN** from scratch in PyTorch, trains both on CartPole-v1 with the same seed and hyperparameters, and compares their rewards, Q-values and greedy performance. The only difference between the agents is the network: the dueling network splits into a value stream and an advantage stream. It is a single-seed run, so the Summary cell reports what this run showed rather than a general result.

## Section-by-Section Breakdown

### 1. Setup & Imports
- Installs `gymnasium` quietly if needed; seeds `random`, NumPy and PyTorch with `SEED = 42`
- One CPU thread and deterministic PyTorch kernels, so results do not depend on the machine's core count
- Figures shown as JPEG at 80 dpi; on Kaggle an `atexit` hook copies the executed notebook to `executed_solution.ipynb`

### 2. Hyperparameters
- `STATE_DIM = 4`, `ACTION_DIM = 2`, `HIDDEN_DIM = 128`
- `LR = 1e-3`, `GAMMA = 0.99`, `BUFFER_SIZE = 10000`, `BATCH_SIZE = 64`
- `TARGET_UPDATE_FREQ = 10` (episodes between target-network copies)
- `EPS_START = 1.0`, `EPS_END = 0.05`, `EPS_DECAY = 0.995` (per episode)
- `NUM_EPISODES = 300`, `MAX_STEPS = 500`

### 3. Environment Setup
- CartPole-v1 and one seeded random episode for scale

### 4. Standard DQN Network (`DQNetwork`)
- `Linear(4, 128)` → ReLU → `Linear(128, 128)` → ReLU → `Linear(128, 2)`

### 5. Dueling DQN Network (`DuelingDQNetwork`)
- Shared layers: `Linear(4, 128)` → ReLU → `Linear(128, 128)` → ReLU
- Value stream: `Linear(128, 1)` → V(s); advantage stream: `Linear(128, 2)` → A(s, a)
- Aggregation: `Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')`, the mean form the paper uses in all its experiments
- `get_value_advantage()` returns V and A separately for the plots
- The cell prints the parameter counts: 17,410 for the standard network and 17,539 for the dueling one

### 6. Experience Replay Buffer (`ReplayBuffer`)
- Uniform random sampling of (s, a, r, s', done) transitions

### 7. ε-Greedy Action Selection (`select_action`)
- Random action with probability ε, otherwise the argmax of the online network's Q-values

### 8. Optimization Step (`optimize_step`)
- Same for both networks: target `r + γ · max_a' Q(s', a'; θ⁻)` from the target network, Huber loss, Adam

### 9. Training Loop (`train_agent`)
- Reseeds, then trains for 300 episodes; copies the target network every 10 episodes; decays ε each episode
- Records episode reward, mean loss and mean Q-value; prints the 50-episode average every 50 episodes

### 10–11. Train Standard DQN and Dueling DQN
- One call to `train_agent` per network, with the same seed

### 12. Comparison Plots
- 20-episode rolling reward, mean Q-value and loss per episode for both agents

### 13. Evaluation: Greedy Policy (`evaluate`)
- 10 episodes with ε = 0 and fixed seeds; prints mean, standard deviation and maximum

### 14. Value and Advantage Visualization
- V(s) and A(s, a) for 100 seeded sample states; prints their means, standard deviations and the mean gap between the two actions' advantages

### 15. Learning Speed Comparison
- First episode where the 10-episode rolling reward reaches 475, and the final 50-episode average

### 16. Summary
- `### What the run above shows`: the real numbers from the saved run, with a single-seed caveat

## Key Functions/Classes

| Component | Class/Function | Purpose |
|-----------|---------------|---------|
| DQNetwork | `DQNetwork(nn.Module)` | Standard Q-network with direct Q-value output |
| DuelingDQNetwork | `DuelingDQNetwork(nn.Module)` | Value and advantage streams aggregated into Q-values |
| ReplayBuffer | `ReplayBuffer` | Stores and randomly samples transitions |
| select_action | `select_action(...)` | ε-greedy action selection |
| optimize_step | `optimize_step(...)` | DQN optimization step (works with either network) |
| train_agent | `train_agent(...)` | Full training loop for either network |
| evaluate | `evaluate(...)` | Greedy evaluation episodes |

## Data Flow / Shapes

```
State (4,) → shared layers (128,) → value_stream → V(s) (1,)
                                  → advantage_stream → A(s,a) (2,)
Q(s,a) = V(s) + A(s,a) - mean(A)  → Q-values (2,) → argmax → action

(s, a, r, s', done) → ReplayBuffer → sample batch (64, ...)
  states (64, 4) → policy_net → gather at action → current_q (64,)
  next_states (64, 4) → target_net → max → next_q (64,)
  target = reward + gamma * next_q * (1 - done)
  loss = Huber(current_q, target)
```

## Deliberate Simplifications vs Full Paper

1. **Environment**: CartPole-v1 (4-dim state, 2 actions) instead of Atari 2600 (3 to 18 actions). The paper reports that the dueling advantage grows with the number of actions, so 2 actions is the least favorable case.
2. **Network**: MLP instead of the convolutional DQN network.
3. **Buffer size**: 10,000 instead of 1,000,000.
4. **Target update**: every 10 episodes instead of every 10,000 steps.
5. **Training length**: 300 episodes, one seed.
6. **No Double DQN, prioritized replay or gradient clipping**: the paper's Atari agents combine dueling with Double DQN and gradient clipping, and its best agent adds prioritized replay. This notebook compares the two networks with the same plain DQN update.
7. **No matched baseline**: the paper retrains a single-stream baseline with 1,024 hidden units so both networks have roughly the same number of parameters. Here the two networks differ by 129 parameters.
