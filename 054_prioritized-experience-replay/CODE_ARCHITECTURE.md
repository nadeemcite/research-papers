---
title: "Prioritized Replay: Code architecture"
parent: "054 Prioritized Replay"
nav_order: 1
---
# Code Architecture — Prioritized Experience Replay

## Overview

The notebook implements prioritized experience replay from scratch in NumPy: a sum-tree, a proportional prioritized buffer with importance-sampling weights, a uniform buffer for comparison, a small Q-network with hand-written backpropagation and a DQN agent. Two agents, identical except for the buffer, train on a 6-cell gridworld, and the notebook compares their learning curves, TD errors and buffer priorities.

## Section-by-Section Breakdown

### 1. Setup & Imports
- NumPy, `random`, matplotlib; one BLAS thread so results do not depend on the machine's core count
- Seeds `random` and NumPy with `SEED = 42`
- Figures as JPEG at 80 dpi; on Kaggle an `atexit` hook copies the executed notebook to `executed_solution.ipynb`

### 2. Sum-Tree (`SumTree`)
- Array of `2 * capacity - 1` nodes: leaves hold priorities, internal nodes hold the sum of their children
- `add(priority, data)` writes into a circular buffer of leaves; `update(leaf, priority)` propagates the change up to the root
- `get(value)` walks down from the root to the leaf whose cumulative range contains `value`
- O(log N) updates and sampling

### 3. Prioritized Replay Buffer (`PrioritizedReplayBuffer`)
- New transitions get the current maximum priority, so each is replayed at least once
- `sample(batch_size)`: splits the total priority into `batch_size` equal segments and samples one value from each (stratified sampling)
- Importance-sampling weights `(N * P(i)) ** -beta`, divided by their maximum; β is annealed linearly from 0.4 to 1.0 over `beta_anneal_steps` samples
- `update_priorities(indices, td_errors)`: priority `(\|δ\| + 1e-6) ** alpha`, capped at 1e4

### 4. Uniform Replay Buffer (`UniformReplayBuffer`)
- A `deque` with uniform random sampling and all weights equal to 1

### 5. Toy Environment (`SimpleGridworld`)
- 6 cells in a line; the agent starts at cell 0 and the goal is cell 5
- State: one-hot vector of length 6; actions: left (0) or right (1)
- Reward +1 at the goal, −0.01 for every other step; the episode ends at the goal or after 30 steps

### 6. Q-Network (`QNetwork`, NumPy)
- MLP 6 → 64 → 64 → 2 with ReLU and He initialization
- `forward(states)` caches activations; `backward(td_errors, actions, weights)` applies the gradient of the weighted squared TD error, `-2 · w · δ`, to the taken action only, clips gradients to [−10, 10] and takes an SGD step
- `copy_from(other)` copies weights into the target network

### 7. DQN Agent (`DQNAgent`)
- ε-greedy action selection (ε from 1.0, ×0.998 per learning step, floor 0.1)
- `learn()`: samples a batch, computes targets `r + γ · max_a' Q_target(s', a')`, clips TD errors to [−100, 100], runs backprop with the importance-sampling weights, updates priorities, and copies the target network every 100 steps

### 8. Training Loop (`train_agent`)
- Reseeds, builds the environment and agent (learning rate 0.01, γ = 0.95, buffer 5,000, batch 32, α = 0.6, β from 0.4 over 5,000 steps), and trains for 500 episodes with one learning step per environment step
- Records episode rewards, the last mean \|TD error\| of each episode, β and the number of successes; prints a summary every 100 episodes

### 9. Visualization
- Smoothed learning curves, mean \|TD error\| per episode, and β over episodes

### 10. Sample Efficiency Comparison
- Episodes until the 30-episode average reward first reaches 0.5, total successes and the final 50-episode average

### 11. Buffer Priority Distribution
- Histogram of the final priorities and the sorted sampling probabilities in the prioritized buffer

### 12. Summary
- `### What the run above shows`: the saved run's real numbers, the paper's Atari results for context, and a note on the gradient-sign bug fixed in this version

## Key Functions/Classes

| Name | Purpose |
|---|---|
| `SumTree` | Priority sums for O(log N) sampling and updates |
| `PrioritizedReplayBuffer` | Proportional prioritized sampling with importance-sampling weights |
| `UniformReplayBuffer` | Baseline with uniform sampling |
| `SimpleGridworld` | 6-cell 1D environment |
| `QNetwork` | NumPy MLP with manual forward and backward passes |
| `DQNAgent` | ε-greedy DQN with a target network and a configurable buffer |
| `train_agent` | Trains one agent and returns its training history |

## Data Flow / Shapes

```
state (6,) one-hot → QNetwork → Q-values (2,) → ε-greedy action
(s, a, r, s', done) → buffer.add
buffer.sample(32) → states (32, 6), actions (32,), rewards (32,), next_states (32, 6), dones (32,), weights (32,)
targets = r + 0.95 · max Q_target(s') · (1 − done)
δ = targets − Q(s, a)  → backward(δ, actions, weights) → update_priorities(indices, δ)
```

## Deliberate Simplifications vs. the Paper

1. **Environment:** a 6-cell gridworld instead of Atari games.
2. **Network:** a NumPy MLP with plain SGD instead of the DQN convolutional network.
3. **Algorithm:** plain DQN targets, not Double DQN; rewards are not clipped to [−1, 1] as in the paper, and TD errors are clipped at ±100 rather than ±1.
4. **Variant:** only proportional prioritization; no rank-based variant.
5. **Scale:** a buffer of 5,000 instead of 1,000,000, and 500 short episodes.
6. **Step size:** the same learning rate for both agents; the paper reduced the step size by a factor of 4 for prioritized replay.
