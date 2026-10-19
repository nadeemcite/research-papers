---
title: "DQN: Code architecture"
parent: "051 DQN"
nav_order: 1
---
# CODE_ARCHITECTURE.md — Playing Atari with Deep Reinforcement Learning (DQN)

## Overview

The notebook implements a Deep Q-Network (DQN) from scratch following Mnih et al. (2013). The implementation includes the core parts of the paper: a Q-network (a neural network that estimates action values), experience replay (a replay buffer with random minibatch sampling), and ε-greedy exploration.

It also adds a target network (a periodically updated copy for stable targets) and a Huber loss, both from the 2015 Nature follow-up. We train on CartPole-v1 (a classic control environment with 4-dim state and 2 actions) using an MLP, which preserves all algorithmic components while being CPU-runnable. The notebook plots the reward curve over episodes and runs the trained agent. In the saved run the agent partly learns the task but does not solve it (greedy average 173.1 out of 500).

## Section-by-Section Breakdown

### 1. Setup & Imports
- **Purpose**: Import PyTorch, gymnasium (RL environments), matplotlib, numpy. Set random seeds for reproducibility. Configure device (CPU or CUDA).
- **Key detail**: Uses `gymnasium` (the maintained successor to OpenAI Gym). Only pip-installable packages (torch, gymnasium, matplotlib, numpy).

### 2. Hyperparameters
- **Purpose**: Centralize all training configuration.
- **Key values**:
  - `state_dim = 4` — CartPole observation dimension (cart position, velocity, pole angle, angular velocity)
  - `action_dim = 2` — CartPole actions (push left, push right)
  - `hidden_dim = 128` — hidden layer size for the Q-network
  - `lr = 2.5e-4` — learning rate for Adam optimizer
  - `gamma = 0.99` — discount factor
  - `buffer_size = 10000` — replay buffer capacity (the paper used 1M frames)
  - `batch_size = 64` — minibatch size for training
  - `target_update_freq = 100` — steps between target network updates (the 2015 Nature version used 10,000; the 2013 paper had no separate target network)
  - `epsilon_start = 1.0` — initial exploration rate
  - `epsilon_end = 0.01` — minimum exploration rate (not reached: ε is about 0.049 after 600 episodes)
  - `epsilon_decay = 0.995` — per-episode decay rate
  - `num_episodes = 600` — training episodes
  - `max_steps = 500` — max steps per episode (CartPole-v1 cap)
  - `min_buffer_size = 500` — transitions collected before training starts
  - `grad_clip = 10.0` — gradient norm clipping
- **Data flow**: These constants flow into network construction, replay buffer, and training loop.

### 3. Environment Setup
- **Purpose**: Create the CartPole-v1 environment and inspect observation/action spaces.
- **Key functions**: `gymnasium.make('CartPole-v1')`
- **Shapes**:
  - Observation: `(4,)` — [cart position, cart velocity, pole angle, pole angular velocity]
  - Action: discrete, 2 values (0=left, 1=right)
- **Simplification vs paper**: The paper used Atari 2600 games with 84×84×4 pixel inputs and 4–18 actions. We use CartPole with 4-dim state and 2 actions for CPU tractability.

### 4. Q-Network
- **Purpose**: Neural network that approximates Q(s, a) for all actions simultaneously.
- **Architecture**: MLP with 3 layers
  - Input: `(batch_size, 4)` — state vector
  - Hidden: `(batch_size, 128)` — with ReLU
  - Hidden: `(batch_size, 128)` — with ReLU
  - Output: `(batch_size, 2)` — Q-value for each action (no activation)
- **Key class**: `QNetwork(nn.Module)` with `forward(state)` returning Q-values for all actions
- **Data flow**: state → linear → ReLU → linear → ReLU → linear → Q-values (one per action)
- **Key detail**: The network outputs Q-values for ALL actions in one forward pass, so `argmax` over the output gives the greedy action. This is more efficient than passing each action separately.
- **Simplification vs paper**: The 2013 paper used a CNN (2 conv layers + 1 fully connected hidden layer of 256 units) for pixel inputs; 3 conv + 2 FC is the 2015 Nature version. We use an MLP for the 4-dim CartPole state. The Q-learning algorithm is identical.

### 5. Experience Replay Buffer
- **Purpose**: Store transitions (s, a, r, s', done) and sample random minibatches to decorrelate training updates.
- **Key class**: `ReplayBuffer` with `push(state, action, reward, next_state, done)` and `sample(batch_size)` methods
- **Implementation**: Uses `collections.deque` with maxlen for automatic eviction of old transitions
- **Data flow**: Agent collects transitions → buffer → random sampling → training minibatch
- **Key detail**: The `sample()` method returns tensors ready for the network. The `done` flag is used to zero out the target for terminal states (no future reward).

### 6. Target Network
- **Purpose**: A frozen copy of the Q-network used to compute stable training targets.
- **Mechanism**: `target_net.load_state_dict(policy_net.state_dict())` every `target_update_freq` steps
- **Key detail**: The target network's parameters θ⁻ are not updated by gradients — they are periodically hard-copied from the policy network θ. This prevents the "moving target" problem where the network bootstraps from its own rapidly changing estimates.

### 7. ε-Greedy Action Selection
- **Purpose**: Balance exploration and exploitation.
- **Mechanism**: With probability ε, select a random action; otherwise select argmax_a Q(s, a; θ)
- **Schedule**: ε starts at 1.0 (full exploration) and is multiplied by 0.995 after each episode, with a floor of 0.01. After 600 episodes it is about 0.049, so the floor is never reached.
- **Key function**: `select_action(state, policy_net, epsilon, action_dim)`

### 8. Training Loop
- **Purpose**: The main DQN training loop: Algorithm 1 from the paper plus the target network from the 2015 Nature version. The loop is inline in the cell; each update calls `optimize_model`.
- **Per-episode flow**:
  ```
  for each episode:
    reset environment → initial state s
    for each step:
      1. Select action a via ε-greedy
      2. Execute a, observe reward r and next state s'
      3. Store (s, a, r, s', done) in replay buffer
      4. If buffer has at least min_buffer_size (500) samples, call optimize_model:
         a. Sample random minibatch from buffer
         b. Compute current Q-values: Q(s, a; θ)
         c. Compute target: y = r + γ * max_a' Q(s', a'; θ⁻) * (1 - done)
         d. Loss = Huber / smooth L1 (y, Q(s, a; θ))
         e. Backprop, clip gradient norm at 10, update θ (Adam)
      5. Every target_update_freq (100) steps: copy θ → θ⁻
      6. s = s'
    decay ε (once per episode)
    log episode reward
  ```
- **Tracking**: Episode rewards, epsilon values, losses logged per episode
- **Key detail**: The target computation uses the target network (θ⁻), not the policy network (θ). The `done` flag zeros the future reward term for terminal states.

### 9. Visualization: Reward Curve
- **Purpose**: Plot episode reward over training. Should show an upward trend as the agent learns to balance the pole longer.
- **Output**: Raw reward curve + smoothed (moving average) curve. CartPole-v1 max reward is 500.

### 10. Visualization: Exploration Decay
- **Purpose**: Plot ε over episodes showing the exploration-to-exploitation transition.

### 11. Visualization: Loss Curve
- **Purpose**: Plot the Q-learning loss over training steps. Should decrease as Q-values stabilize.

### 12. Evaluation
- **Purpose**: Run the trained agent with ε=0 (pure greedy) for several episodes and report average reward.
- **Key detail**: This measures the agent's learned policy without exploration noise.

## Key Functions/Classes Summary

| Component | Type | Input Shape | Output Shape | Notes |
|-----------|------|-------------|--------------|-------|
| `QNetwork` | nn.Module | (B, 4) | (B, 2) | 3-layer MLP, outputs Q-values per action |
| `ReplayBuffer` | Class | — | — | Stores transitions, samples random minibatches |
| `select_action` | Function | (4,) | int | ε-greedy action selection |
| `optimize_model` | Function | replay buffer | scalar | One DQN update: Huber (smooth L1) loss between target and predicted Q-values, gradient clipping at 10, Adam step |

## Data Flow Diagram

```
State s ──► ε-greedy ──► Action a ──► Environment
                                         |
                                    Reward r, Next state s'
                                         |
                                    Store (s,a,r,s',done) in Replay Buffer D
                                         |
                              ┌──────────┴──────────┐
                              |                      |
                         Sample batch            (if buffer large enough)
                              |
                    ┌─────────┴─────────┐
                    |                   |
              Policy Net θ         Target Net θ⁻
              Q(s,a; θ)            Q(s',a'; θ⁻)
                    |                   |
                    |     max_a'        |
                    |        |          |
                    |   y = r + γ·max   |
                    |        |          |
                    └─ Huber(y, Q) ─────┘
                              |
                         Backprop θ
                              |
                    Every C steps: θ⁻ = θ
```

## Deliberate Simplifications vs Full Paper

1. **Environment**: CartPole-v1 (4-dim state, 2 actions) instead of Atari 2600 (84×84×4 pixels, 4–18 actions) — CPU tractable
2. **Network**: MLP instead of CNN — appropriate for the low-dimensional CartPole state
3. **Replay buffer**: 10,000 capacity instead of 1,000,000 — sufficient for CartPole
4. **Target update frequency**: Every 100 steps instead of 10,000 (2015 Nature version; the 2013 paper had no separate target network) — fits the smaller problem
5. **Frame preprocessing**: None — CartPole provides a clean 4-dim state; Atari requires frame stacking, grayscaling, and resizing
6. **Training length**: 600 episodes instead of 10 million frames
7. **No frame skipping**: Atari DQN uses frame skip of 4; not needed for CartPole
8. **Algorithm preserved**: Experience replay, ε-greedy and the Q-learning target follow the paper; the target network and Huber loss follow the 2015 Nature version
