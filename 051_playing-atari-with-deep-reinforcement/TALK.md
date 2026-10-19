---
title: "DQN: Talk & Q&A"
parent: "051 DQN"
nav_order: 2
---
# TALK.md — Playing Atari with Deep Reinforcement Learning (DQN)

## Press / Blog Coverage

1. **Nature — "Human-level control through deep reinforcement learning" (2015):** The follow-up paper published in Nature, where DQN was scaled to 49 Atari games and achieved human-level performance on the majority. This paper brought deep RL to mainstream scientific attention and is one of the most cited AI papers ever. (URL: https://www.nature.com/articles/nature14236)

2. **Wired — "Google's AI Is Now Smart Enough to Play Atari Like the Pros" (2015):** Wired's story on the 2015 Nature paper, in which DeepMind's system learned to play 49 Atari games from screen pixels and the score alone. (URL: https://www.wired.com/2015/02/google-ai-plays-atari-like-pros/)

3. **Andrej Karpathy — "Deep Reinforcement Learning: Pong from Pixels" (2016):** Karpathy's influential blog post implementing a simplified RL agent for Atari Pong from raw pixels, explaining the core concepts of policy gradients and deep RL. While using policy gradients rather than Q-learning, it popularized deep RL implementations and referenced DQN as foundational. (URL: https://karpathy.github.io/2016/05/31/rl/)

4. **OpenAI Spinning Up — "Part 2: Kinds of RL Algorithms" (2018):** OpenAI's guide to deep RL. This page sorts RL algorithms into families and places DQN among the Q-learning (value-based) methods. (URL: https://spinningup.openai.com/en/latest/spinningup/rl_intro2.html)

## Q&A (answers from the paper)

**Q1: What was the key insight that made deep Q-learning work when previous attempts at neural network function approximation in RL had failed?**
A1 (from the paper, Sections 1 and 4): The authors point to two problems: back-to-back samples are strongly correlated, and the data distribution keeps changing as the agent learns new behaviour. Their answer is experience replay: store past transitions and train on random samples of them, which breaks the correlations and smooths out the changing distribution. The 2013 paper does not use a separate target network; its targets use the network weights from the previous iteration (Section 2). The separate, periodically copied target network came in the 2015 Nature follow-up.

**Q2: Why did the authors use the same architecture and hyperparameters across all seven games?**
A2 (from the paper, Section 5): The authors wanted to demonstrate that their method was general — not a collection of game-specific solutions. By using the same CNN architecture, the same learning rate, the same replay buffer size, and the same exploration schedule across all seven games, they showed that DQN is a general-purpose deep RL algorithm. This generality was a key selling point: previous approaches required game-specific feature engineering or tuning.

**Q3: How does experience replay help, beyond just storing data?**
A3 (from the paper, Section 4): The paper lists three advantages: (1) each transition can be used in many weight updates, which is more data-efficient; (2) random samples break the strong correlations between consecutive samples, which reduces the variance of the updates; and (3) the behaviour distribution is averaged over many past states, which avoids feedback loops, oscillations or divergence. The random sampling from the buffer ensures that the training distribution is closer to i.i.d., which is what SGD assumes.

**Q4: How does the paper compute its Q-learning targets?**
A4 (from the paper, Section 2): The target is y = r + γ max_a' Q(s', a'; θ_{i−1}), using the parameters from the previous iteration, which are held fixed while the loss for the current iteration is optimized. The paper notes that, unlike supervised learning, these targets depend on the network weights. There is no separate target network in the 2013 paper. The 2015 Nature version added one: a frozen copy θ⁻ that is updated only every C = 10,000 steps, so the targets stay fixed between copies. This notebook uses that later recipe (with C = 100).

**Q5: What were the limitations of this initial DQN paper?**
A5 (from the paper, Sections 4 and 5, plus follow-up work): Section 5 says the agent is far from human performance on Q*bert, Seaquest and Space Invaders, because these games need a strategy that extends over long time scales. Section 4 notes that the replay memory treats all transitions as equally important and always overwrites the oldest ones, and suggests smarter sampling. Section 5 also notes that clipping rewards to −1, 0 and +1 means the agent cannot tell rewards of different size apart.

The follow-up Nature paper (2015) scaled DQN to 49 games. Subsequent work addressed Q-value overestimation (Double DQN), better architecture (Dueling DQN), better sampling (Prioritized Replay), and distributional Q-learning (C51). The 2013 paper was the proof of concept; the field spent years improving upon it.

## Common Misconceptions

1. **"DQN uses policy gradients."** No. DQN is a **value-based** method — it learns the action-value function Q(s, a) and derives a policy by acting greedily with respect to Q. It does not directly parameterize or optimize a policy. Policy gradient methods (REINFORCE, PPO, A3C) are a different family of RL algorithms that directly optimize the policy. DQN combines Q-learning (a value-based method) with deep neural networks.

2. **"The target network is updated by gradient descent."** No. The target network's parameters θ⁻ are **hard-copied** from the policy network θ periodically (every C steps). The target network does not receive gradients — it is a frozen snapshot used only for computing training targets. This is what makes it "stable": between copies, its outputs don't change regardless of what the policy network learns. In this repo's CartPole ablation, refreshing the target at every step (so it is just the live network) left all 5 seeds at about 9.6 reward for all 600 episodes, so they never learned.

3. **"Experience replay is just for data efficiency."** While replay does improve data efficiency (each transition is used multiple times), its primary purpose is **stabilization**. Without replay, consecutive training samples are highly correlated (they come from a single trajectory), which violates the i.i.d. assumption of SGD and causes unstable updates. Random sampling from the buffer decorrelates the samples, making training much more stable. Be careful with how far this carries over to easy tasks: in this repo's small CartPole ablation, a buffer holding only the last 64 transitions did not do worse than the full 10,000-transition buffer (5 seeds each, mean 351.6 vs 269.9, within the large seed-to-seed spread). The paper's argument is about Atari, which the notebook does not test.

4. **"DQN learns the optimal policy directly."** DQN learns the optimal **Q-function** (action values), not the policy directly. The policy is extracted by acting greedily: π(s) = argmax_a Q(s, a). In practice, ε-greedy is used during training for exploration. The distinction matters: the network approximates values, and the policy is a simple function of those values.

5. **"You need a CNN for DQN."** The original paper used a CNN because the input was raw pixels (84×84×4 images). But the DQN algorithm — experience replay, target network, Q-learning loss — works with any function approximator. For low-dimensional state spaces like CartPole (4-dim), a simple MLP is sufficient and more appropriate. The architecture should match the input modality; the algorithm is architecture-agnostic.

## Real Citations

1. Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). "Playing Atari with Deep Reinforcement Learning." NIPS Deep Learning Workshop. arXiv:1312.5602. — The original DQN paper.

2. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., et al. (2015). "Human-level control through deep reinforcement learning." Nature, 518(7540), 529–533. — The Nature follow-up scaling DQN to 49 Atari games with human-level performance. One of the most cited AI papers ever.

3. van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep Reinforcement Learning with Double Q-learning." AAAI 2016. arXiv:1509.06461. — Double DQN, addressing Q-value overestimation in DQN by decoupling action selection and evaluation.

4. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). "Prioritized Experience Replay." ICLR 2016. arXiv:1511.05952. — Extends DQN's uniform replay buffer with TD-error-based prioritized sampling for more efficient learning.

5. Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). "Dueling Network Architectures for Deep Reinforcement Learning." ICML 2016. arXiv:1511.06581. — Introduces the dueling Q-network architecture that separates state value and action advantage estimation.
