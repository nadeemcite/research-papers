---
title: "Dueling DQN: Talk & Q&A"
parent: "053 Dueling DQN"
nav_order: 2
---
# TALK.md — Dueling Network Architectures for Deep Reinforcement Learning

## Press & Blog Coverage

1. [**Key Papers in Deep RL**](https://spinningup.openai.com/en/latest/spinningup/keypapers.html), OpenAI Spinning Up. Lists this paper among the key deep RL papers.

2. [**Rainbow: Combining Improvements in Deep Reinforcement Learning**](https://arxiv.org/abs/1710.02298), Hessel et al. A follow-up paper that combines the dueling architecture with other DQN improvements in one agent.

## Q&A (answers from the paper)

**Q1: What is the key insight?**
A: For many states it is unnecessary to estimate the value of each action choice. In Enduro, for example, moving left or right only matters when a collision is imminent. But for bootstrapping methods, the value of every state matters (Section 3).

**Q2: How is the network built?**
A: It keeps DQN's convolutional layers, then splits into two streams of fully connected layers: one estimates the state value V(s), the other the advantage of each action, A(s, a). An aggregating layer combines them into Q-values (Section 3).

**Q3: Why not simply add V and A?**
A: Given Q, you cannot recover V and A uniquely, and the paper says this lack of identifiability is mirrored by poor practical performance (Section 3).

**Q4: Why subtract the mean advantage instead of the max?**
A: The max form makes V an estimate of the state value. The mean form loses that meaning, since V and A are then off by a constant, but it increases the stability of optimization: the advantages only need to change as fast as their mean, instead of compensating for any change to the best action's advantage. A softmax version gave similar results to the mean, so all reported experiments use the mean (Section 3).

**Q5: Why does it help?**
A: With every update of the Q-values, the value stream is updated, while a single-stream network updates the value of only one action. The authors say this lets the network approximate state values better, and the gain grows when there are many actions (Section 5).

## Common Misconceptions

1. **"Dueling is a new RL algorithm."** It is a network architecture. The paper combines it with existing algorithms such as Double DQN and prioritized replay (Sections 4.2 and 5).

2. **"It needs two separate networks."** It is one network that shares its lower layers and splits into two streams near the output (Section 3).

3. **"V(s) in the mean form is the true state value."** With the mean aggregation, V and A lose their original meaning because they are off by a constant (Section 3).

4. **"The paper compared it against a smaller network."** For the Atari comparison, the authors retrained a single-stream baseline with 1,024 hidden units so that both networks have roughly the same number of parameters (Section 4.2).

## Real Citations

1. Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). *Dueling Network Architectures for Deep Reinforcement Learning.* ICML 2016. [arXiv:1511.06581](https://arxiv.org/abs/1511.06581)

2. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning.* Nature, 518, 529–533. [Nature article](https://www.nature.com/articles/nature14236)

3. van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep Reinforcement Learning with Double Q-learning.* AAAI 2016. [arXiv:1509.06461](https://arxiv.org/abs/1509.06461)

4. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). *Prioritized Experience Replay.* ICLR 2016. [arXiv:1511.05952](https://arxiv.org/abs/1511.05952)

5. Hessel, M., et al. (2018). *Rainbow: Combining Improvements in Deep Reinforcement Learning.* AAAI 2018. [arXiv:1710.02298](https://arxiv.org/abs/1710.02298)

6. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press. [Book page](http://incompleteideas.net/book/the-book-2nd.html)

**Citation count:** about 3,600 ([OpenAlex](https://openalex.org/W2173564293), 3,631 when checked on 14 November 2026).
