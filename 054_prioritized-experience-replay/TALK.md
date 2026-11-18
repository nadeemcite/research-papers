---
title: "Prioritized Replay: Talk & Q&A"
parent: "054 Prioritized Replay"
nav_order: 2
---
# TALK.md — Prioritized Experience Replay

## Press & Blog Coverage

1. [**Key Papers in Deep RL**](https://spinningup.openai.com/en/latest/spinningup/keypapers.html), OpenAI Spinning Up. Lists this paper among the key deep RL papers, under the algorithm name Prioritized Experience Replay.

2. [**Rainbow: Combining Improvements in Deep Reinforcement Learning**](https://arxiv.org/abs/1710.02298), Hessel et al. A follow-up paper that combines prioritized replay with other DQN improvements in one agent.

## Q&A (answers from the paper)

**Q1: Why use the TD error as the priority?**
A: The ideal criterion would be how much the agent can learn from a transition, which is not directly available. The magnitude of the TD error is a reasonable proxy: it measures how "surprising" a transition is, and Q-learning already computes it (Section 3.2). It can be a poor estimate in some cases, for example with noisy rewards, and Appendix A discusses alternatives.

**Q2: How do the authors make sampling fast?**
A: For the rank-based variant, they store transitions in a priority queue implemented as an array-based binary heap and use that array as an approximately sorted array. For the proportional variant, a "sum-tree" (every node holds the sum of its children) allows O(log N) updates and sampling (Section 3.3 and Appendix B.2.1).

**Q3: Rank-based or proportional?**
A: The authors expected the rank-based variant to be more robust, because it ignores outliers and error magnitudes. Perhaps surprisingly, both performed similarly in practice, which they suspect is due to DQN's heavy clipping of rewards and TD errors (Section 5).

**Q4: Why anneal β instead of fixing it at 1?**
A: Unbiased updates matter most near convergence at the end of training; earlier, the process is highly non-stationary anyway. So the importance-sampling exponent β is annealed linearly from its starting value to 1 by the end of learning (Section 3.4).

**Q5: How much did it help on Atari?**
A: Added to DQN, prioritized replay substantially improved the score on 41 of 49 games. Combined with Double DQN, the median normalized score went from 111% to 128% with proportional prioritization (113% with rank-based), and the mean from 418% to 551% (Section 4 and Table 1).

## Common Misconceptions

1. **"Prioritized replay always improves learning."** Added to DQN, it improved the score on 41 of 49 games, not all, and on some games one variant stays close to the Double DQN baseline while the other gives a big boost (Section 4). In [this repo's notebook](solution.ipynb), on a 6-cell gridworld, prioritized and uniform replay end up the same.

2. **"Only the sum-tree implements prioritized replay."** The rank-based variant uses stratified sampling over a heap-based, approximately sorted array instead (Section 3.3 and Appendix B.2.1).

3. **"Full prioritization (α = 1) is best."** The authors' coarse grid search found α = 0.6 (proportional) and α = 0.7 (rank-based), trading off aggressiveness and robustness (Section 4).

4. **"Importance-sampling weights only matter in theory."** The paper notes a practical benefit too: they shrink the gradient for high-error transitions, which helps with non-linear function approximation (Section 3.4).

## Real Citations

1. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). *Prioritized Experience Replay.* ICLR 2016. [arXiv:1511.05952](https://arxiv.org/abs/1511.05952)

2. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning.* Nature, 518, 529–533. [Nature article](https://www.nature.com/articles/nature14236)

3. van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep Reinforcement Learning with Double Q-learning.* AAAI 2016. [arXiv:1509.06461](https://arxiv.org/abs/1509.06461)

4. Lin, L.-J. (1992). *Self-improving reactive agents based on reinforcement learning, planning and teaching.* Machine Learning, 8, 293–321. [DOI:10.1007/BF00992699](https://link.springer.com/article/10.1007/BF00992699)

5. Moore, A. W., & Atkeson, C. G. (1993). *Prioritized sweeping: Reinforcement learning with less data and less time.* Machine Learning, 13, 103–130. [DOI:10.1007/BF00993104](https://link.springer.com/article/10.1007/BF00993104)

**Citation count:** about 4,540 ([OpenAlex](https://openalex.org/W2201581102), 4,541 when checked on 16 November 2026).
