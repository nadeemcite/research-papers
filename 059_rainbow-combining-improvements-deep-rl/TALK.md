---
title: "Rainbow: Talk & Q&A"
parent: "059 Rainbow"
nav_order: 2
---
# TALK.md — Rainbow: Combining Improvements in Deep Reinforcement Learning

## Press & Blog Coverage

1. [**Key Papers in Deep RL**](https://spinningup.openai.com/en/latest/spinningup/keypapers.html), OpenAI Spinning Up. Lists the paper among the key deep RL papers on deep Q-learning, under the algorithm name Rainbow DQN.

2. [**Dopamine**](https://github.com/google/dopamine), Google's open-source research framework for deep RL. Its README lists Rainbow (Hessel et al., 2018) among the agents it implements, and its [paper](https://arxiv.org/abs/1812.06110) describes the framework.

## Q&A (answers from the paper)

**Q1: Why combine the extensions at all?**
A: Several independent improvements to DQN had been proposed, but it was unclear which of them are complementary and can be combined (Abstract). They address very different issues and build on a shared framework, so they could plausibly be combined, and only a few pairs had been tried before, such as Prioritized DDQN and Dueling DDQN (Introduction).

**Q2: How do the pieces fit into one loss?**
A: Rainbow replaces the 1-step distributional loss with a multi-step version: the target distribution is shifted by the n-step return and shrunk by the n-step discount. Following double Q-learning, the online network picks the bootstrap action in the state n steps later and the target network evaluates it. The network is a dueling network adapted to output return distributions, and all its linear layers are noisy (Section "The Integrated Agent").

**Q3: What does Rainbow use as the replay priority?**
A: Not the absolute TD error, as in standard proportional prioritized replay, but the KL loss that the distributional agent minimizes. The authors suggest this may be more robust in noisy stochastic environments (Section "The Integrated Agent").

**Q4: How well did it do on Atari?**
A: It matched DQN's final performance after 7M frames and surpassed the best final performance of the baselines within 44M frames (Figure 1 and Section "Analysis"). After training, its median human-normalized score was 223% in the no-ops regime and 153% with human starts (Table 2).

**Q5: Which components mattered most?**
A: In the ablations, removing one component at a time, prioritized replay and multi-step learning were the two most crucial: removing either caused a large drop in median performance. Distributional Q-learning came next. Removing noisy nets made aggregate performance worse. Removing the dueling network made no significant difference in aggregate, and the median difference without double Q-learning was limited; both helped on some games and hurt on others (Section "Ablation studies", Figures 3 and 4).

**Q6: Why did double Q-learning matter so little?**
A: The return distribution's support only spans −10 to +10, and actual returns are often higher than 10, which leads to underestimation rather than overestimation. The authors hypothesize that this clipping counteracts Q-learning's overestimation bias, and note that double Q-learning may matter more if the support is expanded (Section "Ablation studies").

## Common Misconceptions

1. **"Rainbow is a new learning algorithm."** It integrates six existing extensions of DQN into one agent and studies how they combine (Abstract). Its new pieces are the integration details, such as the multi-step distributional loss and KL-based priorities.

2. **"All six components help equally."** The ablations show large drops without prioritized replay or multi-step learning, but no significant aggregate difference without the dueling network and only a limited one without double Q-learning (Section "Ablation studies", Figure 3).

3. **"Removing any component hurts every game."** Figure 4 shows that removing noisy nets gave small increases on some games, and that dueling and double Q-learning helped or hurt depending on the game.

4. **"Double Q-learning is useless in Rainbow."** The paper links its small effect to the clipped [−10, 10] value support, and says it may become more important if the support is expanded (Section "Ablation studies").

5. **"A combination is better at every point of training."** In [this repo's notebook](solution.ipynb), on CartPole-v1 with one seed, the three-part combination finished with the highest 50-episode average (87.2, against 60.0 for PER DQN alone), but it trailed most single extensions between episodes 100 and 200.

## Real Citations

1. Hessel, M., Modayil, J., van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., Horgan, D., Piot, B., Azar, M., & Silver, D. (2018). *Rainbow: Combining Improvements in Deep Reinforcement Learning.* AAAI 2018. [arXiv:1710.02298](https://arxiv.org/abs/1710.02298)

2. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning.* Nature, 518, 529–533. [Nature article](https://www.nature.com/articles/nature14236)

3. van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep Reinforcement Learning with Double Q-learning.* AAAI 2016. [arXiv:1509.06461](https://arxiv.org/abs/1509.06461)

4. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). *Prioritized Experience Replay.* ICLR 2016. [arXiv:1511.05952](https://arxiv.org/abs/1511.05952)

5. Wang, Z., Schaul, T., Hessel, M., van Hasselt, H., Lanctot, M., & de Freitas, N. (2016). *Dueling Network Architectures for Deep Reinforcement Learning.* ICML 2016. [arXiv:1511.06581](https://arxiv.org/abs/1511.06581)

6. Bellemare, M. G., Dabney, W., & Munos, R. (2017). *A Distributional Perspective on Reinforcement Learning.* ICML 2017. [arXiv:1707.06887](https://arxiv.org/abs/1707.06887)

7. Fortunato, M., et al. (2018). *Noisy Networks for Exploration.* ICLR 2018. [arXiv:1706.10295](https://arxiv.org/abs/1706.10295)

8. Badia, A. P., Piot, B., Kapturowski, S., Sprechmann, P., Vitvitskyi, A., Guo, D., & Blundell, C. (2020). *Agent57: Outperforming the Atari Human Benchmark.* ICML 2020. [arXiv:2003.13350](https://arxiv.org/abs/2003.13350)

**Citation count:** about 2,700 for the AAAI record ([OpenAlex](https://openalex.org/W2761873684), 2,703 when checked on 17 November 2026).
