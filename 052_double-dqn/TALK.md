---
title: "Double DQN: Talk & Q&A"
parent: "052 Double DQN"
nav_order: 2
---
# TALK.md — Deep Reinforcement Learning with Double Q-learning

## Press & Blog Coverage

1. [**Key Papers in Deep RL**](https://spinningup.openai.com/en/latest/spinningup/keypapers.html), OpenAI Spinning Up. Lists this paper among the key deep RL papers, under the algorithm name Double DQN.

2. [**Rainbow: Combining Improvements in Deep Reinforcement Learning**](https://arxiv.org/abs/1710.02298), Hessel et al. A follow-up paper that combines Double DQN with other DQN improvements in one agent.

## Q&A (answers from the paper)

**Q1: What questions does the paper set out to answer?**
A: Q-learning was known to overestimate action values under certain conditions. The paper's Abstract says it was not known whether such overestimations are common in practice, whether they harm performance, and whether they can generally be prevented, and that the paper answers all three affirmatively.

**Q2: Why does taking a max lead to overestimation?**
A: Theorem 1 considers a state where all true action values are equal and the estimates are unbiased on average but not all correct. It shows the max over the estimates is then above the true value by at least a positive amount, so estimation errors from any source can push values up (Section "Overoptimism due to estimation errors").

**Q3: What exactly does Double DQN change?**
A: It evaluates the greedy policy chosen by the online network using the target network's values. The authors call it "perhaps the minimal possible change to DQN towards Double Q-learning", with no extra networks (Section "Double DQN").

**Q4: Did DQN really overestimate on Atari?**
A: Yes, according to Figure 3: DQN was consistently and sometimes vastly overoptimistic about the value of its greedy policy, and overestimations were observed in all 49 tested games, in varying amounts (Section "Results on overoptimism").

**Q5: Did Double DQN score better?**
A: On the 49 games, the median normalized score rose from 93.5% to 114.7% (Table 1, no-op starts) and from 47.5% to 88.4% with human starts (Table 2), where a tuned Double DQN reached 116.7%. The authors note the hyperparameters were tuned for DQN, not Double DQN (Section "Quality of the learned policies").

## Common Misconceptions

1. **"Overestimation always ruins the policy."** The paper says overoptimism does not always hurt: DQN achieves optimal behavior in Pong despite slightly overestimating the policy value (Section "Quality of the learned policies").

2. **"Double DQN needs a second network to train."** It reuses DQN's existing target network for evaluation, without additional networks or parameters (Discussion).

3. **"Double DQN wins on every task."** The paper reports clear gains in its summary statistics over 49 Atari games, not a win on every task. In [this repo's notebook](solution.ipynb), on a short CartPole run, Double DQN did not beat DQN.

## Real Citations

1. van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep Reinforcement Learning with Double Q-learning.* AAAI 2016. [arXiv:1509.06461](https://arxiv.org/abs/1509.06461)

2. van Hasselt, H. (2010). *Double Q-learning.* NIPS 2010, now NeurIPS. [Proceedings page](https://proceedings.neurips.cc/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html)

3. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning.* Nature, 518, 529–533. [Nature article](https://www.nature.com/articles/nature14236)

4. Mnih, V., et al. (2013). *Playing Atari with Deep Reinforcement Learning.* NIPS Deep Learning Workshop. [arXiv:1312.5602](https://arxiv.org/abs/1312.5602)

5. Hessel, M., et al. (2018). *Rainbow: Combining Improvements in Deep Reinforcement Learning.* AAAI 2018. [arXiv:1710.02298](https://arxiv.org/abs/1710.02298)

6. Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). *Prioritized Experience Replay.* ICLR 2016. [arXiv:1511.05952](https://arxiv.org/abs/1511.05952)

7. Wang, Z., et al. (2016). *Dueling Network Architectures for Deep Reinforcement Learning.* ICML 2016. [arXiv:1511.06581](https://arxiv.org/abs/1511.06581)

8. Fujimoto, S., van Hoof, H., & Meger, D. (2018). *Addressing Function Approximation Error in Actor-Critic Methods.* ICML 2018. [arXiv:1802.09477](https://arxiv.org/abs/1802.09477)

**Citation count:** about 3,900 ([OpenAlex](https://openalex.org/W2155968351), 3,880 when checked on 13 November 2026).
