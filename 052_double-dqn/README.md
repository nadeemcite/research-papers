---
title: "052 Double DQN"
nav_order: 53
permalink: "/052_double-dqn/"
---
# Deep Reinforcement Learning with Double Q-learning
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/double-dqn-deep-reinforcement-learning)

- **Paper:** [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- **Authors:** Hado van Hasselt, Arthur Guez, David Silver
- **Lab, year:** Google DeepMind, 2015 (AAAI 2016)
- **Video:** YouTube Short coming on 14 Nov 2026 on [Backprop Daily](https://www.youtube.com/@BackpropDaily)

## Summary

DQN is a reinforcement-learning agent: it learns, by trial and error, a value Q(s, a) for each action in each situation, and picks the action with the highest value. Q-learning was already known to overestimate these values under certain conditions.

This paper shows that DQN does overestimate in practice, often by a lot, on Atari games, and that it can hurt the agent. The cause is the max in DQN's learning target: the same estimates are used both to pick the best next action and to score it.

The fix, **Double DQN**, uses the online network to pick the action and the slower-changing target network to score it. It is a small change to DQN with no extra network. Across 49 Atari games, the median normalized score rose from 93.5% to 114.7%.

## What Problem Does It Solve

Imagine choosing a dish at a buffet with a friend who is a bit too optimistic about food. You ask them which dish looks best, and then how good that dish is. They point to the one they are most optimistic about, and then praise it with the same optimism, so you end up disappointed.

DQN does the same thing. It uses one set of value estimates both to choose the next action and to judge it. Random errors in those estimates are more likely to be picked when they are too high, so the targets drift upward.

Double DQN asks a second "friend" for the score: the target network, an older copy of the network. Its errors are different, so one network's optimism is not counted twice.

## Core Idea

- **DQN target:** `y = r + γ · max_a' Q(s', a'; θ⁻)`, where the target network θ⁻ both selects and evaluates the next action.
- **Why it overestimates:** even if the value estimates are right on average, taking the max over noisy estimates gives a value that is too high. The paper's Theorem 1 gives a lower bound on this error.
- **Double DQN target:** `y = r + γ · Q(s', argmax_a' Q(s', a'; θ); θ⁻)`. The online network θ selects the action, and the target network θ⁻ evaluates it.

## Key Method Details

- **Minimal change:** the target network was already part of DQN, so Double DQN needs no new network or parameters. The authors call it "perhaps the minimal possible change to DQN towards Double Q-learning".
- **Evidence of overestimation:** DQN overestimated the value of its policy in all 49 tested Atari games, by varying amounts, and sometimes vastly (Figure 3).
- **Scores:** with DQN's own hyperparameters, the median normalized score went from 93.5% to 114.7% (Table 1). With human starts it went from 47.5% to 88.4%, and to 116.7% with a tuned Double DQN (Table 2).

### Simplifications in This Notebook

The notebook trains DQN and Double DQN on CartPole-v1 instead of Atari, with a small MLP (4 → 128 → 128 → 2), a 10,000-transition replay buffer, a target network copied every 10 episodes and 300 training episodes per agent. Both agents use the same seed and hyperparameters, so the only difference is the target.

### What the Notebook Result Shows

On this short run neither agent is clearly better. Over the last 30 training episodes Double DQN averaged **120.3** against DQN's **102.2**, but in the 10-episode greedy evaluation DQN scored **183.8** against **152.7**. An earlier run of the same notebook went the other way on training reward (60.2 vs 91.6), so the comparison is not stable.

The Q-value gap is small: DQN's mean Q-value ended **1.43** higher than Double DQN's (5.6%). Both agents **underestimate**: their predicted values were **58.9** (DQN) and **51.6** (Double DQN) below the discounted returns they actually got. Both curves were still rising at episode 300.

So the notebook shows how the Double DQN target differs from DQN's, but it does not reproduce the paper's overestimation result. That would need longer training and several seeds.

## Influence

OpenAlex lists about 3,900 citations (3,880 when checked on 13 November 2026). Double DQN is one of the six DQN improvements combined in [Rainbow](https://arxiv.org/abs/1710.02298), and the same concern about overestimated values carried over to actor-critic methods such as [TD3](https://arxiv.org/abs/1802.09477).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/research-papers/blob/main/052_double-dqn/solution.ipynb) | From-scratch PyTorch DQN and Double DQN trained side by side on CartPole-v1, with reward, Q-value and evaluation plots |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, Q&A (answers from the paper), citations, and common misconceptions about the paper |
