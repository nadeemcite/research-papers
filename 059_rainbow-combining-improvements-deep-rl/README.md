---
title: "059 Rainbow"
nav_order: 60
permalink: "/059_rainbow-combining-improvements-deep-rl/"
---
# Rainbow: Combining Improvements in Deep Reinforcement Learning
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/rainbow-combining-improvements-deep-rl)

- **Paper:** [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
- **Authors:** Matteo Hessel, Joseph Modayil, Hado van Hasselt, Tom Schaul, Georg Ostrovski, Will Dabney, Dan Horgan, Bilal Piot, Mohammad Azar, David Silver
- **Lab, year:** DeepMind, 2017 (AAAI 2018)
- **Video:** [Watch the YouTube Short](https://youtube.com/shorts/lpdiA9qWcVw) on [Backprop Daily](https://www.youtube.com/@BackpropDaily)

## Summary

After DQN learned to play Atari games, researchers proposed many separate fixes for its weak spots. Each fix was tested on its own, so nobody knew which ones work well together.

This paper puts six of them into one agent, called Rainbow: double Q-learning, prioritized replay, dueling networks, multi-step returns, distributional RL and noisy nets.

Rainbow set a new state of the art on 57 Atari games, in both learning speed and final score. The authors then removed one part at a time. Prioritized replay and multi-step returns turned out to matter most, while removing dueling or double Q-learning made little difference to the median score.

## What Problem Does It Solve

Imagine six cooks, each with a trick that makes soup taste better: one adds salt, one adds herbs, one cooks it slower. Each trick works on its own. But will the soup still be good if you use all six at once? Maybe two tricks fight each other.

DQN had the same question. Six improvements each beat plain DQN, but they had been tested mostly one at a time, and only a few pairs had been tried together.

Rainbow cooks the soup with all six tricks, checks that it tastes better, and then leaves out one trick at a time to see which ones really matter.

## Core Idea

Each of the six parts fixes a different weakness of DQN:

- **Double Q-learning:** DQN tends to overestimate values. The online network picks the next action and the target network scores it.
- **Prioritized replay:** replay the experiences the agent got most wrong more often, instead of picking them uniformly.
- **Dueling network:** split the network into a value stream V(s) and an advantage stream A(s, a), then combine them into Q-values.
- **Multi-step returns:** add up the rewards of the next n steps before bootstrapping, so rewards spread back faster.
- **Distributional RL:** predict a distribution over future returns (51 atoms) instead of only the average.
- **Noisy nets:** add learned noise to the network's weights, so the agent explores without ε-greedy.

## Key Method Details

- **One loss for everything:** Rainbow uses a multi-step distributional loss. The target distribution is shifted by the 3-step return; the online network picks the bootstrap action and the target network evaluates it (Section "The Integrated Agent").
- **Priority from the KL loss:** replay priorities come from the distributional KL loss rather than the TD error, with exponent ω = 0.5 and β annealed from 0.4 to 1 (Table 1).
- **Network:** a dueling network that outputs return distributions, with all linear layers made noisy.
- **Results:** Rainbow matched DQN's final performance after 7M frames and beat every baseline's final performance within 44M frames. Its final median human-normalized score was 223% with no-op starts and 153% with human starts (Figure 1, Table 2).
- **Ablations:** prioritized replay and multi-step learning were the two most crucial parts; removing either caused a large drop in median performance. Distributional RL came next, and removing noisy nets also hurt in aggregate. Removing dueling made no significant difference in aggregate, and removing double Q-learning made only a limited difference; both helped on some games and hurt on others (Section "Ablation studies", Figures 3 and 4).

### Simplifications in This Notebook

The notebook implements three of the six parts in PyTorch: Double DQN, dueling networks and proportional prioritized replay with a sum-tree. It trains five agents on CartPole-v1 with the same seed and settings: vanilla DQN, Double DQN, Dueling DQN, PER DQN and the three combined. Dueling DQN and PER DQN also use the Double DQN target. Each agent has a small MLP (two hidden layers of 128), Adam with learning rate 1e-3, ε-greedy exploration, and 250 episodes. A separate section trains Double DQN with 3-step returns. There is no distributional RL, no noisy nets, and priorities use the TD error with α = 0.6.

### What the Notebook Result Shows

The combined agent finished with the highest 50-episode average reward, **87.2**, against **39.4** for vanilla DQN, **44.1** for Double DQN, **51.7** for Dueling DQN and **60.0** for PER DQN. But it improved late: it was below most other agents between episodes 100 and 200, and only climbed in the last 30 or so episodes. 3-step Double DQN finished at **71.1**, well above 1-step Double DQN. No agent reached the 500-step limit.

This is one seed on a small task, so the gaps say little about which part is best. An earlier version of this notebook gave the PER agent the dueling network too, so "PER DQN" and "Rainbow" were the same agent with identical numbers; that bug is fixed here.

## Influence

OpenAlex lists about 2,700 citations for the AAAI version ([2,703 when checked on 17 November 2026](https://openalex.org/W2761873684)). OpenAI's Spinning Up lists it among its [key papers in deep RL](https://spinningup.openai.com/en/latest/spinningup/keypapers.html), and Google's [Dopamine](https://github.com/google/dopamine) research framework includes a Rainbow agent. Later DeepMind Atari work went further: [Agent57](https://arxiv.org/abs/2003.13350) (Badia et al., 2020) was the first deep RL agent to beat the human benchmark on all 57 Atari games.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/research-papers/blob/main/059_rainbow-combining-improvements-deep-rl/solution.ipynb) | From-scratch PyTorch implementation of Double DQN, dueling networks and prioritized replay, combined and compared on CartPole-v1 |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, Q&A (answers from the paper), citations, and common misconceptions about the paper |
