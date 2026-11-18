---
title: "World Models: Talk & Q&A"
parent: "062 World Models"
nav_order: 2
---
# TALK.md — World Models

## Press & Blog Coverage

1. [**Key Papers in Deep RL**](https://spinningup.openai.com/en/latest/spinningup/keypapers.html), OpenAI Spinning Up. Lists the NeurIPS 2018 version, "Recurrent World Models Facilitate Policy Evolution" (Ha and Schmidhuber, 2018), in its model-based RL section.

2. [**"Recurrent World Models Facilitate Policy Evolution"**](https://papers.nips.cc/paper/2018/hash/2de5d16682c3c35007e4e92982f1a2ba-Abstract.html), NeurIPS 2018 proceedings. The conference version of this work, published as a separate paper.

## Q&A (answers from the paper)

**Q1: What are the three parts of the agent?**
A: A vision model V, a variational autoencoder that compresses each frame into a small latent vector z; a memory model M, an MDN-RNN that predicts the distribution of the next z; and a controller C, a simple linear map from [z_t, h_t] to the action (Section 2).

**Q2: Why is the controller so small?**
A: The authors deliberately put most of the model's complexity and parameters in V and M, and keep C tiny. That makes it possible to train C with evolution strategies, specifically CMA-ES, which works well for up to a few thousand parameters (Sections 2.3 and 2.4). The CarRacing controller has 867 parameters (Section 3.2).

**Q3: How well did it do on CarRacing?**
A: The full world model scored 906 ± 21 over 100 random trials, which the authors call the first known solution of the task. A controller that sees only z scored 632 ± 251; adding a hidden layer gave 788 ± 141 (Section 3.3, Table 1). Here C was trained in the real environment.

**Q4: Where did the agent actually train inside its dream?**
A: In VizDoom: Take Cover. M was extended to also predict whether the agent dies, which makes it a full environment; C was trained entirely inside it and then deployed in the real game (Sections 4.1–4.4). With temperature τ = 1.15 the transferred policy survived 1092 ± 556 time steps, against 210 ± 108 for a random policy (Table 2).

**Q5: What can go wrong when training inside the dream?**
A: The controller can find adversarial policies that exploit M's mistakes. In early experiments the agent learned to move so that the dream's monsters never fired, or so that fireballs disappeared. Raising the temperature τ makes the dream more uncertain and harder to exploit, but too much makes it too hard to learn anything (Section 4.5, Table 2).

**Q6: What if a random policy cannot explore the environment?**
A: The authors propose an iterative procedure: collect rollouts, train M and C, then collect new rollouts with the improved agent and repeat. For this, M would also predict rewards and actions (Section 5).

## Common Misconceptions

1. **"The CarRacing agent learned inside its dream."** Its controller was trained in the real environment using the world model's features. Training fully inside the dream and transferring back was the VizDoom experiment (Sections 3 and 4).

2. **"A controller that scores well in the dream will score well in reality."** Not necessarily. At τ = 0.1 the agent scored 2086 ± 140 in the dream but only 193 ± 58 in the real game, below a random policy (Table 2).

3. **"World Models was a NeurIPS paper."** The arXiv paper is separate from the NeurIPS 2018 paper "Recurrent World Models Facilitate Policy Evolution" by the same authors.

4. **"The approach only applies to CarRacing."** The paper covers two tasks, CarRacing-v0 and VizDoom: Take Cover (Sections 3 and 4).

5. **"A world model automatically gives a good controller."** In [this repo's notebook](solution.ipynb), trained on only 50 random rollouts, the controller's predicted return improves inside the dream, but in the real toy task it scores −133.5 against −101.6 for random actions, because V's latents barely encode where the agent and target are.

## Real Citations

1. Ha, D., & Schmidhuber, J. (2018). *World Models.* [arXiv:1803.10122](https://arxiv.org/abs/1803.10122)

2. Ha, D., & Schmidhuber, J. (2018). *Recurrent World Models Facilitate Policy Evolution.* NeurIPS 2018. [Proceedings](https://papers.nips.cc/paper/2018/hash/2de5d16682c3c35007e4e92982f1a2ba-Abstract.html)

3. Kingma, D. P., & Welling, M. (2014). *Auto-Encoding Variational Bayes.* ICLR 2014. [arXiv:1312.6114](https://arxiv.org/abs/1312.6114)

4. Schmidhuber, J. (2015). *On Learning to Think: Algorithmic Information Theory for Novel Combinations of Reinforcement Learning Controllers and Recurrent Neural World Models.* [arXiv:1511.09249](https://arxiv.org/abs/1511.09249)

5. Hafner, D., Lillicrap, T., Ba, J., & Norouzi, M. (2020). *Dream to Control: Learning Behaviors by Latent Imagination.* ICLR 2020. [arXiv:1912.01603](https://arxiv.org/abs/1912.01603)

6. Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). *Mastering Diverse Domains through World Models.* [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)

**Citation count:** about 2,300 for the arXiv paper ([OpenAlex](https://openalex.org/W2795843265), 2,315 when checked on 18 November 2026), and about 1,180 for the NeurIPS 2018 paper ([OpenAlex](https://openalex.org/W2890208753), 1,177 when checked on 18 November 2026).
