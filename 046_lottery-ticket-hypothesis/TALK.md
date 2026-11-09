---
title: "Lottery Ticket: Talk & Q&A"
parent: "046 Lottery Ticket"
nav_order: 2
---
# TALK.md — The Lottery Ticket Hypothesis

## Press & Blog Coverage

1. [**"Smarter training of neural networks"**](https://news.mit.edu/2019/smarter-training-neural-networks-0506), MIT News, 6 May 2019. MIT's own article on the paper. It says the project was named one of ICLR's two best papers, out of roughly 1,600 submissions.

2. [**"A new way to build tiny neural networks could create powerful AI on your phone"**](https://www.technologyreview.com/2019/05/10/135426/a-new-way-to-build-tiny-neural-networks-could-create-powerful-ai-on-your-phone/), MIT Technology Review, 10 May 2019. Explains the hypothesis with the bag-of-lottery-tickets image.

3. [**ICLR 2019 Awards**](https://iclr.cc/Conferences/2019/Awards). The conference's page listing the paper's Best Paper Award.

## Q&A (answers from the paper)

**Q1: What problem did the authors start from?**
A: The abstract says that pruning can cut a trained network's parameter count by over 90% without losing accuracy, but that the sparse architectures it produces are hard to train from the start. The paper asks why, and finds that the initial weights of the surviving connections matter (abstract, Section 1).

**Q2: Why "lottery ticket"?**
A: The abstract says the tickets found have won the initialization lottery, meaning their connections have initial weights that make training particularly effective. Section 1 states the hypothesis itself.

**Q3: How do the authors find a winning ticket?**
A: Train the network, prune p% of the weights, and reset the survivors to their original values. The authors focus on iterative pruning, which repeats this over several rounds and prunes a share of the weights that survived the previous round (Section 1).

**Q4: What happens if a winning ticket is randomly reinitialized?**
A: For Lenet on MNIST, reinitialized networks learn increasingly slower than the original and lose test accuracy after little pruning. With 21% of the weights left, the winning ticket reached minimum validation loss 2.51x faster than when reinitialized and was half a percentage point more accurate (Section 2).

**Q5: Does the method work the same way on deeper networks?**
A: Not without changes. On VGG-19 and Resnet-18 the authors report that finding winning tickets is sensitive to the learning rate and requires warmup at higher rates (Sections 1 and 4).

## Common Misconceptions

1. **"The lottery ticket is the pruned architecture."** The ticket is the mask together with the original initial weights. In the paper's experiments the same structure, randomly reinitialized, learns more slowly and reaches lower accuracy once pruning is heavy (Section 2).

2. **"This means you can skip training the big network."** The paper's method finds tickets by training the full network first. The authors list finding tickets more efficiently as future work (Section 6).

3. **"Tickets always beat random re-initialization."** The gap depends on how heavily the network is pruned. In [this repo's notebook](solution.ipynb), at 67.2% sparsity on a small MLP, the ticket learned faster early on but ended only 0.10 points ahead on a single seed.

## Real Citations

1. Frankle, J., & Carbin, M. (2019). *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.* ICLR 2019. [arXiv:1803.03635](https://arxiv.org/abs/1803.03635)

2. Liu, Z., Sun, M., Zhou, T., Huang, G., & Darrell, T. (2019). *Rethinking the Value of Network Pruning.* ICLR 2019. [arXiv:1810.05270](https://arxiv.org/abs/1810.05270)

3. Frankle, J., Dziugaite, G. K., Roy, D. M., & Carbin, M. (2019). *Stabilizing the Lottery Ticket Hypothesis.* [arXiv:1903.01611](https://arxiv.org/abs/1903.01611)

4. Frankle, J., Dziugaite, G. K., Roy, D. M., & Carbin, M. (2019). *Linear Mode Connectivity and the Lottery Ticket Hypothesis.* [arXiv:1912.05671](https://arxiv.org/abs/1912.05671)

5. Malach, E., Yehudai, G., Shalev-Shwartz, S., & Shamir, O. (2020). *Proving the Lottery Ticket Hypothesis: Pruning is All You Need.* [arXiv:2002.00585](https://arxiv.org/abs/2002.00585)

**Citation count:** about 2,770 ([OpenAlex](https://openalex.org/W2805003733), checked November 2026).
