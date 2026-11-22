---
title: "LayerNorm: Talk & Q&A"
parent: "042 LayerNorm"
nav_order: 2
---
# TALK.md — Layer Normalization

## Press & Blog Coverage

1. [**`torch.nn.LayerNorm`**](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html), PyTorch documentation. The built-in layer that "applies Layer Normalization over a mini-batch of inputs", with the formula and a comparison to batch and instance normalization.

2. [**"The Illustrated Transformer"**](https://jalammar.github.io/illustrated-transformer/), Jay Alammar's blog. Explains that each sub-layer in a Transformer encoder has a residual connection and is followed by a layer-normalization step, linking to this paper.

3. [**"Attention Is All You Need"**](https://arxiv.org/abs/1706.03762), Vaswani et al., 2017. The Transformer paper, which uses layer normalization after each sub-layer and cites this paper.

## Q&A (answers from the paper)

**Q1: Why was layer normalization needed when batch normalization already existed?**
A: Batch normalization needs running averages of the summed-input statistics. In RNNs it seems to need separate statistics for each time step, and it cannot be applied to online learning or to very large distributed models with small mini-batches (Section 1). Layer normalization computes its statistics from the summed inputs within one layer for one training case, so it adds no dependencies between training cases and works with batch size 1 (Sections 1 and 3).

**Q2: How is it applied inside an RNN?**
A: At every time step the summed inputs a_t = W_hh h_{t−1} + W_xh x_t are re-centred and re-scaled by their own mean and standard deviation, then a shared gain and bias are applied before the non-linearity. Because the statistics depend only on the current time step, there is no problem when a test sequence is longer than any training sequence (Section 3.1).

**Q3: What is the key mathematical difference from batch normalization?**
A: Batch normalization computes the mean and variance of each hidden unit over the training cases in a mini-batch. Layer normalization computes them over all H hidden units of a layer for a single training case, so all units in a layer share the same μ and σ, but different training cases have different ones (Section 3).

**Q4: What does layer normalization not do well?**
A: It is not invariant to re-scaling a single neuron's weight vector, unlike batch and weight normalization; it is invariant to scaling the whole weight matrix and to shifting all incoming weights (Section 5.1, Table 1). In preliminary experiments with convolutional networks, batch normalization outperformed it. The authors' explanation is that hidden units whose receptive fields lie near the image boundary are rarely turned on, so they have very different statistics from the rest of the layer (Section 6.7).

**Q5: Why does it help RNN training?**
A: In a standard RNN the average magnitude of the summed inputs tends to grow or shrink at every time step, leading to exploding or vanishing gradients. Layer normalization makes the recurrent layer invariant to re-scaling all of its summed inputs, which gives much more stable hidden-to-hidden dynamics (Section 3.1).

## Common Misconceptions

1. **"LayerNorm and BatchNorm are interchangeable."** They normalize over different axes and have different invariance properties: layer normalization is invariant to re-scaling a single training case, batch normalization to re-centring the dataset (Section 5.1, Table 1).

2. **"LayerNorm always beats BatchNorm."** In the paper's preliminary ConvNet experiments, batch normalization did better, because units near the image boundary are rarely active and have very different statistics (Section 6.7). The authors found RNNs benefit the most, especially with long sequences and small mini-batches (Section 7).

3. **"LayerNorm needs a minimum batch size."** The opposite: it can be used in the pure online regime with batch size 1 (Section 3).

4. **"The paper introduced LayerNorm for Transformers."** The paper predates the Transformer and focuses on RNNs. The [Transformer](https://arxiv.org/abs/1706.03762) adopted it a year later.

5. **"A toy experiment shows LayerNorm is faster."** In [this repo's notebook](solution.ipynb), the toy task is so easy that no normalization, BatchNorm and LayerNorm all reach 100% test accuracy within 3 epochs; see its Summary.

## Real Citations

1. Ba, J. L., Kiros, J. R., & Hinton, G. E. (2016). *Layer Normalization.* [arXiv:1607.06450](https://arxiv.org/abs/1607.06450)

2. Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.* ICML 2015. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167)

3. Vaswani, A., et al. (2017). *Attention Is All You Need.* NIPS 2017 (now NeurIPS). [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

4. Salimans, T., & Kingma, D. P. (2016). *Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks.* NIPS 2016 (now NeurIPS). [arXiv:1602.07868](https://arxiv.org/abs/1602.07868)

5. Cooijmans, T., Ballas, N., Laurent, C., Gülçehre, Ç., & Courville, A. (2016). *Recurrent Batch Normalization.* [arXiv:1603.09025](https://arxiv.org/abs/1603.09025) (the recurrent batch normalization the paper compares against in Section 6.2)

**Citation count:** about 14,900 ([OpenAlex](https://openalex.org/W3037932933), 14,907 when checked on 22 November 2026).
