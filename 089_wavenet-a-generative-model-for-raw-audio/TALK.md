---
title: "WaveNet: Talk & Q&A"
parent: "089 WaveNet"
nav_order: 2
---
# TALK — WaveNet: A Generative Model for Raw Audio

## Press & Blog Coverage

1. [**"WaveNet: A generative model for raw audio"**](https://deepmind.google/blog/wavenet-a-generative-model-for-raw-audio/), DeepMind blog. The announcement, with audio samples. It says WaveNet sounded more natural than the best existing text-to-speech systems, reducing the gap with human performance by over 50%.

2. [**"WaveNet launches in the Google Assistant"**](https://deepmind.google/blog/wavenet-launches-in-the-google-assistant/), DeepMind blog. The follow-up on putting an improved WaveNet into the Google Assistant. It says the new model generates audio 1,000 times faster than the original, taking 50 milliseconds per second of speech.

## Q&A (answers from the paper)

**Q1: What question does the paper ask?**
A: Whether the autoregressive approach that worked for images and text can generate wideband raw audio, a signal with at least 16,000 samples per second (Section 1).

**Q2: How does WaveNet see far back in time without huge filters?**
A: It uses dilated causal convolutions. A block with dilations 1, 2, 4, ..., 512 has a receptive field of 1,024 samples, and stacking blocks grows it further (Section 2.1).

**Q3: Why μ-law quantization?**
A: Raw 16-bit audio would need a softmax over 65,536 values per timestep. Applying μ-law companding first and quantizing to 256 values makes this tractable (Section 2.2).

**Q4: Which activation does it use?**
A: The gated activation unit from gated PixelCNN, a tanh filter multiplied by a sigmoid gate (Section 2.3).

**Q5: Can one model speak in many voices?**
A: Yes. Conditioned on a one-hot speaker identity, a single WaveNet modeled all 109 speakers of its multi-speaker dataset (Section 3.1).

## Common Misconceptions

1. **"WaveNet is an RNN."** It has no recurrent connections. The paper models the conditional distributions with a stack of convolutional layers (Section 2).

2. **"Dilated convolutions are strided convolutions."** A dilated convolution skips input values with a step but keeps the output the same length as the input, unlike a strided convolution (Section 2.1).

3. **"The original WaveNet was fast enough for products."** The faster version that shipped in the Google Assistant came later, as the [DeepMind blog post](https://deepmind.google/blog/wavenet-launches-in-the-google-assistant/) describes. See also [Parallel WaveNet](https://arxiv.org/abs/1711.10433).

4. **"A low training loss means the model can speak."** In [this repo's notebook](solution.ipynb) the loss gets close to zero, but the model only learns to continue one repeating synthetic tone. The paper's speech results need real recordings and conditioning.

## Real Citations

- van den Oord, A., et al. (2016). *WaveNet: A Generative Model for Raw Audio.* [arXiv:1609.03499](https://arxiv.org/abs/1609.03499)
- van den Oord, A., Kalchbrenner, N., & Kavukcuoglu, K. (2016). *Pixel Recurrent Neural Networks.* ICML 2016. [arXiv:1601.06759](https://arxiv.org/abs/1601.06759)
- van den Oord, A., et al. (2016). *Conditional Image Generation with PixelCNN Decoders.* NIPS 2016, now NeurIPS. [arXiv:1606.05328](https://arxiv.org/abs/1606.05328) (source of the gated activation unit)
- He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Deep Residual Learning for Image Recognition.* [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)
- van den Oord, A., et al. (2018). *Parallel WaveNet: Fast High-Fidelity Speech Synthesis.* ICML 2018. [arXiv:1711.10433](https://arxiv.org/abs/1711.10433)
- Prenger, R., Valle, R., & Catanzaro, B. (2019). *WaveGlow: A Flow-based Generative Network for Speech Synthesis.* [arXiv:1811.00002](https://arxiv.org/abs/1811.00002)

**Citation count:** about 8,150 ([OpenAlex](https://openalex.org/W2519091744), 8,151 when checked on 12 November 2026).
