# Attention Is All You Need (Transformer)

## Summary

"Attention Is All You Need," published in 2017 by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin (all at Google Brain / Google Research), introduced the **Transformer** — a neural network architecture based entirely on attention mechanisms, dispensing with recurrence and convolutions entirely for sequence transduction tasks. The Transformer replaced the sequential processing bottleneck of RNNs/LSTMs with **self-attention**, where every position in a sequence can directly attend to every other position in parallel. The model achieved 28.4 BLEU on WMT 2014 English-to-German translation (improving over existing best results by over 2 BLEU) and 41.8 BLEU on English-to-French — both state-of-the-art at the time — while training in just 3.5 days on 8 GPUs, a fraction of the cost of competing models. This paper is arguably the most influential neural network architecture paper of the last decade: every major modern language model (GPT, BERT, T5, LLaMA, Claude, Gemini) and increasingly vision, audio, and multimodal models are built on the Transformer architecture it introduced.

**arXiv link:** https://arxiv.org/abs/1706.03762

## Core Idea

The central insight is that **self-attention** — computing relationships between all pairs of positions in a sequence simultaneously — is a more powerful and more parallelizable primitive than recurrent or convolutional processing. The paper's key contributions:

1. **Scaled Dot-Product Attention:** Given queries Q, keys K, and values V, attention is computed as softmax(QK^T / √d_k) · V. The scaling factor 1/√d_k prevents the dot products from growing large in magnitude, which would push the softmax into regions with extremely small gradients.

2. **Multi-Head Attention:** Instead of performing a single attention function with d_model-dimensional keys/values/queries, the model projects Q/K/V h times into different learned linear subspaces (each of dimension d_k = d_v = d_model / h), attends in each subspace in parallel, concatenates the results, and projects back. This allows the model to jointly attend to information from different representation subspaces at different positions.

3. **Positional Encoding:** Since self-attention is permutation-invariant (no inherent notion of word order), the model injects positional information using sine and cosine functions of different frequencies: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)), PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model)). This allows the model to easily learn to attend by relative positions.

4. **Encoder-Decoder Architecture:** The encoder consists of N=6 identical layers, each with a multi-head self-attention sub-layer and a position-wise feed-forward sub-layer (two linear transformations with a ReLU in between), with residual connections and layer normalization around each sub-layer. The decoder is similar but adds a third sub-layer performing multi-head attention over the encoder output (cross-attention), and uses masking in the self-attention to prevent positions from attending to future positions.

## Key Method Details

- **Model dimension:** d_model = 512, with h = 8 attention heads, d_k = d_v = 64 per head.
- **Feed-forward dimension:** d_ff = 2048 (inner layer of the position-wise FFN).
- **Number of layers:** N = 6 encoder layers and 6 decoder layers.
- **Residual connections:** Each sub-layer output is LayerNorm(x + Sublayer(x)). All sub-layers and embedding layers produce outputs of dimension d_model = 512 to enable residual connections.
- **Embedding sharing:** The input embeddings, output embeddings, and pre-softmax linear transformation share the same weight matrix, multiplied by √d_model.
- **Training:** Adam optimizer (β1=0.9, β2=0.98, ε=10^-9) with a learning rate that warms up over 4,000 steps then decays inversely with the square root of step number. Dropout 0.1, label smoothing ε=0.1.
- **Results:** 28.4 BLEU on WMT 2014 En-De (new SOTA, +2.0 over previous best ensemble), 41.8 BLEU on WMT 2014 En-Fr (new single-model SOTA). Training: 3.5 days on 8 × P100 GPUs.

## Influence

The Transformer is the foundational architecture of modern deep learning. It directly enabled BERT (Devlin et al., 2018), GPT series (Radford et al. 2018–2020), T5 (Raffel et al., 2019), and virtually every subsequent large language model. The architecture was adapted to vision (ViT, Dosovitskiy et al., 2020), audio (Whisper), protein structure (AlphaFold 2), and code (Codex). The paper has been cited over 170,000 times — one of the most cited computer science papers of all time. "Attention is all you need" became a widely recognized phrase in the AI community and a template for paper titles across the field.

## What Problem Does It Solve

Imagine you're reading a long book and need to answer a question about something mentioned in chapter 1, but you're currently in chapter 20. The old way AI handled this was like reading the book page by page — by the time it reached chapter 20, the details from chapter 1 had faded from its "memory" (this was called the vanishing gradient problem with RNNs). Even worse, it had to read every single page in order, one at a time, so it was very slow — even on powerful computers that are designed to do many things simultaneously.

This paper introduced a brilliant idea: instead of reading page by page, what if every page could "look at" every other page at the same time? So when the AI is reading chapter 20, it can directly "pay attention to" chapter 1 without having to remember everything in between. This is called **self-attention**. It's like being at a dinner party where instead of passing a message down the table one person at a time, everyone can talk to everyone else directly. This made AI both smarter (it could connect ideas across long distances) and much faster (computers could process all the "conversations" at the same time instead of one by one). This single idea became the foundation for ChatGPT, Google Translate, and almost every modern AI system.
