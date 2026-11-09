---
title: "Seq2Seq: Talk & Q&A"
parent: "011 Seq2Seq"
nav_order: 2
---
# Talk / Press / Coverage — Sequence to Sequence Learning with Neural Networks

## Press and blog coverage

- **Google Research blog:** [A Neural Network for Machine Translation, at Production Scale](https://research.google/blog/a-neural-network-for-machine-translation-at-production-scale/) — Quoc V. Le and Mike Schuster (27 Sep 2016) describe moving Google Translate from phrase-based translation to neural machine translation, and cite this paper for learning the input-sequence-to-output-sequence mapping directly.
- **Google Research blog:** [Introducing tf-seq2seq: An Open Source Sequence-to-Sequence Framework in TensorFlow](https://research.google/blog/introducing-tf-seq2seq-an-open-source-sequence-to-sequence-framework-in-tensorflow/) — Anna Goldie and Denny Britz (11 Apr 2017) release a seq2seq framework and list this paper among its references.
- **Google Research publications:** [Sequence to Sequence Learning with Neural Networks](https://research.google/pubs/sequence-to-sequence-learning-with-neural-networks/) — the paper's page on Google's research site.
- **Wikipedia:** [Seq2seq](https://en.wikipedia.org/wiki/Seq2seq) — names Sutskever et al. (2014) as one of the papers cited as an originator of seq2seq and links it to Google Neural Machine Translation in 2016.
- **PyTorch tutorial:** [NLP From Scratch: Translation with a Sequence to Sequence Network and Attention](https://docs.pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — still teaches the encoder/decoder recurrent pattern, with attention added.

## Q&A (answers from the paper)

**Q: What was the main limitation the paper tackled?**

A (from the paper, Section 1): Deep networks worked well when inputs and outputs could be encoded as vectors of fixed size, but they could not map sequences to sequences whose lengths are unknown in advance. Translation is exactly that kind of problem. The paper uses one LSTM to read the input into a fixed-size vector and a second LSTM to generate the output from it.

**Q: Why LSTMs instead of plain RNNs?**

A (from the paper, Sections 1 and 3.3): The LSTM is able to learn data with long-range temporal dependencies, and here there is a long gap between an input and its matching output. The authors report that they were unable to train a standard RNN on the non-reversed translation problem. They believe one should be easy to train on reversed sources, but say they did not verify that experimentally.

**Q: What is the reversing trick and why does it help?**

A (from the paper, Section 3.3): Each source sentence is fed in backwards while the target stays in normal order. The first few source words then sit close to the first few target words, which gives the optimizer short-range dependencies to start from. Test perplexity fell from 5.8 to 4.7 and BLEU rose from 25.9 to 30.6. The authors say they were surprised by how large the improvement was.

**Q: How close did it get to the best existing systems?**

A (from the paper, Sections 1 and 3.6): An ensemble of 5 deep LSTMs reached a BLEU score of 34.81, against 33.30 for the phrase-based baseline on the same data. Using the LSTMs to rescore the baseline's 1000-best lists reached 36.5, close to the best published result of 37.0 at the time.

**Q: Did the paper use attention?**

A (general background, not from the paper): No. The whole input is squeezed into one fixed-size vector. Bahdanau, Cho and Bengio added attention in work published at ICLR 2015, letting the decoder look back at every encoder state.

**Q: Is this the same as the Transformer?**

A (general background, not from the paper): No. Transformers (Vaswani et al., 2017) replace recurrence with self-attention. They keep the same encoder/decoder split and the same one-token-at-a-time generation, but what happens inside each block is different.

## Common misconceptions

- **"Seq2seq is just machine translation."** The paper's experiments are on translation, but the authors describe it as a general approach to sequence learning, and the template was later used for summarization, speech recognition and image captioning.
- **"The fixed vector makes the paper irrelevant."** The fixed vector was a real bottleneck, and attention later removed it. The paper's point was that one vector could already carry enough to beat a phrase-based baseline, which is what made the follow-up work worth doing.
- **"Reversing the source is an arbitrary hack."** The authors give a reason: it shortens the distance between related source and target words. Their Table 1 shows the effect, with a single reversed LSTM at 30.59 BLEU against 26.17 for a single forward LSTM.
- **"The notebook here shows reversal helps."** It does not. The notebook always reverses the source and has no forward-reading comparison.

## Citations

- Sutskever, I., Vinyals, O., & Le, Q. V. (2014). [Sequence to sequence learning with neural networks](https://arxiv.org/abs/1409.3215). *Advances in Neural Information Processing Systems*, 27.
- Cho, K., et al. (2014). [Learning phrase representations using RNN encoder-decoder for statistical machine translation](https://arxiv.org/abs/1406.1078). *EMNLP*.
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). [Neural machine translation by jointly learning to align and translate](https://arxiv.org/abs/1409.0473). *ICLR*.
- Wu, Y., et al. (2016). [Google's neural machine translation system: Bridging the gap between human and machine translation](https://arxiv.org/abs/1609.08144).
- Vaswani, A., et al. (2017). [Attention is all you need](https://arxiv.org/abs/1706.03762). *Advances in Neural Information Processing Systems*, 30.
- [OpenAlex record for the paper](https://openalex.org/works/W2130942839) — 18,335 citations when checked on 9 Nov 2026.
