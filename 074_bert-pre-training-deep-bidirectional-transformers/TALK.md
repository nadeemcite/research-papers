---
title: "BERT: Talk & Q&A"
parent: "074 BERT"
nav_order: 2
---
# TALK — 074 BERT

## Press & Blog Coverage

1. [**"Open Sourcing BERT: State-of-the-Art Pre-training for Natural Language Processing"**](https://research.google/blog/open-sourcing-bert-state-of-the-art-pre-training-for-natural-language-processing/), Google Research blog, 2 November 2018. Posted by paper co-authors Jacob Devlin and Ming-Wei Chang to announce the code and pre-trained models.

2. [**"Understanding searches better than ever before"**](https://blog.google/products/search/search-language-understanding-bert/), Google, 25 October 2019. Says BERT would help Search better understand one in 10 searches in the U.S. in English.

3. [**NAACL 2019 Best Paper Awards**](https://naacl2019.org/blog/best-papers/). The conference's page listing BERT as Best Long Paper.

4. [**"BERT 101"**](https://huggingface.co/blog/bert-101), Hugging Face blog. A plain-language explainer of the model.

## Q&A (answers from the paper)

**Q1: Why does BERT mask words instead of predicting the next word?**
A: Standard language models are unidirectional, and the authors argue this limits the architectures usable during pre-training (Section 1). Bidirectional conditioning would let each word "see itself", so BERT replaces 15% of input tokens and predicts them from both sides (Section 3.1).

**Q2: How does BERT differ from ELMo?**
A: ELMo concatenates separately trained left-to-right and right-to-left language models. The authors say that approach costs twice as much as one bidirectional model, and is strictly less powerful than a deep bidirectional model, which can use both left and right context at every layer (Section 5.1).

**Q3: How expensive is fine-tuning compared to pre-training?**
A: The authors call it relatively inexpensive. They report that all the paper's results can be replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU, starting from the same pre-trained model (Section 3.2).

**Q4: How accurate is next sentence prediction?**
A: A footnote in Section 3.1 says the final model reaches 97%-98% accuracy on that task.

**Q5: What happened when the authors removed each pre-training task?**
A: Table 5 and Section 5.1 show that removing next sentence prediction hurts performance significantly on QNLI, MNLI and SQuAD 1.1. Replacing the masked objective with a left-to-right model performs worse on all the tasks, with large drops on MRPC and SQuAD.

## Common Misconceptions

1. **"BERT is a text-generation model."** It is an encoder trained to fill in masked tokens and to judge sentence pairs. The paper evaluates it on classification and question answering (Section 4), not on generating text.

2. **"The [CLS] vector is a ready-made sentence embedding."** A footnote in Section 3.1 says this vector "is not a meaningful sentence representation without fine-tuning, since it was trained with NSP."

3. **"Next sentence prediction did nothing."** In the paper's own ablation (Table 5), removing it hurts QNLI, MNLI and SQuAD 1.1 significantly. Later work, [RoBERTa](https://arxiv.org/abs/1907.11692), dropped it with different training choices, so that is a separate result.

4. **"BERT needs no labels at all."** Pre-training uses unlabelled text, but the downstream tasks are fine-tuned on labelled data (Section 3.2).

## Real Citations

1. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL 2019. [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

2. Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). *Improving Language Understanding by Generative Pre-Training*.

3. Peters, M. E., et al. (2018). *Deep Contextualized Word Representations*. NAACL 2018. [arXiv:1802.05365](https://arxiv.org/abs/1802.05365)

4. Vaswani, A., et al. (2017). *Attention Is All You Need*. NIPS 2017, now NeurIPS. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

5. Liu, Y., et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach*. [arXiv:1907.11692](https://arxiv.org/abs/1907.11692)

**Citation count:** about 54,000 for the NAACL record ([OpenAlex](https://openalex.org/W2963341956), checked November 2026).
