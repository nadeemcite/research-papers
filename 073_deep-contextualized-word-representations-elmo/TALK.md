---
title: "ELMo: Talk & Q&A"
parent: "073 ELMo"
nav_order: 2
---
# TALK.md — ELMo: Deep Contextualized Word Representations

## Press / Blog Coverage

1. **The Gradient:** [NLP's ImageNet moment has arrived](https://thegradient.pub/nlp-imagenet/) (8 Jul 2018). An essay that names ELMo, ULMFiT and the OpenAI transformer as the pretrained language models changing NLP, by analogy with ImageNet-pretrained models in computer vision.
2. **Jay Alammar:** [The Illustrated BERT, ELMo, and co.](https://jalammar.github.io/illustrated-bert/) An illustrated walkthrough of how ELMo builds a contextual embedding and how it led towards BERT.
3. **NAACL HLT 2018 Chairs Blog:** [Outstanding Papers](https://naacl2018.wordpress.com/2018/04/11/outstanding-papers/) (11 Apr 2018). Lists this paper among the conference's four outstanding papers.
4. **AllenNLP ELMo page:** [archived copy from Dec 2018](https://web.archive.org/web/20181230124950/https://allennlp.org/elmo). The project's own page for the model; the original URL no longer serves it.
5. **Authors' code:** [allenai/bilm-tf](https://github.com/allenai/bilm-tf), the TensorFlow implementation of the biLM.
6. **ACL Anthology:** [the published paper, N18-1202](https://aclanthology.org/N18-1202/).

## Q&A (answers from the paper)

**Q1: What was the core intuition behind ELMo: why not just use better static embeddings?**

A (from the paper, Sections 1 and 3): A word's meaning depends on its context, and a single vector per word type cannot show that. ELMo uses the internal states of a pretrained bidirectional language model, so each occurrence of a word gets its own vector. The layers carry different information, so each task learns its own weighted mix of them.

**Q2: How do the layers differ?**

A (from the paper, Section 5.3, Tables 5 and 6): The top layer is better at word sense disambiguation (F1 69.0 against 67.4 for the first layer). The first layer is better at part-of-speech tagging (accuracy 97.3 against 96.8).

**Q3: Is the biLM frozen when ELMo is added to a task?**

A (from the paper, Section 3.3): Yes. The authors freeze the biLM's weights and concatenate the ELMo vector with the task model's input. The layer weights and the scale γ are the parts trained with the task.

**Q4: How large were the gains?**

A (from the paper, Section 4, Table 1): The text reports relative error reductions of 6 to 20% over strong base models. Table 1 lists 5.8% for SNLI up to 24.9% for SQuAD, across six tasks.

**Q5: Does mixing all layers beat using only the top one?**

A (from the paper, Section 5.1, Table 2): Yes on the development sets of SQuAD, SNLI and SRL, where using all layers improved on using only the last layer. The strength of the weight regularization λ matters: a large λ such as 1 reduces the mix to a plain average of the layers.

## Common Misconceptions

1. **"ELMo is just another word embedding like Word2Vec."** — No. Word2Vec and GloVe produce a single static vector per word type. ELMo produces a *different* vector for each token occurrence based on its sentence context. The same word type in different sentences gets different ELMo vectors.

2. **"ELMo was the first contextual embedding."** — Not quite. Context2Vec (Melamud et al., 2016) and CoVe (McCann et al., 2017) proposed context-dependent representations earlier. ELMo's contribution was the multi-layer biLM architecture and the demonstration that learnable layer combinations give consistent improvements across many tasks.

3. **"ELMo requires the downstream model to be an LSTM."** — No. ELMo embeddings can be added to any model: CNNs, Transformers, attention-based architectures. The paper showed improvements across diverse architectures.

4. **"ELMo and BERT are the same thing."** — They share the idea of pre-trained contextual representations but differ architecturally (biLSTM vs. Transformer) and methodologically (feature extraction vs. fine-tuning). BERT also uses masked language modeling rather than standard LM objectives.

## Real Citations

1. Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K., & Zettlemoyer, L. (2018). Deep contextualized word representations. *NAACL 2018*. [arXiv:1802.05365](https://arxiv.org/abs/1802.05365)
2. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *arXiv:1810.04805*. (Cites ELMo as prior work on contextual representations.)
3. Howard, J., & Ruder, S. (2018). Universal Language Model Fine-tuning for Text Classification. *ACL 2018*. [arXiv:1801.06146](https://arxiv.org/abs/1801.06146)
4. Melamud, O., Goldstein, J., & Dagan, I. (2016). context2vec: Learning Generic Context Embeddings with Bidirectional LSTM. *Repl4NLP 2016*.
5. McCann, B., Bradbury, J., Xiong, C., & Socher, R. (2017). Learned in Translation: Contextualized Word Vectors. *NIPS 2017*.
6. Tenney, I., Das, D., & Pavlick, E. (2019). BERT Rediscovers the Classical NLP Pipeline. *ACL 2019*. (Probing analysis inspired by ELMo's layer-wise findings.)

**OpenAlex:** [13,289 citations](https://openalex.org/works/W2787560479) (checked 9 Nov 2026).
