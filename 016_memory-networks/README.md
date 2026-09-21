# Memory Networks

**Paper:** Weston, J., Chopra, S., & Bordes, A. (2014). Memory Networks. arXiv:1410.3916 [cs.AI].  
**Link:** https://arxiv.org/abs/1410.3916  
**Authors:** Jason Weston, Sumit Chopra, Antoine Bordes (Facebook AI Research)  
**Published:** 15 October 2014 (v1); 29 November 2015 (v11)

## Summary

Memory Networks (MemNNs) introduce a class of learning models that combine a neural inference engine with an explicit, readable and writable long-term memory. Unlike RNNs or LSTMs, which compress all past information into a fixed-size hidden state vector, MemNNs store each incoming fact as a separate slot in an external memory array. A MemNN has four components: **I** (input feature map) converts text to an internal representation; **G** (generalization) writes new facts into memory slots; **O** (output feature map) reads from memory by scoring the relevance of each slot to the current question using learned embedding vectors, retrieving the top-k supporting memories; and **R** (response) produces the final answer by ranking candidate words against the retrieved memories. The model is trained end-to-end with a margin ranking loss, using supervised labels that indicate which memory slots are the correct supporting facts.

The core scoring function is an embedding model: `s(x, y) = Φ_x(x)^T U^T U Φ_y(y)`, where `Φ` maps sentences to bag-of-words feature vectors and `U` is a learned embedding matrix. For multi-hop reasoning (k=2), the model first finds the most relevant memory given the question, then finds a second memory given the question and the first retrieved memory — effectively chaining facts to answer questions that require multiple supporting sentences. The paper evaluates MemNNs on a large-scale QA task (14M facts from ReVerb/ClueWeb) and a simulated world with 4 characters, 3 objects, and 5 rooms where answering questions like "Where is the milk now?" requires understanding verb semantics (picked up, left, dropped) and temporal ordering. The MemNN with k=2 hops and time features achieves near-perfect 100% accuracy on the simulated task, vastly outperforming RNNs (27.9%) and LSTMs (49.1%) on the harder settings.

## What Problem Does It Solve

Imagine you're reading a little story to a child: "Joe went to the kitchen. Joe picked up the milk. Joe went to the office. Joe left the milk. Joe went to the bathroom." Then you ask: "Where is the milk now?" To answer, you need to remember that Joe picked up the milk in the kitchen, then went to the office and left it there. A regular neural network tries to squish the entire story into one tiny brain-box (a hidden state vector) — and by the time you ask the question, the details have blurred together. It's like trying to remember a phone number someone told you five minutes ago without writing it down. Memory Networks solve this by giving the AI a notebook: every sentence gets written down as a separate note. When a question comes in, the AI flips through its notes, finds the most relevant ones, and pieces together the answer — just like you would look back at your notes to answer a question about a story.

## Influence

Memory Networks were one of the foundational papers in the "neural memory" line of research, alongside Neural Turing Machines (Graves et al., 2014, submitted to arXiv the same week). The explicit memory + attention retrieval paradigm directly inspired End-to-End Memory Networks (Sukhbaatar et al., 2015), which removed the need for supervised memory labels, and the key-value memory networks used in modern QA systems. The multi-hop reasoning pattern — iteratively retrieving and combining evidence — became a central idea in reasoning architectures and can be seen as a precursor to the attention mechanisms in Transformers. The bAbI QA dataset introduced alongside this work became a standard benchmark for reasoning in neural models.

## arXiv Link
https://arxiv.org/abs/1410.3916
