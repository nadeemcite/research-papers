---
title: "RAG: Talk & Q&A"
parent: "098 RAG"
nav_order: 2
---
# TALK — RAG (Retrieval-Augmented Generation): Context, Discussion, and Q&A

**Paper:** [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (arXiv:2005.11401)](https://arxiv.org/abs/2005.11401)

## Verifiable Press/Blog Coverage

1. [Hugging Face Transformers documentation: RAG](https://huggingface.co/docs/transformers/model_doc/rag) — the official model page for the RAG models released in the Transformers library, with the paper's abstract and usage notes.

2. [Hugging Face blog: "Retrieval Augmented Generation with Huggingface Transformers and Ray"](https://huggingface.co/blog/ray-rag) (February 2021) — explains how RAG fine-tuning was scaled with Ray for the retrieval step.

## Q&A (answers from the paper)

**Q: What was the key insight that led to RAG?**
A (from the paper, Section 1): Large pre-trained models store a lot of knowledge in their weights, but they cannot easily expand or revise that memory, cannot show where an answer came from, and may "hallucinate". Hybrid models with a retrieval-based memory can fix some of this. RAG brings this hybrid memory to general seq2seq models and fine-tunes the retriever and generator together, with no direct supervision on which document to retrieve.

**Q: Why is the document index kept fixed during training?**
A (from the paper, Section 2.4): Updating the document encoder would mean rebuilding the index from time to time, as REALM does during pre-training, which is costly. The authors did not find this necessary for strong results, so they keep the document encoder and the index fixed and fine-tune only the query encoder and the BART generator.

**Q: What's the difference between RAG-Sequence and RAG-Token, and which is better?**
A (from the paper, Sections 2.1, 4.1 and 4.3): RAG-Sequence uses the same document for the whole answer; RAG-Token can use a different document for each token. Results are mixed: RAG-Sequence is a little better on Natural Questions (44.5 vs 44.1), TriviaQA and CuratedTrec, while RAG-Token is better on WebQuestions (45.5 vs 45.2) and on Jeopardy question generation. The authors think RAG-Token does well on Jeopardy because those questions often combine facts from several documents.

**Q: How does RAG compare to later approaches like RETRO or Atlas?**
A (general background, not from the paper): RETRO (DeepMind, 2021) retrieves chunks from a database of trillions of tokens and uses them through cross-attention. Atlas (Meta, 2022) focuses on few-shot learning with retrieval. They change how retrieval is scaled and trained, but keep the same basic "retrieve, then generate" idea.

**Q: Can RAG really update its knowledge without retraining?**
A (from the paper, Section 4.5): Yes, in their test. With an index built from a December 2016 Wikipedia dump, RAG answered 70% of questions about 2016 world leaders correctly; with the 2018 index, it answered 68% for 2018 leaders. With the wrong index, accuracy fell to 12% and 4%. So swapping the index changed the model's knowledge.

## Common Misconceptions

1. **"RAG requires fine-tuning."** — While the original paper fine-tunes the retriever and generator end-to-end, modern RAG systems often use off-the-shelf retrievers and generators without any fine-tuning. The paper's inference-time architecture (retrieve → generate) works with frozen models, which is how most production RAG systems operate today.

2. **"RAG and fine-tuning are alternatives."** — They're complementary. RAG provides updatable external knowledge; fine-tuning improves the model's parametric capabilities. Many production systems use both: fine-tune the model for task performance, then add RAG for factual grounding and knowledge updates.

3. **"RAG eliminates hallucination."** — RAG significantly reduces hallucination by grounding generation in retrieved evidence, but it doesn't eliminate it entirely. The generator can still produce text that doesn't match the retrieved passages, especially if the retriever fetches irrelevant documents.

4. **"RAG-Token and RAG-Sequence are very different."** — They differ only in where the sum over retrieved documents happens: once per sequence or at every token. Their scores are close on most tasks in the paper (for example 44.5 vs 44.1 on Natural Questions).

5. **"The retriever must be neural."** — The paper uses DPR (a neural retriever), but it also tested a word-overlap BM25 retriever (Section 4.5, Table 6). BM25 was best on FEVER, while the learned dense retriever was better on all other tasks, especially open-domain QA.

## Real Citations

- Lewis, P., Perez, E., Piktus, A., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *Advances in Neural Information Processing Systems (NeurIPS 2020)*, 33, 9459–9474.
- Karpukhin, V., Oğuz, B., Min, S., et al. (2020). "Dense Passage Retrieval for Open-Domain Question Answering." *EMNLP 2020*. — The DPR retriever used in RAG.
- Guu, K., Lee, K., Tung, Z., Pasupat, P., Chang, M. (2020). "REALM: Retrieval-Augmented Language Model Pre-Training." *NeurIPS 2020*. — Concurrent work on retrieval-augmented pre-training.
- Borgeaud, S., Mensch, A., Hoffmann, J., et al. (2022). "Improving Language Models by Retrieving from Trillions of Tokens." *NeurIPS 2022*. — RETRO, scaling RAG to trillions of tokens.
- Izacard, G., Lewis, P., Lomeli, M., et al. (2022). "Atlas: Few-shot Learning with Retrieval-Augmented Language Models." *arXiv:2208.03299*. — RAG for few-shot learning.
- Petroni, F., Piktus, A., Fan, A., et al. (2019). "Language Models as Knowledge Bases?" *EMNLP 2019*. — Foundational work showing LLMs store factual knowledge, motivating RAG.
