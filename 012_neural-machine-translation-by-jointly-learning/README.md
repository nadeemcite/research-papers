# Neural Machine Translation by Jointly Learning to Align and Translate

**Paper:** Neural Machine Translation by Jointly Learning to Align and Translate  
**Authors:** Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio  
**Venue:** International Conference on Learning Representations (ICLR), 2015 (oral)  
**arXiv:** https://arxiv.org/abs/1409.0473

## One-paragraph summary

This paper introduced **additive (Bahdanau) attention** to neural machine translation. The previous encoder–decoder architecture (Sutskever et al., 2014; Cho et al., 2014) compressed an entire source sentence into a single fixed-length vector from which the decoder generated a translation—a bottleneck that caused performance to collapse on long sentences. Bahdanau, Cho and Bengio replaced this fixed vector with a **per-target-word context vector** computed as a weighted sum over all encoder hidden states. The weights (attention scores) are produced by a small feedforward alignment network that scores how well each source position matches the decoder's current state, then softmax-normalized. The encoder is a **bidirectional RNN** so each annotation captures both left and right context. On WMT'14 English→French, the model (RNNsearch-50) scored 26.75 BLEU, close to the phrase-based Moses system (33.30), and—critically—showed no degradation on long sentences, where the plain encoder–decoder fell apart. Qualitative analysis revealed that the learned soft alignments match human intuition, correctly handling word reordering between English and French.

## Core idea

Instead of forcing the encoder to squash a whole sentence into one vector, let the decoder **dynamically look back** at all encoder hidden states at every generation step. A learned alignment model scores each source position against the decoder's previous hidden state; the resulting softmax weights form a context vector that is a weighted average of source annotations. This "soft search" is fully differentiable, so the alignment model and the translation model are trained jointly end-to-end.

## Key method details relevant to the code

- **Bidirectional RNN encoder:** forward and backward GRUs read the source sequence; their hidden states are concatenated to form annotations `h_j = [h→_j ; h←_j]`, each capturing full-sentence context focused on position j.
- **Attention (alignment) model:** `e_ij = v^T tanh(W_s s_{i-1} + W_h h_j)` — a small feedforward network scoring decoder state `s_{i-1}` against encoder annotation `h_j`.
- **Attention weights:** `α_ij = softmax_j(e_ij)` — normalized over all source positions for each target step.
- **Context vector:** `c_i = Σ_j α_ij h_j` — weighted sum of annotations.
- **Decoder:** `s_i = f(s_{i-1}, y_{i-1}, c_i)`, then `p(y_i | …) = g(y_{i-1}, s_i, c_i)` — conditioned on a *different* context vector at each step.
- **Training:** SGD with Adadelta, minibatches of 80 sentences, ~5 days on a GPU. Beam search at inference.

## Influence

With over 30,000 citations (Semantic Scholar, 2026), this is one of the most influential papers in NLP history. The attention mechanism it introduced became the foundation of the Transformer (Vaswani et al., 2017), which replaced recurrence entirely but kept the attention paradigm. Every modern LLM—GPT, BERT, T5, Llama—traces its core mechanism to this paper. The bidirectional encoder + attention decoder template directly inspired Google's GNMT (2016), and the soft-alignment visualization became a standard interpretability tool.

## What problem does it solve

Imagine you're translating a long English paragraph into French for a friend. The old way was like asking someone to read the entire English paragraph, memorize it as one "summary blob," and then write the French translation from memory alone. If the paragraph is short, the summary blob works fine. But if it's very long, the person forgets details from the beginning by the time they reach the end—they can't hold it all in their head at once. This paper's solution is like giving the translator the original English paragraph to look at *while* they write each French word. Before writing each word, they glance back at the English text and focus on the part that matters most right now. Writing "zone"? Look at "Area." Writing "économique"? Look at "Economic." They don't have to memorize everything—they just need to know *where to look* at each step. This simple "look back and focus" trick let the model handle long sentences perfectly and became the basis for how all modern AI models pay attention to the right information.
