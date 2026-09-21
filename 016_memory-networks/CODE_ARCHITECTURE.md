# Code Architecture — Memory Networks Notebook

## Overview

The notebook implements a simplified Memory Network (MemNN) with an external memory matrix over short facts, trains it on a toy bAbI-style question-answering task, and visualises which memory slot the model attends to when answering each question.

## Section-by-Section Breakdown

### 1. Setup & Imports
- Installs and imports `torch`, `torch.nn`, `numpy`, `matplotlib`.
- Sets random seed for reproducibility.
- Detects CUDA (Kaggle provides Tesla P100/T4 GPUs).

### 2. bAbI-Style Task Data Generator
Generates simulated stories in the style of the paper's simulated world: characters move between rooms, pick up and drop objects, and questions are asked about locations.

**Key function:**
- `generate_babi_story()`: Creates a random story with 4 characters, 3 objects, 5 rooms. Characters perform actions (go to room, pick up object, drop object). Generates questions like "Where is the milk now?" and "Where is Joe?" with labeled supporting facts.
- Returns: list of (sentence, is_question, answer, supporting_facts) tuples.

**Data flow:**
```
Random seed → Generate story sentences (actions) → Generate questions with labels
→ Return (story_lines, questions, answers, supporting_fact_indices)
```

**Deliberate simplification vs paper:**
- The paper uses a fully automated grammar simulator with 4 characters, 3 objects, 5 rooms. We replicate this but with a simplified vocabulary.
- The paper's large-scale QA task (14M facts from ReVerb) is not included — we focus on the simulated world task only.

### 3. Vocabulary & Encoding
- `Vocabulary` class: Maps words to integer indices. Built from all stories in the training set.
- `encode_sentence(words, vocab)`: Converts a sentence (list of words) to a bag-of-words vector — a binary vector of length |W| indicating which words are present.
- Three separate embedding spaces are used for the scoring function (matching the paper's D = 3|W|): one for memory slots (Φ_y), one for input questions (Φ_x input), one for supporting memories (Φ_x memory).

**Shapes:**
- Bag-of-words vector: `(vocab_size,)` — binary
- Memory matrix: `(max_memory, vocab_size)` — each row is a sentence's BoW vector
- Embedding matrix U: `(embed_dim, 3 * vocab_size)`

### 4. Memory Network Model (`MemNN`)
The core model with four components (I, G, O, R) implemented as a PyTorch `nn.Module`.

**Key parameters:**
- `vocab_size`: Size of the vocabulary
- `embed_dim`: Embedding dimension (paper uses 100)
- `max_memory`: Maximum number of memory slots
- `k_hops`: Number of supporting memories to retrieve (paper uses 2)

**Component implementations:**
- **I (Input):** `encode_sentence()` — bag-of-words encoding of input text
- **G (Generalization):** `write_memory()` — stores each new sentence in the next available memory slot (simple slot writing per Eq. 1)
- **O (Output):** `retrieve_memories()` — scores all memory slots against the question using embedding model `s_O(x, m_i) = Φ_x(x)^T U_O^T U_O Φ_y(m_i)`, retrieves top-k via argmax. For k=2, the second retrieval conditions on [question, first_retrieved_memory].
- **R (Response):** `generate_response()` — ranks all candidate words using `s_R([x, m_o1, m_o2], w) = Φ_x([x, m_o1, m_o2])^T U_R^T U_R Φ_y(w)`, returns argmax word.

**Scoring function (Eq. 5):**
```python
def score(x_bow, y_bow, U, embed_dim):
    # x_bow: (batch, D), y_bow: (batch, D)
    ex = x_bow @ U.T  # (batch, embed_dim)
    ey = y_bow @ U.T  # (batch, embed_dim)
    return (ex * ey).sum(dim=1)  # (batch,) — dot product in embedding space
```

**Data flow for a single question:**
```
Question text → encode → BoW (D=3|W|)
                ↓
Memory slots (N × D) → score each slot → argmax → o1 (first supporting memory)
                ↓
[q_bow, o1_bow] → score remaining slots → argmax → o2 (second supporting memory)
                ↓
[q_bow, o1_bow, o2_bow] → score all words → argmax → response word
```

### 5. Training Loop
- **Loss:** Margin ranking loss (hinge loss) per the paper's Eqs. 6-8. For each question with true supporting memories m_o1, m_o2 and true answer r:
  - `L_O1 = Σ max(0, γ - s_O(x, m_o1) + s_O(x, f̄))` — push correct memory above negatives
  - `L_O2 = Σ max(0, γ - s_O([x, m_o1], m_o2) + s_O([x, m_o1], f̄′))`
  - `L_R = Σ max(0, γ - s_R([x, m_o1, m_o2], r) + s_R([x, m_o1, m_o2], r̄))`
  - Total loss = L_O1 + L_O2 + L_R
- **Optimizer:** SGD with learning rate 0.01 (per paper)
- **Margin γ:** 0.1 (per paper)
- Negative sampling: randomly sample a subset of non-supporting memories as negatives rather than computing over all.
- **Epochs:** 10 (per paper)

### 6. Evaluation & Attention Visualisation
- Tests the trained model on held-out stories.
- For each test question, shows: the story, the question, the predicted answer vs. true answer, and which memory slots were retrieved (the attention).
- Visualises memory attention scores as a bar chart showing the model's relevance score for each memory slot when answering a question.

**Key visualisation:**
```python
# For each test question, show attention over memory slots
scores = model.score_all_memories(question_bow, memory_matrix)
# Plot bar chart: x = memory slot index, y = relevance score
```

### 7. Comparison with Baseline (RNN/LSTM)
- Implements a simple LSTM baseline that processes the story word-by-word and predicts the answer.
- Compares accuracy between MemNN and LSTM on the same bAbI-style task.
- Replicates the paper's finding that MemNN with k=2 vastly outperforms LSTM on questions requiring multi-hop reasoning.

### 8. Summary
- Prints final accuracy table (MemNN vs LSTM)
- Shows example predictions with retrieved memory slots

## Key Classes/Functions Summary

| Component | Class/Function | Purpose |
|-----------|---------------|---------|
| Data | `generate_babi_story()` | Generate simulated world stories with questions |
| Encoding | `Vocabulary`, `encode_sentence()` | Word-to-index mapping, BoW encoding |
| Model | `MemNN(nn.Module)` | Full memory network with I, G, O, R components |
| Training | `train_memnn()` | Training loop with margin ranking loss |
| Evaluation | `evaluate_memnn()` | Test accuracy and attention extraction |
| Visualisation | `plot_memory_attention()` | Bar chart of memory slot relevance scores |
| Baseline | `BaselineLSTM(nn.Module)` | Simple LSTM baseline for comparison |

## Deliberate Simplifications vs Full Paper

1. **No time features (Section 3.4):** The paper adds temporal features to model write time. We use a simplified version without the time-aware triple scoring `s_Ot`.
2. **No word-sequence segmentation (Section 3.2):** We assume inputs are already segmented into sentences.
3. **No memory hashing (Section 3.3):** Our memory is small enough to score all slots directly.
4. **No unseen word modeling (Section 3.5):** We assume a fixed vocabulary.
5. **No bag-of-words exact match features (Section 3.6):** We use pure embedding-based scoring.
6. **No large-scale QA task (Section 5.1):** We only implement the simulated world task.
7. **Single-word answers only:** We use the ranking-based R component (Eq. 4), not the RNN-based response generation.
8. **Simplified vocabulary:** The paper's grammar generates diverse sentence structures; we use a simplified but faithful version covering the key reasoning patterns.
