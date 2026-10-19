---
title: "GPT-3: Code architecture"
parent: "080 GPT-3"
nav_order: 1
---
# Code Architecture — GPT-3: Language Models are Few-Shot Learners

## Overview

This notebook demonstrates the **in-context learning** paradigm introduced by GPT-3, using a small pretrained language model (GPT-2) on a toy sentiment classification task. We replicate the key experimental finding: accuracy improves as the number of in-context examples (K) increases.

## Section-by-Section Breakdown

### 1. Setup & Imports
- Install `transformers` and `torch` (pip-installable).
- Import standard libraries: `torch`, `transformers`, `numpy`, `matplotlib`.

### 2. Load Pretrained Model
- Load GPT-2 (small, 124M parameters) via `GPT2LMHeadModel.from_pretrained('gpt2')`.
- This is our "language model" — a stand-in for the much larger GPT-3.
- The model is frozen (`model.eval()`); no gradient updates occur.

### 3. Toy Dataset Construction
- Build a small sentiment classification dataset with positive and negative sentences.
- Split into "demonstration pool" (for in-context examples) and "evaluation set" (for testing).
- Each sample: `(text, label)` where label ∈ {positive, negative}.

### 4. Prompt Construction
- **Zero-shot:** `TASK_DESCRIPTION`, a blank line, then `Sentence: {text}` and `Sentiment:`, where `TASK_DESCRIPTION = "Classify the sentiment of the sentence as positive or negative."`
- **Few-shot:** Prepend K labeled examples:
  ```
  Classify the sentiment of the sentence as positive or negative.

  Sentence: {text_1}
  Sentiment: {label_1}

  Sentence: {text_2}
  Sentiment: {label_2}

  ...
  Sentence: {query_text}
  Sentiment:
  ```
- The K examples are drawn at random from the 20-sentence demonstration pool, anew for every query.
- The model is expected to generate "positive" or "negative" as the next token.

### 5. Inference Function
- For each query, construct the prompt, tokenize, and run a forward pass.
- Extract the logits for the next token and compare probabilities of "positive" vs "negative" tokens.
- Predict the label with higher probability.
- This simulates how GPT-3 performs few-shot inference — purely via next-token prediction.

### 6. Few-Shot Evaluation Sweep
- Evaluate at K = 0, 1, 2, 4, 8, 16 in-context examples.
- For each K, run 3 trials over the full 30-sentence test set (each query gets its own random K examples) and average the accuracy.
- Report the mean and per-trial accuracy for each K. The sweep itself is a plain loop over `k_values` in the cell.

### 7. Visualization
- Plot accuracy vs. number of in-context examples (K).
- This mirrors, at toy scale, the paper's in-context learning curves (accuracy vs number of examples, e.g. Figure 1.2).

## Key Functions/Classes

| Function | Purpose |
|---|---|
| `construct_prompt(examples, query_text, k)` | Builds the text prompt with k random examples + query |
| `get_token_id(token_text)` | Token id of " positive" / " negative" |
| `predict_sentiment(model, tokenizer, prompt, device)` | Runs forward pass, compares the two label-token probabilities |
| `evaluate_few_shot(model, tokenizer, demo_pool, eval_set, k, device, n_trials=3)` | Mean accuracy (and per-trial accuracies) at a given k |

## Data Flow / Shapes

```
Input text (str) → Prompt construction (str) → Tokenization (tensor[1, seq_len])
  → GPT-2 forward (logits[1, seq_len, vocab_size]) → Extract last token logits
  → Compare P("positive") vs P("negative") → Predicted label (str)
```

## Deliberate Simplifications vs Full Paper

1. **Model size:** We use GPT-2 (124M) instead of GPT-3 (175B). The in-context learning effect is weaker at small scale but still observable.
2. **Single task:** We evaluate only sentiment classification. The paper evaluates over two dozen NLP datasets plus several new tasks.
3. **Token-level comparison:** We compare probabilities of "positive"/"negative" tokens rather than free generation. This is a cleaner evaluation for small models.
4. **Small dataset:** We use 50 hand-written sentences (20 for examples, 30 for testing) rather than standard benchmarks (e.g., SST-2).
5. **No few-shot fine-tuning comparison:** The paper compares against fine-tuned models; we focus solely on the in-context scaling effect.
6. **Single model size:** The paper sweeps across model sizes (125M to 175B). We use a single model and sweep over K only.
