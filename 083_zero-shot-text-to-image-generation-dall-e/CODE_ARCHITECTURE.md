---
title: "DALL-E: Code architecture"
parent: "083 DALL-E"
nav_order: 1
---
# Code Architecture — DALL-E (Zero-Shot Text-to-Image Generation)

## Overview

The notebook builds a very small version of DALL-E's two-stage method in PyTorch. Stage 1 is a discrete VAE trained with the gumbel-softmax relaxation that turns a 32×32 image into an 8×8 grid of tokens from 128 codes. Stage 2 is a decoder-only transformer that models a 4-token caption followed by the 64 image tokens. The data are synthetic coloured shapes (6 colours × 3 shapes). Three colour-shape combinations are left out of all training, so their captions test zero-shot composition. A checker (a colour rule and a shape CNN) scores whether the generated images match their captions. It runs on a GPU if there is one.

## Section-by-Section Breakdown

### 1. Setup and Imports
- PyTorch, NumPy, PIL, matplotlib; nothing is installed
- Seeds 42; `CUBLAS_WORKSPACE_CONFIG` set before importing torch, cuDNN deterministic, `torch.use_deterministic_algorithms(True, warn_only=True)`
- Figures as JPEG at 80 dpi; on Kaggle an `atexit` hook copies the executed notebook to `executed_solution.ipynb`

### 2. Hyperparameters
- Image 32×32, grid 8×8, `N_CODES = 128`, so 64 image tokens per image
- dVAE: 60 epochs, Adam with learning rate 1e-3 (cosine decay), hidden width 256, `RECON_WEIGHT = 50` (MSE / 2σ² with σ = 0.1), τ from 1 to 1/16 over the first half of training, KL weight rising from 0 to 0.5 over the first 2% of updates
- Transformer: `D_MODEL = 256`, 4 layers, 8 heads, 40 epochs, AdamW with learning rate 3e-4 (one-cycle schedule), text and image loss weights 1/8 and 7/8
- Batch size 128; 400 training images per seen combination, 40 test images per combination; 32 generated images per caption

### 3. Synthetic Dataset with Held-Out Combinations
- `draw_shape(color, shape, rng)`: one circle, square or triangle (radius 6–10, centre jittered by up to 4 pixels) on a light grey background
- `caption(color, shape)`: "a red circle", "an orange square", …
- `HELD_OUT = [('red', 'square'), ('blue', 'triangle'), ('green', 'circle')]`; the other 15 combinations are `SEEN`
- `make_set`: 6,000 training images (15 × 400), 600 test images of seen combinations and 120 of held-out combinations; an assert checks that no held-out caption is in the training set

### 4. Stage 1: Discrete VAE with the Gumbel-Softmax Relaxation
- `DiscreteVAE`: encoder Conv 4×4 stride 2 (3→64) → Conv 4×4 stride 2 (64→256) → Conv 3×3 → 1×1 Conv to 128 logits per cell; decoder 1×1 Conv (128→256, the codebook lookup) → Conv 3×3 → two transposed convs back to 32×32 → sigmoid
- `forward(x, tau)`: `F.gumbel_softmax` over the 128 logits of each cell, decode the soft one-hot grid, and return the reconstruction and the KL to a uniform prior per cell
- `encode`: argmax of the logits (no gumbel noise); `decode`: one-hot tokens through the decoder
- Training loss `RECON_WEIGHT * MSE + kl_w * KL`; `hard_mse` measures test reconstruction error with hard argmax codes after every epoch
- Plots of the reconstruction error, KL and τ; test reconstruction error, codebook use and reconstructions of seen and held-out test images

### 5. A Checker for Colour and Shape
- `foreground`: pixels whose colour differs clearly from the background
- Colour: the mean foreground colour matched to the nearest of the 6 colours
- `shape_net`: a small CNN (two conv + pool blocks, two linear layers) that classifies the foreground mask as circle, square or triangle; trained for 5 epochs on the training images and their dVAE reconstructions (seen combinations only)
- `check_images`, `checker_accuracy`: accuracy on real test images and on reconstructions, seen and held-out

### 6. Stage 2: Transformer over Text and Image Tokens
- Word-level text: `[BOS]`, then "a"/"an", colour and shape (text vocabulary 12); image tokens offset by 12, so the joint vocabulary has 140 tokens and a sequence has 68
- `Block`: pre-LayerNorm, `nn.MultiheadAttention` with a causal mask, GELU MLP (4× width), dropout 0.1
- `TextImageTransformer`: token and learned position embeddings, 4 blocks, final LayerNorm and a linear head
- `seq_losses`: cross-entropy averaged separately over the 3 word targets and the 64 image targets; training loss `1/8 · text + 7/8 · image`, gradient clipping at 1; the image loss on the seen-combination test set is logged every epoch and plotted

### 7. Generating Images from Captions
- `generate(model, caps, gen)`: starts from the 4 text tokens and samples 64 image tokens one at a time at temperature 1, only from the image part of the vocabulary, then decodes them with the dVAE
- 32 images for each of the 18 captions; `show_samples` shows the first 8 for some seen captions

### 8. Zero-Shot Test
- Samples for the 3 held-out captions
- `score`: share of the 32 samples with the right colour, the right shape, and both, for every caption; means over seen and held-out captions, and the chance level 1/18
- For each held-out caption, the most common (colour, shape) the checker read

### 9. Summary
- `### What the run above shows` with the saved run's numbers, then what the real DALL-E has that this notebook does not

## Key Functions/Classes

| Name | Purpose |
|---|---|
| `draw_shape`, `caption`, `make_set` | Synthetic images and captions, with held-out combinations |
| `DiscreteVAE` | Stage 1: gumbel-softmax dVAE with `encode` (argmax) and `decode` |
| `hard_mse` | Reconstruction error with hard codes |
| `shape_net`, `check_images`, `checker_accuracy` | Colour rule and shape CNN for scoring images |
| `TextImageTransformer`, `Block` | Stage 2: decoder-only transformer over text and image tokens |
| `seq_losses` | Separate text and image cross-entropies |
| `generate` | Samples image tokens for captions and decodes them |
| `score` | Colour, shape and joint accuracy of the samples for one caption |

## Data Flow / Shapes

```
image (B, 3, 32, 32) → encoder → logits (B, 128, 8, 8)
  training: gumbel_softmax(τ) → soft one-hot (B, 128, 8, 8) → decoder → (B, 3, 32, 32)
  encoding: argmax → tokens (B, 8, 8) → flattened to (B, 64)
caption "a red circle" → [BOS, a, red, circle] (4 tokens)
sequence = 4 text tokens + (64 image tokens + 12) → (B, 68) → transformer → logits (B, 67, 140)
generation: 4 text tokens → 64 sampled image tokens → (N, 8, 8) → dVAE decoder → (N, 3, 32, 32)
```

## Deliberate Simplifications vs. the Paper

1. **Data:** 6,000 synthetic 32×32 images of single coloured shapes with 3-word captions, instead of 250 million 256×256 internet images with captions.
2. **dVAE:** 128 codes and 8×8 = 64 tokens instead of 8192 codes and 32×32 = 1024 tokens; small conv nets instead of ResNets; a Gaussian (MSE) pixel likelihood instead of logit-Laplace; no iterate averaging.
3. **Text:** a word-level vocabulary of 12 tokens and fixed-length captions, instead of up to 256 BPE tokens from 16,384 with a learned padding token per position.
4. **Transformer:** about 3.2M parameters, 4 layers and a plain causal mask, instead of a 12-billion-parameter sparse transformer with 64 layers and row, column and convolutional attention masks.
5. **Sampling:** no reranking; the paper ranks up to 512 samples per caption with CLIP.
6. **Evaluation:** a colour rule and a small shape classifier on held-out colour-shape combinations, instead of FID, IS and human evaluation on MS-COCO and CUB.
