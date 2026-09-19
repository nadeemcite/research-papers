# Code Architecture — Show and Tell: A Neural Image Caption Generator

This notebook implements a **CNN encoder + LSTM decoder** image captioning model from scratch in PyTorch, following the architecture described in Vinyals et al. (2014). It uses a pretrained ResNet-18 (substituting for the paper's GoogLeNet/Inception) as the image encoder and a from-scratch LSTM as the language decoder. The model is trained on a small synthetic image-caption dataset, and then used to generate captions for sample images with beam search.

## Section 1 — Setup and imports

**What it does:** Installs PyTorch, torchvision, matplotlib, numpy, and Pillow. Imports all required modules and seeds RNGs for reproducibility.

**Key packages:** `torch`, `torchvision`, `torch.nn`, `matplotlib`, `numpy`, `PIL`

## Section 2 — Vocabulary builder

**What it does:** Builds a word vocabulary from a list of captions, keeping words that appear at least `min_freq` times (the paper uses 5). Creates `<START>`, `<END>`, `<UNK>`, and `<PAD>` special tokens.

**Class:** `Vocabulary`
- `add_word(word)` — adds a word to the vocabulary if not already present.
- `add_caption(text)` — tokenises a caption (lowercase + split on non-alphanumeric) and adds each token.
- `numericalise(text)` — converts a caption string to a list of integer token IDs, with `<UNK>` for OOV words.
- `build_vocab(captions, min_freq)` — counts word frequencies and keeps only words above threshold.

**Shapes:** Vocabulary maps `str → int` (word to index). A caption of length N becomes a list of N+2 integers (with `<START>` and `<END>`).

**Simplification vs paper:** The paper uses basic tokenisation; we use a simple regex-based tokenizer. The vocabulary size is much smaller because we use a synthetic dataset rather than COCO (which has ~12,000 words).

## Section 3 — Synthetic image-caption dataset

**What it does:** Creates a small synthetic dataset where images are simple geometric shapes (coloured circles, rectangles, triangles on coloured backgrounds) and captions describe the scene (e.g., "a red circle on a blue background"). This avoids downloading multi-GB datasets while still demonstrating the full training loop.

**Class:** `ShapeCaptionDataset(torch.utils.data.Dataset)`
- `__init__(num_samples, image_size, vocab)` — generates `num_samples` random images and their corresponding captions.
- `__getitem__(idx)` — returns `(image_tensor, caption_tensor)` where image is `(3, H, W)` float tensor and caption is `(L,)` long tensor.
- `_generate_image()` — draws a random shape on a random background colour using PIL, returns a numpy array.
- `_generate_caption(shape, colour, bg_colour)` — creates a templated caption string like "a red circle on a blue background".

**Shapes:**
- Image: `(3, 64, 64)` — normalised to [0, 1]
- Caption: variable length `(L,)`, padded to max length in batch

**Simplification vs paper:** The paper uses COCO/Flickr30k with real photographs and human-written captions. We use synthetic shapes with templated captions so the notebook runs on a single GPU in minutes. The *architecture* is identical; only the data scale differs.

## Section 4 — CNN image encoder

**What it does:** Wraps a pretrained ResNet-18 (from torchvision) and replaces the final classification layer with a linear projection to the LSTM input dimension (512). The image is passed through the CNN and the output of the penultimate layer is projected to the embedding dimension.

**Class:** `CNNEncoder(nn.Module)`
- `__init__(embed_dim)` — loads `resnet18(pretrained=True)`, removes the final FC layer, adds a `Linear(512, embed_dim)` projection.
- `forward(images)` — `images: (B, 3, H, W)` → `features: (B, embed_dim)`

**Shapes:**
- Input: `(B, 3, 224, 224)` — standard ImageNet size after resize/normalise
- Output: `(B, 512)` — one feature vector per image

**Simplification vs paper:** The paper uses GoogLeNet/Inception-v1 (and later Inception-v2/v3). We use ResNet-18 because it's lighter and pretrained weights are readily available in torchvision. The paper also experiments with fine-tuning the CNN; we keep it frozen for simplicity.

## Section 5 — LSTM decoder from scratch

**What it does:** Implements an LSTM cell from scratch (Eqs. from the paper's Figure 2), then wraps it in a decoder that takes image features as the initial input and generates captions word by word.

**Class:** `LSTMCellScratch(nn.Module)`
- **Input gate:** `i = σ(W_i·x + U_i·h_{t-1} + b_i)`
- **Forget gate:** `f = σ(W_f·x + U_f·h_{t-1} + b_f)`
- **Output gate:** `o = σ(W_o·x + U_o·h_{t-1} + b_o)`
- **Candidate:** `g = tanh(W_c·x + U_c·h_{t-1} + b_c)`
- **Cell state:** `c_t = f ⊙ c_{t-1} + i ⊙ g`
- **Hidden state:** `h_t = o ⊙ tanh(c_t)`

**Class:** `CaptionDecoder(nn.Module)`
- `__init__(vocab_size, embed_dim, hidden_dim)` — creates word embedding `nn.Embedding`, `LSTMCellScratch`, and output projection `nn.Linear(hidden_dim, vocab_size)`.
- `forward(features, captions)` — teacher-forcing training pass. `features: (B, embed_dim)` fed as x_{-1}; captions: `(B, L)` fed one word at a time. Returns logits `(B, L, vocab_size)`.
- `generate(features, max_len, beam_size)` — beam search inference. Returns the highest-probability caption as a list of token IDs.

**Shapes:**
- Features: `(B, 512)`
- Captions (training): `(B, L)` — includes `<START>` token
- Output logits: `(B, L, vocab_size)`
- Generated caption (inference): list of ints, ending at `<END>`

**Simplification vs paper:** The LSTM equations are faithful to the paper. We use a single-layer LSTM (the paper also uses a single layer). Beam search is implemented but simplified to width 3 (the paper uses width 20). Dropout is applied between LSTM steps as in the paper.

## Section 6 — Full NIC model

**What it does:** Combines the CNN encoder and LSTM decoder into a single `nn.Module`.

**Class:** `NICModel(nn.Module)`
- `__init__(encoder, decoder)` — stores both sub-modules.
- `forward(images, captions)` — encodes images, then decodes captions. Returns logits.
- `caption_image(image, beam_size)` — generates a caption for a single image.

## Section 7 — Training loop

**What it does:** Trains the NIC model on the synthetic dataset using cross-entropy loss (ignoring `<PAD>` tokens) and Adam optimiser. Logs loss every epoch.

**Key details:**
- Loss: `nn.CrossEntropyLoss(ignore_index=PAD_idx)` — standard next-token prediction loss.
- Optimiser: Adam, lr=1e-3 (the paper uses SGD with fixed lr; Adam converges faster on the small dataset).
- Batch size: 32
- Epochs: 20 (the paper trains for much longer on much larger data)
- Teacher forcing: ground-truth captions fed during training.

## Section 8 — Caption generation and visualisation

**What it does:** After training, generates captions for held-out test images using beam search, displays the images alongside generated captions using matplotlib.

**Output:** A grid of images with their generated captions printed below each one.

## Deliberate simplifications vs the full paper

| Aspect | Paper | Notebook |
|--------|-------|----------|
| CNN architecture | GoogLeNet/Inception-v1 | ResNet-18 (pretrained) |
| Dataset | MS COCO (82K images, 400K captions) | Synthetic shapes (~2K samples) |
| Vocabulary | ~12,000 words | ~30 words |
| Training time | Hours on multi-GPU | Minutes on single GPU |
| Beam search width | 20 | 3 |
| Fine-tuning CNN | Yes (later versions) | No (frozen) |
| Ensembling | Yes | No |
| Evaluation | BLEU-1/4, METEOR, CIDEr | Training loss + qualitative |

The architectural design (CNN encoder → embedding → LSTM decoder with teacher forcing and beam search) is a faithful reproduction. The simplifications are purely about dataset scale and training budget to make the notebook runnable in a Colab/Kaggle environment.
