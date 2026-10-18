---
title: "YOLO: Code architecture"
parent: "032 YOLO"
nav_order: 1
---
# Code Architecture — YOLO Notebook

## Overview

The notebook implements a simplified YOLO (v1) object detector from scratch in PyTorch. It trains on synthetic images of colored shapes (red circles, green rectangles, blue triangles on a black background), predicting bounding boxes and class labels with YOLO's 7×7 grid, and measures the result with mAP@0.5 on 200 test images it never trained on.

## Section-by-Section Breakdown

### 1. Imports & Configuration

Sets random seeds and defines the key constants:
- `S = 7` — grid size (7×7)
- `B = 1` — bounding boxes per cell (simplified from the paper's B=2)
- `C = 3` — number of classes (circle, rectangle, triangle)
- `IMG_SIZE = 224` — input image resolution (reduced from 448 for speed)
- `LAMBDA_COORD = 5`, `LAMBDA_NOOBJ = 0.5` — loss weights, as in the paper
- `BATCH_SIZE = 32`, `EPOCHS = 20`, `LR = 1e-3`
- `TRAIN_PER_EPOCH = 3000` — new synthetic training images every epoch
- `TEST_SIZE = 200` — fixed test images

### 2. Synthetic Dataset

**Class: `ShapeDataset`**
- `make_image(idx)` draws 1–3 shapes on a 224×224 black image with PIL and returns the image and its boxes `(x1, y1, x2, y2, class_id)`. Each index uses its own random seed, so image `idx` is always the same (fixed, reproducible test set).
- Shapes never overlap, and no two shape centers fall in the same grid cell. With B=1, YOLO v1 can predict only one object per cell, so a second object there would be impossible to learn.
- `__getitem__` returns the image tensor `(3, 224, 224)` and the target tensor `(7, 7, 8)`.
- Target per cell: `[confidence, x, y, w, h, class_0, class_1, class_2]`
  - x, y: the box center's offset inside its cell, in [0, 1]
  - w, h: box size relative to the whole image; the loss takes their square roots

**Class: `TrainShapeDataset`** — same as above, but setting `.epoch` shifts the seeds, so every epoch gets 3,000 brand-new images. This prevents the model from memorizing a small training set.

The cell also plots 4 example images with their ground-truth boxes.

### 3. YOLO Network Architecture

**Class: `YOLONet`** (about 7.2 million parameters)
- **Backbone:** 10 conv layers. 3×3 convs alternate with 1×1 "reduction" convs (the GoogLeNet / Network-in-Network idea the paper uses), each followed by BatchNorm and Leaky ReLU (slope 0.1). Five 2×2 max-pools shrink 224 → 112 → 56 → 28 → 14 → 7.
- **Detection head:** Flatten → FC(7·7·256 → 512) → Leaky ReLU → Dropout(0.5) → FC(512 → 7·7·8 = 392)
- **Output:** sigmoid, then reshape to `(batch, 7, 7, 8)`. The paper's last layer is linear; the sigmoid keeps every output in [0, 1] to match the targets.
- **BatchNorm** is not in YOLOv1 (it arrived in YOLOv2). It makes training from scratch, with no ImageNet pretraining, faster and steadier.

**Data flow:**
```
Input: (batch, 3, 224, 224)
  → conv/pool backbone → (batch, 256, 7, 7)
  → Flatten → (batch, 12544)
  → FC → (batch, 512)
  → FC → (batch, 392)
  → Sigmoid + reshape → (batch, 7, 7, 8)
```

### 4. YOLO Loss Function

**Class: `YOLOLoss`** — the multi-part sum-squared error of the paper's Equation 3, summed over cells and averaged over the batch:

1. **Coordinates** (× λ_coord = 5), at cells with an object: `(x − x̂)² + (y − ŷ)² + (√w − √ŵ)² + (√h − √ĥ)²`
2. **Confidence, object cells:** `(1 − Ĉ)²`. The paper's target is the IoU between predicted and true box; we use 1 for simplicity.
3. **Confidence, empty cells** (× λ_noobj = 0.5): `(0 − Ĉ)²`
4. **Classification**, at cells with an object: sum over classes of `(p_c − p̂_c)²`

It returns the total loss and each part (per image) for logging.

### 5. IoU and Non-Maximal Suppression

- `compute_iou(box1, box2)` — overlap of two `(x1, y1, x2, y2, ...)` boxes
- `non_max_suppression(dets, iou_threshold=0.4)` — keeps the highest-scoring box and drops lower-scoring boxes *of the same class* that overlap it by IoU ≥ 0.4

### 6. Decoding Predictions

`decode_prediction(pred, S, C, score_thresh, img_size)` turns the `(7, 7, 8)` output into boxes in pixels. Each box's score is confidence × class probability (as in the paper); boxes below `score_thresh` are dropped, then NMS runs.

### 7. Evaluation

`evaluate(model, dataset)` returns:
- **mAP@0.5** — per-class average precision (PASCAL-style 11-point interpolation) with IoU ≥ 0.5 to count a detection as correct, averaged over the 3 classes
- **per-class AP**
- **recall** — share of real shapes that get a correct box (right class, IoU ≥ 0.5) with score ≥ 0.2

It runs once before training as a baseline (0.0).

### 8. Training Loop

- Adam (lr 1e-3) with cosine learning-rate decay over 20 epochs
- Each epoch: new training images, forward pass, YOLO loss, backward pass, optimizer step
- Logs the total loss and its 4 parts every epoch, and test mAP@0.5 every 5 epochs

### 9–11. Curves, Pictures, Final Numbers

- Loss curves (total + 4 parts) and test mAP over training
- Predicted boxes on 8 unseen test images, with class and score labels and "detected / real" counts
- Final per-class AP, mAP@0.5 and recall on the 200 test images

## Key Functions

| Function / Class | Purpose |
|---|---|
| `ShapeDataset.make_image(idx)` | Draws a seeded synthetic image and returns its boxes |
| `ShapeDataset.__getitem__(idx)` | Encodes the boxes into the 7×7×8 YOLO target |
| `YOLONet` | Conv backbone + fully connected detection head |
| `YOLOLoss.forward(pred, target)` | Multi-part YOLO loss |
| `compute_iou(box1, box2)` | IoU between two boxes |
| `non_max_suppression(dets, iou_threshold)` | Removes duplicate boxes of the same class |
| `decode_prediction(pred, S, C, score_thresh)` | Network output → scored boxes |
| `evaluate(model, dataset)` | mAP@0.5, per-class AP and recall |

## Deliberate Simplifications vs Full Paper

| Aspect | Paper | Notebook |
|---|---|---|
| Input size | 448×448 | 224×224 |
| Conv layers | 24 | 10 (+ BatchNorm, from YOLOv2) |
| Boxes per cell (B) | 2 | 1 |
| Classes (C) | 20 (PASCAL VOC) | 3 (synthetic shapes) |
| Dataset | PASCAL VOC 2007/2012 | Synthetic colored shapes, 3,000 new images per epoch |
| Pretraining | ImageNet (about a week) | None (from scratch) |
| Optimizer / epochs | SGD with momentum, ~135 epochs | Adam with cosine decay, 20 epochs |
| Object-cell confidence target | IoU(pred, truth) | 1 |
| Output layer | Linear | Sigmoid |
| Output tensor | 7×7×30 | 7×7×8 |
