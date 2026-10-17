---
title: "SSD: Code architecture"
parent: "033 SSD"
nav_order: 1
---
# Code Architecture: SSD Notebook

## Overview

The notebook builds an SSD-style single-shot detector from scratch in PyTorch and trains it on a synthetic multi-scale shapes dataset. Everything specific to SSD is implemented by hand: default box generation, the matching strategy, offset encoding and decoding, the MultiBox loss with hard negative mining, and the detection head layout. The only library pieces used are `torchvision.ops.nms` and `box_iou`. After training, it evaluates VOC-style mAP@0.5 and recall by object size. It then visualizes **which feature map each detection came from**, which is the central claim of the paper. Runtime is about 18 minutes on a Kaggle T4 (16.4 min of training); the validated run reaches mAP@0.5 = 0.959.

## Section-by-Section Breakdown

### 1. Imports & Configuration
- Seeds, device, and all hyperparameters in one place.
- `FMAP_SIZES = [32, 16, 8, 4, 2, 1]` are the 6 source maps, like SSD300's 38/19/10/5/3/1.
- `ASPECT_RATIOS = [[2], [2,3], [2,3], [2,3], [2], [2]]` gives 4 or 6 boxes per location, the same pattern as SSD300.
- `S_FIRST=0.1, S_MIN=0.2, S_MAX=0.9` are the scales. The first map uses 0.1, as SSD300 does for `conv4_3`.
- Training: 50 epochs, batch 32, AdamW at 1e-3 with OneCycle, IoU match threshold 0.5, and a 3:1 negative:positive ratio.

### 2. Synthetic Multi-Scale Shapes Dataset
- `make_sample(rng)` draws a noisy background, random distractor lines, and 1–6 non-overlapping shapes. Classes are 1 = circle, 2 = rectangle (random aspect ratio) and 3 = triangle, with random colors that don't depend on class.
- Object size is log-uniform from 16 to 170 px, so each feature-map scale has work to do.
- `build_split(n, seed)` returns a `uint8` tensor of shape `(n, 3, 256, 256)` plus per-image lists of normalized `xyxy` boxes and labels. 4,000 train and 500 val images are generated in memory.

### 3. Multi-Scale Default Boxes
- `compute_scales()` implements `s_k = s_min + (s_max − s_min)(k−1)/(m−2)` over the maps after the first.
- `make_default_boxes()` loops over map, then row, then column. For each location it emits the square box at `s_k`, the extra square box at `√(s_k s_{k+1})`, and an `(s√a, s/√a)` / `(s/√a, s√a)` pair for each aspect ratio. It returns `(6132, 4)` boxes in `cxcywh` format and a `(6132,)` tensor recording each box's source map.
- An assertion checks that the total equals `Σ f²·A`. The order matters because it must match how the heads are flattened.
- A plot shows the default boxes at one central cell of each map.

### 4. Matching Strategy & Box Encoding
- `encode` / `decode` implement the paper's offset parameterization with variances (0.1, 0.2), and a round-trip test checks them.
- `match(gt_boxes, gt_labels, ...)` follows the paper:
  1. Each ground truth takes its best-IoU default box. That box's IoU is forced to 2.0 so it always counts as a positive.
  2. Every other default box with IoU > 0.5 to some ground truth becomes a positive for that ground truth. Everything else is background (label 0).
- It returns encoded loc targets `(N, 4)` and class targets `(N,)`.
- Diagnostics show the average number of positives per object and the number of positives per feature map. A plot shows matched default boxes colored by feature map.

### 5. SSD Model
- `conv_bn` is a Conv → BatchNorm → ReLU block. BatchNorm is there because training starts from scratch, with no pretrained VGG.
- `L2Norm` is the paper's `conv4_3` normalization with a learnable per-channel scale (initialized to 20).
- `SSD` is built from these parts:
  - `base`: 8 conv layers and 3 max-pools, taking 256×256 down to `(128, 32, 32)`. This map is L2-normalized and becomes source map 0.
  - `fc7`: a max-pool and two convs, giving `(256, 16, 16)`.
  - `extras`: 4 blocks of 1×1 reduce plus 3×3 stride-2 conv, giving 8×8, 4×4, 2×2 and 1×1 maps.
  - `loc_heads` / `conf_heads`: a 3×3 conv per source map with `A·4` and `A·C` output channels.
- `forward` permutes each head output to `(B, H, W, A·k)` and reshapes to `(B, H·W·A, k)`, then concatenates over maps to `loc (B, 6132, 4)` and `conf (B, 6132, 4)`.
- Assertions check the source map sizes and that the number of predictions equals the number of default boxes. The model has about 1.9M parameters.

### 6. MultiBox Loss with Hard Negative Mining
- `multibox_loss(loc_p, conf_p, loc_t, labels_t)` computes:
  - Smooth L1 over positive boxes only.
  - Per-box cross-entropy. Positive losses are zeroed in a detached copy, and negatives are ranked with a double `sort`. Each image keeps its top `3 × num_pos` negatives.
  - Confidence loss summed over positives and the selected negatives.
  - Both terms divided by the total number of positives `N` (α = 1).

### 7. Training
- Images stay on the GPU as `uint8` and are normalized per batch.
- Random horizontal flips are applied to images and boxes together, then each image is matched against the default boxes on the GPU.
- Mixed precision (`autocast` + `GradScaler`), with the loss computed in fp32.
- Train and val loc/conf losses are logged per epoch and plotted afterwards.

### 8. Inference
- `detect(model, x)` applies softmax, decodes all default boxes, and keeps boxes with score > 0.05. It runs per-class NMS at IoU 0.45 and keeps the top 100 per image.
- It returns boxes, scores, labels **and the source feature-map index** of every detection.

### 9. Evaluation
- `average_precision` uses VOC-style greedy matching at IoU ≥ 0.5 and computes AP as the area under the precision-recall curve, using all points. Results are per-class AP, mAP, and precision-recall curves.
- Recall at score ≥ 0.5 is reported for small (< 32 px), medium (32–96 px) and large (> 96 px) objects.

### 10. Detections Across Feature-Map Scales
- Validation images with detections colored by source map.
- A bar chart of detection counts per map, and a box plot of detected box size per map against the default box scale `s_k`.
- Per-map objectness heatmaps showing `max over anchors of (1 − P(background))` at each cell.

### 11. Summary
- Prints the configuration, final losses, per-class AP, mAP and recall by size.

## Key Functions/Classes

| Name | Role |
|---|---|
| `make_sample`, `build_split` | Synthetic multi-scale shapes dataset |
| `compute_scales`, `make_default_boxes` | Paper's default box scales and aspect ratios |
| `encode`, `decode` | Offset parameterization with variances |
| `match` | Two-step matching strategy (best default box per ground truth, then IoU > 0.5) |
| `L2Norm` | `conv4_3`-style normalization with learnable scale |
| `SSD` | Backbone, extra layers, and 6 loc/conf heads |
| `multibox_loss` | Smooth L1 + softmax CE with 3:1 hard negative mining |
| `detect` | Decode + per-class NMS, tracking the source feature map |
| `average_precision` | VOC-style AP@0.5 |

## Data Flow

```
uint8 images (B,3,256,256) ──normalize──▶ SSD.base ─▶ (B,128,32,32) ─L2Norm─▶ source 0
                                              └▶ fc7 ─▶ (B,256,16,16) ─▶ source 1
                                                   └▶ extras ─▶ 8x8, 4x4, 2x2, 1x1 ─▶ sources 2..5
each source ─3x3 conv─▶ loc (B,H,W,A*4) / conf (B,H,W,A*C) ─flatten+concat─▶ loc (B,6132,4), conf (B,6132,4)

GT boxes ──match()──▶ loc_t (B,6132,4), labels_t (B,6132) ──multibox_loss──▶ loss
inference: softmax + decode(loc, default_boxes) ─▶ threshold ─▶ per-class NMS ─▶ top-100 detections
```

## Deliberate Simplifications vs Full Paper

- **Data:** synthetic shapes instead of PASCAL VOC / COCO, so training is fast, needs no download, and controls the object size distribution.
- **Backbone:** a small VGG-style CNN with BatchNorm trained from scratch, instead of ImageNet-pretrained VGG-16 with dilated `fc6`.
- **Resolution:** a 256×256 input with maps 32/16/8/4/2/1, instead of 300×300 with 38/19/10/5/3/1. There are 6,132 default boxes instead of 8,732.
- **Optimizer:** AdamW + OneCycle + AMP for 50 epochs, instead of SGD with momentum and step LR over about 120k iterations.
- **Augmentation:** horizontal flips only. The paper's random IoU-constrained crops and zoom-out expansion are not implemented, even though the paper shows they matter a lot on real data.
- **Inference:** a confidence threshold of 0.05 and top-100 per image, instead of 0.01 and top-200.
