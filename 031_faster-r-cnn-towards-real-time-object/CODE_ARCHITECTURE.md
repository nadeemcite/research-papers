---
title: "Faster R-CNN: Code architecture"
parent: "031 Faster R-CNN"
nav_order: 1
---
# Code Architecture — Faster R-CNN

## Overview

The notebook builds a small Faster R-CNN from scratch in PyTorch: a CNN backbone that produces a shared feature map, a Region Proposal Network (RPN) with anchor boxes, RoI pooling, and a Fast R-CNN detection head. It trains the RPN and detector together (approximate joint training) on synthetic 64×64 images of white squares and circles, then measures AP@0.5 per class, mAP and RPN recall on held-out images, and draws the detections.

## Section-by-Section Breakdown

### Setup (cell 1)
- PyTorch, torchvision, NumPy, matplotlib; nothing is installed
- Seeds `torch`, NumPy and `random` with 42; one CPU thread and deterministic kernels so reruns give the same numbers; runs on the CPU on Kaggle
- Figures as JPEG at 80 dpi; on Kaggle an `atexit` hook copies the executed notebook to `executed_solution.ipynb`
- Constants: `IMAGE_SIZE = 64`, `FEATURE_MAP_SIZE = 8`, `NUM_CLASSES = 2` (square, circle; background is class index 2), `ANCHOR_SCALES = [8, 16, 24]`, `ANCHOR_RATIOS = [0.5, 1.0, 2.0]`, 9 anchors per location, 576 in total

### 1–2. Synthetic Detection Dataset
- `ToyDetectionDataset`: 64×64 black images with 1–3 white objects; each object is a square (class 0) or a circle (class 1) of size 10–20 pixels at a random position, with its box `(x1, y1, x2, y2)`
- 1,000 training images and 200 test images; batch size 16, seeded shuffle for training
- A figure shows four training images with their boxes

### 3. CNN Backbone
- `Backbone`: three blocks of 3×3 conv + ReLU + 2×2 max-pool, channels 32 → 64 → 128
- (B, 1, 64, 64) → (B, 128, 8, 8); 92,672 parameters; the feature stride is 8 pixels

### 4–5. Anchor Generation
- `generate_anchors`: at each of the 8×8 cells (centres every 8 pixels), 3 scales × 3 aspect ratios with width `scale·√ratio` and height `scale/√ratio`; 576 anchors as `(x1, y1, x2, y2)`
- A figure draws the 9 anchors at cell (4, 4)

### 6. IoU and Box Utilities
- `compute_iou`: pairwise IoU matrix
- `encode_boxes` / `decode_boxes`: the paper's parameterization `dx = (x − xa)/wa`, `dy = (y − ya)/ha`, `dw = log(w/wa)`, `dh = log(h/ha)`
- `clamp_boxes`: clips boxes to the image

### 7. Anchor Labeling
- `assign_targets_to_anchors`: positive if IoU ≥ 0.7 or the best anchor for a GT box, negative if IoU < 0.3, otherwise ignored
- `sample_anchors_for_training`: up to 32 positive and 32 negative anchors per image

### 8. Non-Maximum Suppression
- `apply_nms`: `torchvision.ops.nms` at IoU 0.7, keeping at most `max_proposals`

### 9. Region Proposal Network
- `RPNHead`: 3×3 conv (128 channels) + ReLU, then two 1×1 convs: objectness (2 × 9 outputs) and box deltas (4 × 9 outputs), reshaped to (B, 576, 2) and (B, 576, 4); 154,550 parameters

### 10. RPN Loss
- `compute_rpn_loss`: cross-entropy on the sampled anchors plus smooth L1 on the deltas of positive anchors

### 11. Proposal Generation
- `generate_proposals`: softmax objectness, decode and clamp boxes, keep the top 200 by score, NMS at 0.7, keep at most 30 during training and evaluation (50 by default, as in the proposal plot)

### 12. RoI Pooling
- `roi_pool`: `torchvision.ops.roi_pool` with output 5×5 and `spatial_scale = 1/8`; (N, 4) boxes → (N, 128, 5, 5)

### 13. Detection Head
- `DetectionHead`: flatten 128 × 5 × 5 = 3,200 → FC 128 → ReLU → FC 128 → ReLU, then a classifier with 3 outputs (2 classes + background) and a class-specific box regressor with 3 × 4 outputs; 428,175 parameters

### 14. RoI Sampling and Detection Loss
- `sample_rois`: takes the detached proposals plus the GT boxes, labels each as foreground (IoU ≥ 0.5) or background (IoU < 0.5), and samples 32 RoIs with at most 25% foreground
- `compute_detection_loss`: cross-entropy over the sampled RoIs plus smooth L1 on the class-specific deltas of foreground RoIs

### 15. Full Faster R-CNN Model
- `FasterRCNN.forward`: backbone → RPN → proposals per image → RPN loss + detection loss on sampled RoIs; total loss = RPN cls + RPN reg + det cls + det reg
- `FasterRCNN.predict`: proposals → RoI pooling → class probabilities and class-specific box refinement → for each class, keep scores ≥ `score_thresh` (0.5 by default) and apply NMS at IoU 0.3
- 675,397 parameters in total

### 16. Training Loop
- Adam (learning rate 1e-3, weight decay 1e-4), gradient-norm clipping at 5, 20 epochs; prints each loss component and the elapsed time per epoch

### 17. Training Loss Curves
- The total loss and the four components per epoch

### 18. Visualize Predictions and Test-Set Evaluation
- Four test images with GT boxes (green, dashed) and detections with score ≥ 0.5 (red, with class and score)
- `average_precision`: all-point interpolated AP from scored detections
- Evaluation over all 200 test images with `score_thresh = 0.05`: a detection is a true positive if IoU ≥ 0.5 with an unmatched GT box of its class; prints AP@0.5 per class, mAP@0.5 and the RPN's recall (GT boxes covered by a top-30 proposal with IoU ≥ 0.5)

### 19. Visualize RPN Proposals
- For one test image: the GT box, the top 20 proposals before NMS, and the proposals after NMS

### 20. Summary
- `### What the run above shows`: the saved run's real numbers

## Key Functions/Classes

| Name | Purpose |
|---|---|
| `ToyDetectionDataset`, `collate_fn` | Synthetic squares-and-circles detection data |
| `Backbone` | Shared 8×8 feature map |
| `generate_anchors` | 9 anchors per feature-map cell |
| `compute_iou`, `encode_boxes`, `decode_boxes` | Box geometry and the paper's box parameterization |
| `assign_targets_to_anchors`, `sample_anchors_for_training`, `compute_rpn_loss` | RPN targets and loss |
| `RPNHead`, `generate_proposals` | Objectness and box deltas → proposals after NMS |
| `roi_pool` | Fixed 5×5 features per RoI |
| `DetectionHead`, `sample_rois`, `compute_detection_loss` | Fast R-CNN head, RoI sampling and loss |
| `FasterRCNN` | The full model: training losses and `predict` |
| `average_precision` | AP@0.5 for the evaluation |

## Data Flow / Shapes

```
image (B, 1, 64, 64) → Backbone → features (B, 128, 8, 8)
features → RPNHead → objectness (B, 576, 2), deltas (B, 576, 4)
per image: decode + top-200 + NMS(0.7) → proposals (≤30, 4)
training: proposals (detached) + GT → 32 sampled RoIs → roi_pool (32, 128, 5, 5) → DetectionHead
          → class logits (32, 3), box deltas (32, 12)
inference: roi_pool(proposals) → DetectionHead → per class: decode, score ≥ 0.5, NMS(0.3) → detections
```

## Deliberate Simplifications vs. the Paper

1. **Data:** synthetic 64×64 grayscale images with two shape classes instead of PASCAL VOC and MS COCO.
2. **Backbone:** a 3-layer CNN trained from scratch instead of an ImageNet-pre-trained ZF or VGG-16 network.
3. **Anchors:** sizes 8, 16 and 24 pixels on an 8×8 feature map instead of 128², 256² and 512² box areas on images with a 600-pixel shorter side.
4. **Training scheme:** approximate joint training, one of the paper's options, instead of the 4-step alternating training used in its experiments; Adam instead of SGD.
5. **Sampling sizes:** 64 anchors (instead of 256) for the RPN and 32 RoIs per image for the detector; the GT boxes are added to the RoIs.
6. **RoI pooling:** 5×5 output instead of 7×7.
7. **Proposals:** at most 30 after NMS instead of 2,000 for training and 300 for testing.
