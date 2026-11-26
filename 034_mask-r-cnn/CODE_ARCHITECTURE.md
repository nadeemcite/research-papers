---
title: "Mask R-CNN: Code architecture"
parent: "034 Mask R-CNN"
nav_order: 1
---
# Code Architecture — Mask R-CNN Notebook

## Overview

The notebook builds a small Mask R-CNN in PyTorch: a 4-block CNN backbone, a Region Proposal Network (RPN) with anchors, RoIAlign, and a head with parallel box and mask branches that predicts one sigmoid mask per class. It trains on synthetic 128×128 images of rectangles, circles and triangles, evaluates mask mAP at IoU 0.5 on held-out images, and compares a from-scratch RoIAlign with a quantized RoIPool. RoIAlign is written from scratch for illustration; training and inference use torchvision's `roi_align` (same idea, much faster) and torchvision's NMS. Everything runs on the CPU.

## Section-by-Section Breakdown

### Setup
- PyTorch, torchvision, NumPy, matplotlib, PIL; nothing is installed
- Seeds 42; runs on the CPU (default PyTorch threads)
- Figures as JPEG at 80 dpi; on Kaggle an `atexit` hook copies the executed notebook to `executed_solution.ipynb`

### 1–2. Synthetic Shape Dataset
- `ShapeDataset(num_images, image_size=128, seed)`: 1–4 shapes per image (rectangle = 1, circle = 2, triangle = 3, background = 0), random colours, positions and sizes; returns the image (3, 128, 128), boxes (N, 4), labels (N,) and binary masks (N, 128, 128)
- 400 training images and 100 test images; `visualize_sample` draws boxes, labels and masks

### 3. Anchor Generation
- `generate_anchors`: anchors at each cell of the 8×8 feature map (stride 16); the model uses scales 32, 64, 96 and aspect ratios 0.5, 1, 2 (9 per cell, 576 in total)

### 4. Bounding Box Utilities
- `box_iou`, `box_encode` / `box_decode` (Faster R-CNN offsets), and `nms` (torchvision's greedy NMS)

### 5. RoIAlign
- `roi_align(features, rois, output_size, spatial_scale, sampling_ratio=4)`: from scratch, no rounding of RoI or bin coordinates; 4×4 bilinear samples per bin, averaged
- `fast_roi_align`: torchvision's `roi_align` (sampling ratio 4, `aligned=False`), used for training and inference

### 6. Backbone
- `SimpleBackbone`: four blocks of Conv 3×3 → BatchNorm → ReLU → MaxPool, channels 16 → 32 → 64 → 128; (B, 3, 128, 128) → (B, 128, 8, 8)

### 7. Region Proposal Network
- `RPN`: 3×3 conv (128) + ReLU, then 1×1 convs for objectness (9 logits per cell) and box deltas (9 × 4)
- `assign_anchors`: positive if IoU ≥ 0.7 or the best anchor for a GT box, negative if IoU < 0.3
- `sample_anchors`: 256 anchors per image, up to half positive

### 8. Detection Head
- `MaskRCNNHead`: box branch on 7×7 RoI features (flatten → FC 1024 → FC 1024 → class logits for 4 classes and class-specific box deltas); mask branch on 14×14 RoI features (three 3×3 convs with 256 channels → 2×2 transposed conv to 28×28 → 1×1 conv to 4 mask logits)

### 9. Full Mask R-CNN Model
- `MaskRCNN.forward` in training: RPN loss (BCE + smooth L1 on sampled anchors); proposals from the RPN (decode, top 200, NMS 0.7, top 100) plus the GT boxes; up to 64 positive (IoU ≥ 0.5) and 192 negative RoIs per image; RoIAlign with the correct batch index for each image; classification (cross-entropy), box (smooth L1 on positives) and mask loss (per-pixel BCE on the GT class's 28×28 mask, positives only, targets cropped from the GT mask and resized)
- Returns `loss`, `rpn_cls_loss`, `rpn_box_loss`, `cls_loss`, `box_loss`, `mask_loss`
- Inference: class scores (softmax), box refinement, per-class NMS 0.5, top 50 detections; each 28×28 mask is resized to its box and pasted into a 128×128 mask (threshold 0.5)

### 10. Training
- `EPOCHS = 6`, `BATCH_SIZE = 4`, Adam (learning rate 1e-3, weight decay 1e-4), learning rate ×0.1 after 4 epochs, gradient clipping at 5; prints each loss per epoch and the elapsed time

### 11. Training Loss Curves
- The total loss and the five loss parts per epoch

### 12. Inference Examples
- `visualize_detections`: detections with score ≥ 0.5 on four test images (boxes, labels with scores, pasted masks)

### 13. Detection Confidence by Object Size
- Confidence of detections that overlap a GT box with IoU > 0.5, grouped by object area

### 14. Mask Quality Evaluation
- `compute_mask_iou` and `compute_ap` (11-point interpolation)
- `evaluate_map`: over all 100 test images, detections with score ≥ 0.05 are matched per class to unmatched GT masks at mask IoU ≥ 0.5; prints AP per class, mAP and the mean mask IoU of correct detections

### 15. Ground Truth vs Prediction
- Side-by-side GT and predictions for three test images

### 16. RoIAlign vs RoIPool
- `roi_pool_quantized`: RoIPool with rounded coordinates and bins; compared with the from-scratch `roi_align` on a random 16×16 feature map and a fractional RoI

### 17. More Detection Examples
- Six more test images and a short printed summary (mAP, epochs, final loss)

### 18. Summary
- `### What the run above shows` with the saved run's real numbers

## Key Functions/Classes

| Name | Purpose |
|---|---|
| `ShapeDataset` | Synthetic instance-segmentation data |
| `generate_anchors`, `assign_anchors`, `sample_anchors` | RPN anchors and targets |
| `box_iou`, `box_encode`, `box_decode`, `nms` | Box utilities |
| `roi_align` / `fast_roi_align` | From-scratch and torchvision RoIAlign |
| `SimpleBackbone`, `RPN`, `MaskRCNNHead` | The network parts |
| `MaskRCNN` | Full model: training losses and inference with pasted masks |
| `compute_mask_iou`, `compute_ap`, `evaluate_map` | Mask mAP@0.5 evaluation |
| `roi_pool_quantized` | RoIPool for the alignment comparison |

## Data Flow / Shapes

```
image (B, 3, 128, 128) → SimpleBackbone → features (B, 128, 8, 8)
features → RPN → objectness (B, 576), deltas (B, 576, 4) → proposals (≤100 per image)
training RoIs = proposals + GT boxes → 256 sampled RoIs → roi_align 7×7 (box) and 14×14 (mask)
box branch → class logits (N, 4), deltas (N, 4, 4); mask branch → mask logits (N, 4, 28, 28)
inference: per-class NMS → top 50 → 28×28 mask resized into its box → (n, 128, 128) binary masks
```

## Deliberate Simplifications vs. the Paper

1. **Data:** synthetic 128×128 shapes with 3 classes instead of COCO.
2. **Backbone:** a 4-block CNN trained from scratch (stride 16, no FPN) instead of ImageNet-pre-trained ResNet/ResNeXt with FPN.
3. **Training:** Adam for 6 epochs on 400 images instead of SGD for 160k iterations on 8 GPUs.
4. **RoIs:** GT boxes are added to the training RoIs; 256 RoIs per image with up to 25% positives.
5. **Speed:** torchvision's `roi_align` and NMS are used for training and inference; the from-scratch RoIAlign appears in the comparison section.
6. **Evaluation:** mask AP at IoU 0.5 only (11-point interpolation), not COCO's AP averaged over IoU 0.5–0.95.
