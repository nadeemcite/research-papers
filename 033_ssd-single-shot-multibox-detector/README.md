---
title: "033 SSD"
nav_order: 34
permalink: "/033_ssd-single-shot-multibox-detector/"
---
# SSD: Single Shot MultiBox Detector
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/ssd-single-shot-multibox-detector)


**arXiv:** [1512.02325](https://arxiv.org/abs/1512.02325) | **Authors:** Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg | **Year:** 2016 (ECCV)

## Summary

SSD is a one-stage object detector that predicts every bounding box and class score in a single forward pass of one network, with no separate region-proposal stage. Faster R-CNN reached high accuracy but ran at about 7 FPS. YOLO ran in real time but lost accuracy, especially on small objects, because it predicted from one coarse 7×7 grid. SSD closes most of that gap by attaching small convolutional predictors to **several feature maps of decreasing resolution**. Each cell of each map owns a fixed set of **default boxes** (anchors) with different scales and aspect ratios. For each default box, the network predicts 4 offsets and a score per class. High-resolution early maps handle small objects, and coarse late maps handle large ones. With a 300×300 input, SSD300 reached 74.3% mAP on PASCAL VOC2007 test at 59 FPS, compared with Faster R-CNN's 73.2% at 7 FPS and YOLO's 63.4% at 45 FPS.

## Core Idea

1. **Multi-scale feature maps for detection.** SSD takes a truncated VGG-16 backbone (through `conv4_3` and a convolutionalized `fc7`) and adds extra conv layers that shrink the map step by step (38→19→10→5→3→1 for SSD300). Each of the 6 maps is a source of predictions.
2. **Default boxes.** Map *k* of *m* gets a scale `s_k = s_min + (s_max − s_min)(k − 1)/(m − 1)` with `s_min = 0.2` and `s_max = 0.9`. Each location gets boxes with aspect ratios `{1, 2, 3, 1/2, 1/3}` and one extra square box of scale `√(s_k·s_{k+1})`, for 4 or 6 boxes per location and 8,732 default boxes in SSD300.
3. **Convolutional predictors.** A 3×3 conv on each source map outputs `A·(C + 4)` channels per location, where *A* is the number of default boxes per location and *C* includes background.
4. **Matching and loss.** Each ground truth is matched to its best-overlap default box, plus every default box with Jaccard overlap > 0.5. The loss is `L = (L_conf + α·L_loc) / N`, where *N* is the number of matched boxes. `L_loc` is Smooth L1 on encoded offsets of positives only, and `L_conf` is softmax cross-entropy.
5. **Hard negative mining.** Almost all default boxes are background. SSD sorts negatives by confidence loss and keeps the top ones so that negatives:positives ≤ 3:1.

## Key Method Details

- **Backbone:** ImageNet-pretrained VGG-16. `fc6`/`fc7` are converted to conv layers (dilated `fc6`), and `pool5` is changed to 3×3 stride 1.
- **`conv4_3` normalization:** this early map has a different feature scale, so it is L2-normalized with a learned per-channel scale initialized to 20.
- **Box encoding:** center offsets are divided by the default box's width/height, and widths/heights are encoded as a log ratio. The reference code also divides by variances (0.1, 0.2).
- **Data augmentation:** random crops with minimum Jaccard overlap {0.1, 0.3, 0.5, 0.7, 0.9}, random patches, and horizontal flips. The ablation shows augmentation alone adds **+8.8% mAP**. A "zoom-out" expansion added later helps small objects by about 2–3% mAP.
- **Inference:** confidence threshold 0.01, per-class NMS at Jaccard 0.45, and the top 200 detections kept per image.
- **Results:** SSD300 gets 74.3% mAP on VOC2007 at 59 FPS on a Titan X. SSD512 gets 76.9% mAP. On COCO test-dev, SSD512 reaches 26.8 AP (46.5 AP@0.5).

### This notebook

- A synthetic dataset of circles, rectangles and triangles. Object sizes are log-uniform from 16 to 170 px, so several scales are needed.
- A small VGG-style backbone with BatchNorm, trained from scratch. It has 6 source maps (32, 16, 8, 4, 2, 1), and the 32×32 map is L2-normalized, as in the paper.
- From-scratch default boxes with the paper's `s_k` formula, the two-step matching strategy, box encoding, the MultiBox loss with 3:1 hard negative mining, per-class NMS, and VOC-style mAP@0.5.
- Plots showing **which feature map produced each detection** and the per-map objectness heatmaps. Together they show the multi-scale idea at work.

## What problem does it solve

Imagine you're looking at a photo from a park and someone asks, "Point out every dog, ball and bench." Some things are tiny, like a ball far away, and some are huge, like a bench right in front of the camera. A slow way is to cut the photo into thousands of little pieces and check each one, one by one. That's accurate but takes forever.

SSD does it in one look, using several pairs of glasses at once. One pair is a magnifying glass covered in a fine grid of small squares, great for spotting the tiny far-away ball. Another pair has a few large squares, great for the big bench. Every square already has a few "guess shapes" drawn in it (tall, wide, square), and the network just says, for each guess, "Is there something here? What is it? Nudge the shape a little this way." Because all the glasses look at the same time, SSD finds big and small things in one quick glance, fast enough for live video.

## Influence

SSD, together with YOLO, established the **one-stage detector** family and made anchor-based dense prediction over a feature pyramid standard practice. Its multi-scale prediction heads are a direct ancestor of the Feature Pyramid Network (FPN) and of RetinaNet. RetinaNet replaced SSD's hard negative mining with the focal loss to handle the same foreground/background imbalance. Later YOLO versions (v3 onward) adopted multi-scale anchor predictions in the same spirit. Paired with MobileNet backbones (MobileNet-SSD, SSDLite), SSD became one of the most widely deployed detectors on phones and embedded devices. It shipped in the TensorFlow Object Detection API, and torchvision provides `ssd300_vgg16` and `ssdlite320_mobilenet_v3_large`.

---

## Resources

**Kaggle Notebook:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/ssd-single-shot-multibox-detector)

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview: a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/research-papers/blob/main/033_ssd-single-shot-multibox-detector/solution.ipynb) | Complete from-scratch PyTorch implementation of the paper's architecture, with training loops, visualizations, and worked examples |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook: what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, blog posts, interview Q&A, citations, and common misconceptions about the paper |
