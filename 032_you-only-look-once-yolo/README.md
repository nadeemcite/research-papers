---
title: "032 YOLO"
nav_order: 33
permalink: "/032_you-only-look-once-yolo/"
---
# You Only Look Once: Unified, Real-Time Object Detection (YOLO)
![YOLO banner](../assets/banners/032.jpg)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/you-only-look-once-yolo)

- **Paper:** [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- **Authors:** Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
- **Year:** 2015 (arXiv preprint); published at [CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html)
- **Video:** YouTube Short coming on 18 Oct 2026 on [Backprop Daily](https://www.youtube.com/@BackpropDaily)

## Summary

YOLO reframes object detection as a single regression problem. Instead of running a classifier over many candidate regions (R-CNN checks about 2,000 proposed boxes per image; DPM slides a classifier across the image), YOLO uses a single convolutional network to simultaneously predict all bounding boxes and class probabilities for an entire image in one forward pass. The image is divided into a 7×7 grid; each grid cell predicts bounding box coordinates (x, y, w, h) and class-conditional probabilities for the objects whose centers fall within that cell. This unified architecture processes images at 45 FPS (155 FPS for Fast YOLO), making it one of the first deep-learning detectors fast enough for live video. The network is trained end-to-end with a multi-part sum-squared-error loss that balances localization and classification errors, using λ_coord = 5 to up-weight coordinate predictions. Bounding box dimensions use square-root parameterization so that errors in small boxes are penalized proportionally more than errors in large boxes.

## What Problem Does It Solve

Imagine you're looking at a photo of a busy street and you need to quickly point out every person, car, and bicycle — not just say "there's a car somewhere" but draw a box around each one and label it. Before YOLO, computers did this in a very roundabout way: they'd propose about 2,000 possible boxes in the photo, check each box one by one with a "what is this?" classifier, and then try to clean up and combine the answers. This was slow (taking seconds per photo) and often resulted in the computer "seeing" objects that weren't really there because it was looking at small patches without understanding the whole picture. YOLO solved this by looking at the entire photo all at once — just like a human does — and directly predicting where every object is and what it is, in a single glance. That's why it's called "You Only Look Once."

## Core Idea

- **Unified detection:** A single CNN predicts bounding boxes and class probabilities simultaneously from the full image — no region proposal step, no separate classifier.
- **Grid-based prediction:** The input image (448×448) is divided into a 7×7 grid. Each grid cell predicts B bounding boxes (the paper uses B=2 in the full model, B=1 in the simplified version described here) with confidence scores, plus C conditional class probabilities (20 classes for PASCAL VOC).
- **Output tensor:** The final layer produces a 7×7×(5*B + C) tensor. With B=2 and C=20, that's 7×7×30.
- **Loss function:** Multi-part sum-squared error with λ_coord = 5 in the paper (up-weighting localization; the notebook uses 4) and λ_noobj = 0.5 (down-weighting boxes without objects). Square-root parameterization for width/height.
- **Architecture:** 24 convolutional layers (inspired by GoogLeNet, using 1×1 reduction + 3×3 conv layers) followed by 2 fully connected layers. Max-pooling layers (plus one stride-2 convolution) handle downsampling.
- **Training:** Pretrained on ImageNet (first 20 conv layers, ~1 week), then fine-tuned for detection on PASCAL VOC 2007/2012. ~135 epochs, batch size 64, momentum 0.9, dropout 0.5.
- **Speed:** 45 FPS (base), 155 FPS (Fast YOLO with 9 conv layers).

## Key Method Details

- **Objectness score:** Each predicted bounding box carries a confidence = Pr(Object) × IOU(pred, truth). This decouples "is there something?" from "what is it?" and handles the class imbalance from mostly-empty grid cells.
- **Conditional class probabilities:** Pr(Class_i \| Object) — only updated at cells containing an object. Final detection score = Pr(Class_i \| Object) × Pr(Object) × IOU.
- **Non-maximal suppression:** Post-processing to remove duplicate detections of the same object; adds 2-3% mAP.
- **Leaky ReLU** activation (0.1 slope for negative inputs) in every layer except the last, which is linear in the paper (the notebook applies a sigmoid to bound outputs to [0, 1]).

### Simplifications in This Notebook

- Smaller network: 8 conv layers (stride-2 convs instead of max-pooling) + 2 fully connected layers, about 60 million parameters (mostly in the first fully connected layer)
- Input size 224×224 instead of 448×448
- Synthetic toy dataset (red circles, green rectangles, blue triangles on a black background) instead of PASCAL VOC: 200 training and 50 test images
- B=1 bounding box per cell (original paper uses B=2)
- 3 classes instead of 20
- Very short training: 3 epochs with Adam (lr 1e-3) instead of ~135 epochs with SGD
- No ImageNet pretraining; trained from scratch on the synthetic dataset

### What the Notebook Result Shows

The notebook is a small, from-scratch learning demo. It shows how the YOLO pieces fit together: the 7×7 grid targets, the multi-part loss, the training loop and the box decoding step. It does **not** reach useful detection accuracy yet. After 3 epochs the training loss goes from 14.33 to 12.97 (it goes up a little in epoch 2), and on the 50 test images the evaluation prints **Mean IoU 0.0000** and **Detection Accuracy (IoU > 0.5) 0.0000**. That means no predicted box overlapped a real shape well enough to count. The loss plot and the detection pictures are saved to files but not shown inside the notebook, because the plotting cell switches matplotlib to the non-display `Agg` backend. Treat the notebook as a clear map of how YOLO works, not as a working detector. Training much longer, on more images, is the natural next step.

## Influence

YOLO changed object detection by showing that a single neural network could detect many kinds of objects at real-time speed. It won the [OpenCV People's Choice Award at CVPR 2016](https://pjreddie.com/publications/yolo/). It spawned an entire family of models (YOLOv2/YOLO9000 and YOLOv3 by the original authors, then YOLOv4, YOLOv5, YOLOv8, YOLOv10, YOLO11 and others by different teams) that remain among the most widely used detectors in both research and industry. The original YOLO paper has been cited about 53,000 times ([OpenAlex](https://openalex.org/works/W1483870316), checked October 2026) and is one of the most influential computer vision papers of the 2010s. Its framing of detection as a regression problem influenced single-stage detectors like SSD and RetinaNet.

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/research-papers/blob/main/032_you-only-look-once-yolo/solution.ipynb) | Small from-scratch PyTorch demo of YOLO's grid, loss and training loop on a toy shapes dataset (educational; it does not reach useful detection accuracy) |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, blog posts, interview Q&A, citations, and common misconceptions about the paper |
