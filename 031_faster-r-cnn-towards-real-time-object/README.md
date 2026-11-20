---
title: "031 Faster R-CNN"
nav_order: 32
permalink: "/031_faster-r-cnn-towards-real-time-object/"
---
# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
![Faster R-CNN banner](../assets/banners/031.jpg)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/faster-r-cnn-towards-real-time-object-detection)

- **Paper:** [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
- **Authors:** Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun
- **Lab, year:** Microsoft Research, 2015 (NIPS 2015)
- **Video:** YouTube Short coming on 21 Nov 2026 on [Backprop Daily](https://www.youtube.com/@BackpropDaily)

## Summary

An object detector has to answer two questions: where might there be objects, and what are they? Before this paper, the "where" step used a separate, slow algorithm such as Selective Search, which took about 2 seconds per image on a CPU.

Faster R-CNN replaces that step with a small neural network, the Region Proposal Network (RPN). The RPN reuses the same convolutional features as the detector, so proposing regions costs only about 10 ms per image.

With a VGG-16 network the whole system ran at 5 frames per second on a GPU and reached 73.2% mean average precision (mAP, the standard detection accuracy score) on PASCAL VOC 2007, using only 300 proposals per image.

## What Problem Does It Solve

Imagine searching a photo for your friends. A slow helper first draws 2,000 rough boxes around anything that might be a person, and only then do you look inside each box. Most of the time goes into the helper's guessing.

Fast R-CNN had made the "look inside each box" part quick, so the guessing step became the bottleneck. Faster R-CNN teaches the detector to make its own guesses, using what it already sees. The guesses become nearly free, and they are also better: 300 learned proposals worked as well as 2,000 from Selective Search.

## Core Idea

- **One shared feature map:** the image passes once through a convolutional network. Both the RPN and the detector read from the same features.
- **Region Proposal Network:** a small network slides a 3×3 window over the feature map. At each position it scores k reference boxes for "object or not" and predicts how to adjust each box.
- **Anchors:** the k = 9 reference boxes per position come in 3 sizes and 3 shapes (aspect ratios 1:1, 1:2, 2:1), so one feature map can handle small and large objects.
- **Fast R-CNN detector:** RoI pooling cuts a fixed-size piece of the feature map for each proposal, and a small head classifies it and refines its box.

## Key Method Details

- **Anchor labels:** an anchor is positive if it has the highest IoU (box overlap) with an object or an IoU above 0.7, and negative below 0.3. Each training image samples 256 anchors, up to half positive (Section 3.1.2).
- **Anchor sizes:** box areas of 128², 256² and 512² pixels, on images rescaled so the shorter side is 600 pixels (Section 3.3).
- **Training the two parts together:** the paper's experiments use 4-step alternating training. Approximate joint training, which treats proposals as fixed inputs, gave close results and trained about 25–50% faster (Section 3.2).
- **Results:** 73.2% mAP on VOC 2007 test and 70.4% on VOC 2012 test with VGG-16 (Tables 3 and 4); 5 fps in total, against 0.5 fps for Selective Search + Fast R-CNN (Table 5).

### Simplifications in This Notebook

The notebook builds a tiny Faster R-CNN in PyTorch: a 3-layer CNN backbone that turns a 64×64 grayscale image into an 8×8 feature map, an RPN with 9 anchors per position (sizes 8, 16 and 24 pixels), RoI pooling to 5×5, and a two-layer detection head. It trains on synthetic images with 1–3 white squares and circles (1,000 for training, 200 for testing) for 20 epochs with Adam, using approximate joint training. RoIs for the detection head are sampled from the proposals plus the ground-truth boxes, at most 25% foreground.

### What the Notebook Result Shows

**The toy detector works.** On 200 test images (398 objects), AP@0.5 was **0.945** for squares and **0.885** for circles, so mAP@0.5 = **0.915**. The RPN's top-30 proposals covered 396 of the 398 objects (recall 0.995), and the total training loss fell from 0.7855 to 0.2867.

In the example images, most objects get a red box of the right class, but two overlapping circles share one box, and boxes are often a few pixels too large because the feature map is coarse. This is an easy synthetic task, so the number is not comparable to the paper's VOC results. An earlier version of this notebook produced no detections at all.

## Influence

OpenAlex lists about 80,700 citations ([80,700 when checked on 20 November 2026](https://openalex.org/W639708223); the record joins the arXiv paper and its TPAMI version). In the abstract's words, Faster R-CNN and the RPN were the foundations of the 1st-place entries in several tracks of the ILSVRC and COCO 2015 competitions. Later work built directly on it, including [Feature Pyramid Networks](https://arxiv.org/abs/1612.03144) and [Mask R-CNN](https://arxiv.org/abs/1703.06870), and it ships as a ready-made model in [torchvision](https://pytorch.org/vision/stable/models/faster_rcnn.html).

---

## Resources

| File | Description |
|------|-------------|
| [README.md](README.md) | This overview — a 5-minute read covering the paper's summary, core idea, key method details, influence, and a plain-English explanation of the problem it solves |
| [solution.ipynb](https://nbviewer.org/github/nadeemcite/research-papers/blob/main/031_faster-r-cnn-towards-real-time-object/solution.ipynb) | From-scratch PyTorch Faster R-CNN (backbone, RPN with anchors, RoI pooling, detection head) trained and evaluated with mAP on synthetic shapes |
| [CODE_ARCHITECTURE.md](CODE_ARCHITECTURE.md) | Section-by-section breakdown of the notebook — what each cell does, data flow, tensor shapes, and key design decisions |
| [TALK.md](TALK.md) | Press coverage, Q&A (answers from the paper), citations, and common misconceptions about the paper |
