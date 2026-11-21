---
title: "SSD: Talk & Q&A"
parent: "033 SSD"
nav_order: 2
---
# TALK.md — SSD: Single Shot MultiBox Detector

## Press & Blog Coverage

1. [**weiliu89/caffe, branch `ssd`**](https://github.com/weiliu89/caffe/tree/ssd), by Wei Liu. The authors' reference implementation of SSD as a branch of Caffe.

2. [**"Supercharge your Computer Vision models with the TensorFlow Object Detection API"**](https://research.google/blog/supercharge-your-computer-vision-models-with-the-tensorflow-object-detection-api/), Google Research blog, 15 June 2017. Announced the TF Object Detection API; its launch models include SSD with MobileNets, which the post says are light enough to run in real time on mobile devices.

3. [**"Speed/accuracy trade-offs for modern convolutional object detectors"**](https://arxiv.org/abs/1611.10012), Huang et al., CVPR 2017. Google's large comparison of Faster R-CNN, R-FCN and SSD across backbones.

4. [**"Everything You Need To Know About Torchvision's SSD Implementation"**](https://pytorch.org/blog/torchvision-ssd-implementation/), PyTorch blog. Walks through how torchvision implemented SSD.

5. [**MobileNetV2: Inverted Residuals and Linear Bottlenecks**](https://arxiv.org/abs/1801.04381), Sandler et al., CVPR 2018. Introduced SSDLite, a mobile-friendly variant of SSD's prediction layers.

## Q&A (answers from the paper)

**Q1: What is the main difference between SSD and YOLO?**
A: SSD predicts with small 3×3 convolutional filters on several feature maps of different resolutions, each with default boxes of several scales and aspect ratios (Section 2.1, Fig. 2). On VOC2007 test, SSD300 reached 74.3% mAP at 59 FPS, against 63.4% at 45 FPS for YOLO (Section 1).

**Q2: Where does SSD's speed over Faster R-CNN come from?**
A: The authors say the fundamental improvement in speed comes from eliminating bounding-box proposals and the pixel or feature resampling stage that follows them (Section 1). Faster R-CNN, the fastest high-accuracy detector at the time, ran at 7 FPS with 73.2% mAP.

**Q3: Why is hard negative mining needed?**
A: After matching, most default boxes are negatives, which creates a big imbalance. SSD sorts the negatives by their confidence loss and keeps the top ones so that negatives:positives is at most 3:1; the authors found this leads to faster optimization and more stable training (Section 2.2).

**Q4: Which design choices mattered most in the ablation?**
A: Data augmentation was crucial: the sampling strategy improved mAP by 8.8% (Section 3.2, Table 2). Removing the default boxes with aspect ratios 1/3 and 3 cost 0.6%, and removing 1/2 and 2 as well cost another 2.1%. Using the full VGG16 instead of the atrous version gave about the same result but was about 20% slower. Predicting from fewer output layers lowered mAP steadily from 74.3 to 62.4 (Table 3).

**Q5: How does SSD do on small objects?**
A: Much worse than on large objects; the authors note that small objects may not have any information at the very top layers (Section 3.1). A "zoom out" augmentation, which places the image on a canvas up to 16 times larger before cropping, gave a consistent 2–3% mAP increase across datasets and helped small objects in particular (Section 3.6, Table 6).

## Common Misconceptions

1. **"SSD has no anchors; that's what makes it single-shot."** SSD is built on default boxes: SSD300 uses 8,732 of them (Fig. 2). "Single shot" means there is no separate proposal and resampling stage (Section 1).

2. **"SSD's speed comes from a lighter backbone."** The original SSD uses VGG-16, like Faster R-CNN (Section 2.1); the speed-up comes from removing the proposal and resampling stage (Section 1). Lighter MobileNet-based versions came later, for example in the TF Object Detection API.

3. **"Each feature map's default boxes must match its receptive field."** The paper says the default boxes do not need to correspond to each layer's actual receptive field; the tiling is designed so that specific maps learn to respond to particular object scales (Section 2.2).

4. **"SSD is a feature pyramid network."** SSD predicts from backbone feature maps of different resolutions without a top-down path. [FPN](https://arxiv.org/abs/1612.03144) later added top-down connections.

5. **"A toy reproduction measures SSD's real accuracy."** [This repo's notebook](solution.ipynb) trains a small model on synthetic shapes; its mAP only describes that easy task. See its Summary.

## Real Citations

1. Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.-Y., & Berg, A. C. (2016). *SSD: Single Shot MultiBox Detector.* ECCV 2016. [arXiv:1512.02325](https://arxiv.org/abs/1512.02325)

2. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection.* CVPR 2016. [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)

3. Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.* NIPS 2015 (now NeurIPS). [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)

4. Erhan, D., Szegedy, C., Toshev, A., & Anguelov, D. (2014). *Scalable Object Detection using Deep Neural Networks* (MultiBox). CVPR 2014. [arXiv:1312.2249](https://arxiv.org/abs/1312.2249)

5. Fu, C.-Y., Liu, W., Ranga, A., Tyagi, A., & Berg, A. C. (2017). *DSSD: Deconvolutional Single Shot Detector.* [arXiv:1701.06659](https://arxiv.org/abs/1701.06659)

6. Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). *Feature Pyramid Networks for Object Detection.* CVPR 2017. [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)

7. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal Loss for Dense Object Detection* (RetinaNet). ICCV 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)

8. Huang, J., et al. (2017). *Speed/accuracy trade-offs for modern convolutional object detectors.* CVPR 2017. [arXiv:1611.10012](https://arxiv.org/abs/1611.10012)

9. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018. [arXiv:1801.04381](https://arxiv.org/abs/1801.04381)

**Citation count:** about 33,200 for the ECCV version ([OpenAlex](https://openalex.org/W2193145675), 33,222 when checked on 21 November 2026); OpenAlex also has a separate record with 31,625.
