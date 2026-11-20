---
title: "Faster R-CNN: Talk & Q&A"
parent: "031 Faster R-CNN"
nav_order: 2
---
# TALK.md — Faster R-CNN

## Press & Blog Coverage

1. [**Official MATLAB code**](https://github.com/ShaoqingRen/faster_rcnn), by Shaoqing Ren. The authors' original implementation; its README describes Faster R-CNN as a Region Proposal Network plus an object detection network that share convolutional layers.

2. [**py-faster-rcnn**](https://github.com/rbgirshick/py-faster-rcnn), by Ross Girshick. The Python/Caffe implementation; its README says Faster R-CNN was first described in an arXiv tech report and then published in NIPS 2015.

3. [**torchvision Faster R-CNN models**](https://pytorch.org/vision/stable/models/faster_rcnn.html), PyTorch documentation. Ready-made Faster R-CNN models in torchvision; the page refers to this paper.

## Q&A (answers from the paper)

**Q1: What problem were the authors trying to fix?**
A: Region proposals had become the bottleneck. Fast R-CNN ran in near real time if you ignored proposals, but Selective Search took about 2 seconds per image on a CPU, and even EdgeBoxes took 0.2 seconds (Section 1). By sharing convolutions with the detector, an RPN adds only about 10 ms per image.

**Q2: What are anchors?**
A: Reference boxes at every sliding-window position on the feature map. The paper uses k = 9 anchors per position: 3 scales (box areas of 128², 256² and 512² pixels) times 3 aspect ratios (1:1, 1:2, 2:1). The RPN predicts an objectness score and box offsets relative to each anchor, so it handles many object sizes from a single-scale feature map without an image pyramid (Sections 3.1.1 and 3.3).

**Q3: How is the RPN trained?**
A: An anchor is positive if it has the highest IoU with some ground-truth box or an IoU above 0.7 with any box, and negative if its IoU is below 0.3 for all boxes; the rest are ignored. Each mini-batch samples 256 anchors from one image with up to a 1:1 ratio of positives to negatives. The loss is log loss for objectness plus a box-regression loss for positives (Section 3.1.2).

**Q4: How are the RPN and Fast R-CNN trained to share features?**
A: The paper describes three options. Its experiments use 4-step alternating training: train the RPN, train Fast R-CNN on its proposals, then fine-tune each with the shared layers fixed. Approximate joint training merges both into one network and treats the proposals as fixed, ignoring the gradient through the box coordinates; the authors found it produces close results while cutting training time by about 25–50% (Section 3.2).

**Q5: How accurate and fast was it?**
A: With VGG-16 and 300 proposals per image, it reached 73.2% mAP on PASCAL VOC 2007 test (trained on VOC 2007+2012) and 70.4% on VOC 2012 test (Tables 3 and 4). The whole system runs at 5 fps on a GPU (198 ms per image), against 0.5 fps for Selective Search + Fast R-CNN (Table 5).

**Q6: Are 300 learned proposals really enough?**
A: Yes, in the paper's tests. With the ZF network, Fast R-CNN reached 59.9% mAP using up to 300 RPN proposals, against 58.7% with about 2000 Selective Search proposals and 58.6% with EdgeBoxes (Section 4.1, Table 2).

## Common Misconceptions

1. **"The paper trains everything in one end-to-end pass."** Its experiments use 4-step alternating training. Approximate joint training is described as an alternative that gives close results faster (Section 3.2).

2. **"Faster R-CNN replaced Fast R-CNN."** It keeps Fast R-CNN as its second module. What it replaces is the external proposal method, such as Selective Search, with the learned RPN (Section 3).

3. **"Anchors and SSD's default boxes are the same thing."** In Faster R-CNN, anchors feed a separate proposal stage on a single feature map. [SSD](https://arxiv.org/abs/1512.02325) later used default boxes on several feature maps to predict the final detections directly.

4. **"RoI pooling is the same as RoI Align."** Faster R-CNN uses Fast R-CNN's RoI pooling. RoI Align was introduced later in [Mask R-CNN](https://arxiv.org/abs/1703.06870) to avoid RoI pooling's rounding to the feature-map grid.

5. **"A from-scratch toy version shows the paper's numbers."** [This repo's notebook](solution.ipynb) trains a tiny model on synthetic shapes. Its mAP measures that toy task only, not PASCAL VOC; see its Summary.

## Real Citations

1. Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.* NIPS 2015 (now NeurIPS); extended version in IEEE TPAMI. [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)

2. Girshick, R. (2015). *Fast R-CNN.* ICCV 2015. [arXiv:1504.08083](https://arxiv.org/abs/1504.08083)

3. Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). *You Only Look Once: Unified, Real-Time Object Detection.* CVPR 2016. [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)

4. Liu, W., et al. (2016). *SSD: Single Shot MultiBox Detector.* ECCV 2016. [arXiv:1512.02325](https://arxiv.org/abs/1512.02325)

5. He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). *Mask R-CNN.* ICCV 2017. [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)

6. Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). *Feature Pyramid Networks for Object Detection.* CVPR 2017. [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)

7. Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). *Focal Loss for Dense Object Detection.* ICCV 2017. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)

**Citation count:** about 80,700 ([OpenAlex](https://openalex.org/W639708223), 80,700 when checked on 20 November 2026). OpenAlex keeps the arXiv paper and the TPAMI version in one record.
