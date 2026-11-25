---
title: "Mask R-CNN: Talk & Q&A"
parent: "034 Mask R-CNN"
nav_order: 2
---
# TALK.md — Mask R-CNN

## Press & Blog Coverage

1. [**Detectron**](https://github.com/facebookresearch/Detectron), Facebook AI Research on GitHub. FAIR's object detection software, which its README says implements algorithms including Mask R-CNN and was used for the Mask R-CNN research project.

2. [**Detectron2**](https://github.com/facebookresearch/detectron2), Facebook AI Research on GitHub. FAIR's next-generation detection library, the successor to Detectron and maskrcnn-benchmark.

3. [**matterport/Mask_RCNN**](https://github.com/matterport/Mask_RCNN), GitHub. A widely used implementation of Mask R-CNN in Python 3, Keras and TensorFlow, with an FPN and ResNet-101 backbone.

4. [**torchvision Mask R-CNN models**](https://docs.pytorch.org/vision/stable/models/mask_rcnn.html), PyTorch documentation. Ready-made Mask R-CNN models in torchvision, including a ResNet-50-FPN model "from the Mask R-CNN paper".

## Q&A (answers from the paper)

**Q1: What does Mask R-CNN add to Faster R-CNN?**
A: A third output for each candidate object: alongside the class label and the box offset, a branch predicts an object mask, in parallel with the existing branch for box recognition (Section 3). The authors describe it as simple to train, adding only a small overhead and running at 5 fps (Abstract).

**Q2: Why is RoIAlign needed?**
A: RoIPool rounds the RoI and its bins to the feature-map grid, which misaligns the extracted features with the RoI; this barely matters for classification but hurts pixel-accurate masks. RoIAlign removes the rounding and uses bilinear interpolation at regularly sampled points (Section 3). With ResNet-50-C4 it improved mask AP by about 3 points over RoIPool, from 26.9 to 30.3, with much of the gain at high IoU (Section 4.2, Table 2c).

**Q3: Why per-class sigmoid masks instead of a softmax?**
A: The mask branch predicts one binary mask per class with a per-pixel sigmoid and binary loss, so classes do not compete; the box branch decides the class. Replacing this with a per-pixel softmax and multinomial loss dropped mask AP from 30.3 to 24.8 (Section 4.2, Table 2b).

**Q4: How well does it work?**
A: On COCO test-dev, mask AP was 35.7 with ResNet-101-FPN and 37.1 with ResNeXt-101-FPN (Section 4.1, Table 1), and the models run at about 200 ms per frame on a GPU (Section 1).

**Q5: Does it work beyond segmentation?**
A: Yes. By modelling each keypoint as a one-hot mask, the same framework estimates human poses, and the authors report it surpassing the winner of the 2016 COCO keypoint competition (Sections 1 and 5).

## Common Misconceptions

1. **"Mask R-CNN is just Faster R-CNN plus a mask branch."** The design choices matter: in the ablations, per-pixel softmax masks gave 24.8 mask AP vs 30.3 for sigmoid masks (Table 2b), and RoIPool gave 26.9 vs 30.3 for RoIAlign (Table 2c), both with ResNet-50-C4.

2. **"Mask R-CNN does semantic segmentation."** It does instance segmentation: it detects each object and predicts a separate mask for each one (Section 1).

3. **"RoIAlign is a minor detail."** The paper attributes the gap between RoIPool and RoIAlign to alignment: "Using proper alignment is the only factor that contributes to the large gap between RoI layers" (Table 2c caption). With larger stride-32 features the gain grows to 7.3 mask AP (Table 2d).

4. **"A toy reproduction reproduces COCO numbers."** [This repo's notebook](solution.ipynb) trains a small model on synthetic shapes; its mask mAP only describes that task. See its Summary.

## Real Citations

1. He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). *Mask R-CNN.* ICCV 2017. [arXiv:1703.06870](https://arxiv.org/abs/1703.06870)

2. Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.* NIPS 2015 (now NeurIPS). [arXiv:1506.01497](https://arxiv.org/abs/1506.01497)

3. Girshick, R. (2015). *Fast R-CNN.* ICCV 2015. [arXiv:1504.08083](https://arxiv.org/abs/1504.08083)

4. Long, J., Shelhamer, E., & Darrell, T. (2015). *Fully Convolutional Networks for Semantic Segmentation.* CVPR 2015. [arXiv:1411.4038](https://arxiv.org/abs/1411.4038)

5. Lin, T.-Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). *Feature Pyramid Networks for Object Detection.* CVPR 2017. [arXiv:1612.03144](https://arxiv.org/abs/1612.03144)

6. Li, Y., Qi, H., Dai, J., Ji, X., & Wei, Y. (2017). *Fully Convolutional Instance-aware Semantic Segmentation (FCIS).* CVPR 2017. [arXiv:1611.07709](https://arxiv.org/abs/1611.07709)

7. Wang, X., Zhang, R., Kong, T., Li, L., & Shen, C. (2020). *SOLOv2: Dynamic and Fast Instance Segmentation.* NeurIPS 2020. [arXiv:2003.10152](https://arxiv.org/abs/2003.10152)

**Citation count:** about 34,800 for the ICCV version ([OpenAlex](https://openalex.org/W2963150697), 34,767 when checked on 25 November 2026); OpenAlex also lists the arXiv version separately (7,154).
