---
title: "VGGNet: Talk & Q&A"
parent: "004 VGGNet"
nav_order: 2
---
# TALK.md — VGGNet: Context, Discussion, and Q&A

**Paper:** [Very Deep Convolutional Networks for Large-Scale Image Recognition (arXiv:1409.1556)](https://arxiv.org/abs/1409.1556)

## Press coverage, blog posts, and official announcements

- [Official VGG project page](https://robots.ox.ac.uk/~vgg/research/very_deep/) (Visual Geometry Group, University of Oxford) — the paper, the trained models and the ILSVRC 2014 submission details.
- [arXiv page](https://arxiv.org/abs/1409.1556) — the paper (published at ICLR 2015).
- [ILSVRC 2014 results](https://image-net.org/challenges/LSVRC/2014/results) — the official results of the ImageNet 2014 challenge.
- [The New York Times, Bits blog: "Computer Eyesight Gets a Lot More Accurate"](https://web.archive.org/web/20150316003114/http://bits.blogs.nytimes.com/2014/08/18/computer-eyesight-gets-a-lot-more-accurate/) (18 August 2014, archived copy) — reports the ImageNet 2014 results; it names Oxford University among the winners and says error rates were cut in half.

## Q&A (answers from the paper)

**Q1. Why did VGGNet use 3×3 convolutions instead of larger filters?**  
A (from the paper, Section 2.3): A stack of three 3×3 conv layers has the same 7×7 receptive field as one 7×7 layer. With C channels in and out, the stack has 3·(3²·C²) = 27C² weights, while one 7×7 layer has 49C² (81% more). The stack also has three ReLUs instead of one, which makes the decision function more discriminative.

**Q2. What do the numbers in VGG-16 and VGG-19 mean?**  
A (from the paper, Section 2.2 and Table 1): They count **weight layers**. Configuration D (VGG-16) has 13 conv + 3 fully connected layers; configuration E (VGG-19) has 16 conv + 3 fully connected layers.

**Q3. How did VGGNet do at ILSVRC 2014?**  
A (from the paper, Abstract and Section 4.5): The team got **1st place in localisation** and **2nd place in classification**. The submission (an ensemble of 7 nets) had 7.3% top-5 test error. After the challenge, an ensemble of just 2 nets reached 6.8%, and a single net reached 7.0%.

**Q4. Did VGGNet win ImageNet 2014 classification?**  
A (from the paper, Section 4.5 and Table 7): No. **GoogLeNet** won classification with 6.7% top-5 error (7 nets). But as a single network, VGG was better: 7.0% vs 7.9% for one GoogLeNet.

**Q5. How long did training take?**  
A (from the paper, Sections 3.1 and 3.3): Training used batch size 256, momentum 0.9, weight decay 5·10⁻⁴ and dropout 0.5 on the first two FC layers. The learning rate started at 10⁻² and was cut by 10 three times. Training stopped after 370K iterations (74 epochs). On a system with four NVIDIA Titan Black GPUs, one net took 2–3 weeks to train.

## Common misconceptions or mistakes

- **"VGG-16 has 16 convolutional layers."** Wrong. It has 13 convolutional + 3 fully connected weight layers; "16" is the total.
- **"VGGNet won ImageNet 2014 classification."** No, GoogLeNet won classification. VGG won localization and placed second in classification.
- **"VGG-19 is much better than VGG-16."** In the paper the gap is tiny: 8.1% vs 8.0% top-5 validation error at a single test scale (Table 3), and both 7.5% at multiple test scales (Table 4). VGG-19 also has more parameters (144M vs 138M, Table 2).
- **"VGGNet originally used batch normalization."** The paper predates BatchNorm; BN was added later by the community to stabilize training.
- **"Bigger filters are needed for bigger receptive fields."** VGGNet showed you can stack small filters to get the same receptive field with fewer parameters and more depth.

## Sources

- Simonyan, K., & Zisserman, A. (2015). [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556). ICLR 2015.
- [Official VGG page](https://robots.ox.ac.uk/~vgg/research/very_deep/)
- [ILSVRC 2014 results](https://image-net.org/challenges/LSVRC/2014/results)
- [NYT Bits blog (archived)](https://web.archive.org/web/20150316003114/http://bits.blogs.nytimes.com/2014/08/18/computer-eyesight-gets-a-lot-more-accurate/)
