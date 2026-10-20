---
title: "Wide ResNet: Talk & Q&A"
parent: "039 Wide ResNet"
nav_order: 2
---
# Wide Residual Networks — Talks, Coverage, and Discussion

## Press / Blog Coverage

- **Official code repository:** [github.com/szagoruyko/wide-residual-networks](https://github.com/szagoruyko/wide-residual-networks). The paper's own link to its Torch code and trained models.
- **BMVC 2016 paper page:** [bmva-archive.org.uk/bmvc/2016/papers/paper087](https://www.bmva-archive.org.uk/bmvc/2016/papers/paper087/index.html). The peer-reviewed version.
- **arXiv:** [arxiv.org/abs/1605.07146](https://arxiv.org/abs/1605.07146). The first release, in May 2016.

## Q&A (answers from the paper)

**Q1: Why look at width instead of depth for residual networks?**

A (from the paper, Introduction): Each fraction of a percent of extra accuracy from a very deep ResNet cost nearly a doubling of the number of layers, which makes very deep networks slow to train. The authors also describe *diminishing feature reuse*: since gradients can flow through the identity shortcuts, many residual blocks may learn very little. Their answer was to study how wide a residual network should be.

**Q2: What is the headline result?**

A (from the paper, Introduction and results): A wide 16-layer network has about the same accuracy as a 1000-layer thin network with a comparable number of parameters, and is several times faster to train. WRN-40-4 is 8 times faster to train than ResNet-1001 at about the same accuracy, and WRN-28-10 is 1.6 times faster and 0.92 points better on CIFAR-10. The authors read this as a sign that the power of deep residual networks is in the residual blocks, and that depth is supplementary.

**Q3: Why does dropout work inside residual blocks when earlier work found it harmful?**

A (from the paper, Introduction and the "Dropout in residual blocks" section): The earlier study (He et al., *Identity Mappings in Deep Residual Networks*) inserted dropout in the identity part of the block and saw negative effects. The authors instead insert dropout between the two convolutions, and report consistent gains. Dropout probability was chosen by cross-validation (0.3 on CIFAR, 0.4 on SVHN), and they did not have to train for more epochs.

**Q4: How much does dropout help?**

A (from the paper, Table 6): With WRN-28-10 it lowers CIFAR-10 error from 4.00% to 3.89%. The effect is larger on SVHN, where WRN-16-4 goes from 1.85% to 1.64% and WRN-52-1 from 2.08% to 1.70%. There is one exception: WRN-16-4 on CIFAR-10 gets slightly worse (5.02% to 5.24%), which the authors speculate is due to its relatively small number of parameters. The paper says dropout helps both thin and wide networks.

**Q5: What is the practical takeaway?**

A (from the paper, results and benchmarks): The authors found that widening is a more effective way to improve a residual network than adding depth, and that wide networks run faster on a GPU because wide layers parallelise better. Their best CIFAR-10 result is WRN-28-10 and their best SVHN result is WRN-16-8. They also found that depth is not free: with k = 8 or 10, going from 16 to 28 layers helped, but at 40 layers accuracy decreased (WRN-40-8 lost to WRN-22-8).

## Common Misconceptions

1. **"WRNs replace ResNets."** No. WRNs are a variant of ResNets. They use the same residual structure (the pre-activation form from He et al.) and only increase the number of channels. The shortcut connections are the same.

2. **"Wider is always better than deeper."** Not always. The paper shows widening is a more effective use of parameters than extra depth in the ranges it tests, but it also found that accuracy stopped improving at some depth for a given width (WRN-40-8 lost to WRN-22-8). In this repo's 20-epoch CIFAR-10 run, WRN-16-2 beat WRN-40-1 and WRN-52-1 with a similar parameter count, which fits the paper's argument but is one seed at a short schedule.

3. **"Dropout in ResNets always helps."** Not always. The paper's dropout sits between the convolutions, not on the shortcut, and it helped most on SVHN. It also found a small loss for WRN-16-4 on CIFAR-10 (Table 6). It reports gains for both thin and wide networks, so it is not only for wide ones.

4. **"WRN-16-10 is the best configuration."** There is no WRN-16-10 in the paper's tables. The best CIFAR-10 result is WRN-28-10 (3.89% with dropout), and the best SVHN result is WRN-16-8 (1.54% with dropout). The 16-layer result is notable because a wide 16-layer network matches a 1000-layer thin one, not because it is the best WRN.

## Real Citations

- Zagoruyko, S. & Komodakis, N. (2016). "Wide Residual Networks." British Machine Vision Conference (BMVC) 2016. arXiv:1605.07146.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Identity Mappings in Deep Residual Networks." ECCV 2016. arXiv:1603.05027. The architecture WRNs build on, and the earlier dropout study the paper contrasts with.
- Huang, G., Sun, Y., Liu, Z., Sedra, D., & Weinberger, K. Q. (2016). "Deep Networks with Stochastic Depth." ECCV 2016. arXiv:1603.09382. Randomly disables residual blocks during training; the paper cites it as supporting the diminishing feature reuse idea.
- Veit, A., Wilber, M., & Belongie, S. (2016). "Residual Networks Behave Like Ensembles of Relatively Shallow Networks." NeurIPS 2016. arXiv:1605.06431. The notebook's lesion study follows the spirit of this paper.
- Tan, M. & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." ICML 2019. arXiv:1905.11946. Scales width, depth and resolution together.
