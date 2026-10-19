---
title: "GoogLeNet: Talk & Q&A"
parent: "005 GoogLeNet"
nav_order: 2
---
# TALK.md — GoogLeNet / Inception: Context, Discussion, and Q&A

**Paper:** [Going Deeper with Convolutions (arXiv:1409.4842)](https://arxiv.org/abs/1409.4842)

## Press, blog, and official coverage

1. [Google Research publication page: *Going Deeper with Convolutions*](https://research.google/pubs/going-deeper-with-convolutions/) — lists the paper, its authors and the abstract.

2. [Wikipedia: Inception (deep learning architecture)](https://en.wikipedia.org/wiki/Inception_%28deep_learning_architecture%29) — says: "In 2014, a team at Google developed the GoogLeNet architecture, an instance of which won the ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC14)." It also explains the names: LeNet, and the "we need to go deeper" meme.

3. [Dive into Deep Learning, Section 8.4: Multi-Branch Networks (GoogLeNet)](https://d2l.ai/chapter_convolutional-modern/googlenet.html) — a widely used open textbook chapter that explains the Inception block, its four parallel branches and the 1×1 reductions.

4. [TensorFlow-Slim pre-trained models (tensorflow/models on GitHub)](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models) — lists released Inception V1 (GoogLeNet) checkpoints.

5. [arXiv page](https://arxiv.org/abs/1409.4842) — the paper, abstract and PDF.

## Q&A (answers from the paper)

**Q1: What problem was GoogLeNet trying to solve?**  
A (from the paper, Section 3): Making a network bigger (deeper and wider) usually improves accuracy, but it has two costs: more parameters, which makes overfitting more likely, and much more computation. If two conv layers are chained, any uniform increase in their number of filters gives a quadratic increase in computation. The Inception module tries to grow depth and width while keeping the computational budget roughly constant.

**Q2: Why does the Inception module have four parallel branches?**  
A (from the paper, Section 4): The idea is to cover the useful structure at different scales. The 1×1, 3×3 and 5×5 sizes were chosen "based more on convenience rather than necessity", and a pooling branch was added because pooling had been essential in other strong networks. The branch outputs are concatenated along the channel axis.

**Q3: What is the role of the 1×1 convolutions?**  
A (from the paper, Section 4): They reduce the number of channels before the expensive 3×3 and 5×5 convolutions. Without them, even a small number of 5×5 filters on top of many input channels is very costly, and the pooling branch makes the number of output channels grow from stage to stage. The 1×1 layers also include a ReLU, so they add non-linearity too.

**Q4: Why is it called GoogLeNet and "Inception"?**  
A (from the paper, Sections 1 and 5): GoogLeNet was the team name in ILSVRC14, an homage to Yann LeCun's LeNet-5. The name "Inception" comes from the Network-in-Network paper by Lin et al., together with the "we need to go deeper" internet meme.

**Q5: Did the 22-layer network really have fewer parameters than AlexNet?**  
A (from the paper, Section 1): Yes. The GoogLeNet submission used **12× fewer parameters** than the winning network of Krizhevsky et al. from two years earlier, while being much more accurate. Section 5 adds that using average pooling instead of large fully connected layers at the end improved top-1 accuracy by about 0.6%.

## Common misconceptions

- **"Inception is just many conv layers"** — It is specifically a *multi-scale, multi-branch* design; the parallel branches and 1×1 bottlenecks are the point, not the depth.
- **"GoogLeNet is the architecture and Inception is something else"** — GoogLeNet is the ILSVRC14 submission, a particular instantiation of the Inception architecture family.
- **"1×1 convolutions do nothing on spatial data"** — They perform cross-channel mixing and dimension reduction; in Inception they are essential for efficiency.
- **"Auxiliary classifiers are part of inference"** — They are training-only side heads; their losses are added with a weight of 0.3, and at inference time they are discarded (Section 5).
- **"You need 224×224 images to use Inception"** — The *module* works at any resolution; the original stem was sized for ImageNet, but the idea has been applied to CIFAR-10 and many other scales.

## Real citations

- Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., & Rabinovich, A. (2015). Going deeper with convolutions. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 1–9. DOI: 10.1109/CVPR.2015.7298594 — https://arxiv.org/abs/1409.4842
- Lin, M., Chen, Q., & Yan, S. (2014). Network In Network. *ICLR*. https://arxiv.org/abs/1312.4400
- Arora, S., Bhaskara, A., Ge, R., & Ma, T. (2014). Provable Bounds for Learning Some Deep Representations. *ICML*. https://arxiv.org/abs/1310.6343
- Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016). Rethinking the Inception Architecture for Computer Vision. *CVPR*. https://arxiv.org/abs/1512.00567
- Zhang, A., Lipton, Z. C., Li, M., & Smola, A. J. (2024). *Dive into Deep Learning* (2nd ed.). Cambridge University Press. https://d2l.ai/chapter_convolutional-modern/googlenet.html
