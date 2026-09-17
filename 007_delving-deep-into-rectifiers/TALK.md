# TALK — Delving Deep into Rectifiers

## Verifiable press / blog coverage

* **arXiv trackbacks:** The paper has 16 blog links listed at `https://arxiv.org/tb/1502.01852`. (arXiv's trackback page is the canonical source; specific blog URLs rotate over time.)
* **The Morning Paper** by Adrian Colyer discussed the paper's initialization analysis: https://blog.acolyer.org/2016/04/20/delving-deep-into-rectifiers-surpassing-human-level-performance-on-imagenet-classification/
* **PyTorch documentation for `torch.nn.init.kaiming_normal_`** explicitly cites this paper: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_
* **TensorFlow/Keras docs** for `tf.keras.initializers.HeNormal` reference the He et al. paper: https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal
* **DeepLearning.ai / Towards DataScience** explainers on He initialization commonly reference this paper when contrasting it with Xavier initialization.

## Interview-style Q&A

**Q: What was the most surprising result in this paper?**  
A: That a team could train a 22-layer (and even a 30-layer) ReLU network from scratch without layer-wise pre-training, and that the single-model result beat all existing multi-model results on ImageNet at the time. The 4.94% top-5 error was also the first published number to exceed the reported human-level top-5 error of 5.1%.

**Q: Is PReLU just a fancy Leaky ReLU?**  
A: Not quite. Leaky ReLU fixes the negative slope (usually 0.01). PReLU makes that slope a learnable parameter. The paper shows that the network learns different slopes for different layers and channels, and that this modest change yields a measurable accuracy gain with almost no extra compute or overfitting risk.

**Q: Why is the magic number `2` in He initialization?**  
A: ReLU zeros out half of its inputs, so the expected squared activation is half the variance of the pre-activation. To keep the variance constant through the layer, the weight variance must be doubled relative to the linear/Xavier assumption. Hence `Var(W) = 2/fan_in`.

**Q: Should I always use He initialization for every network?**  
A: It is the right default for ReLU-like activations. For linear/tanh/sigmoid activations, Xavier/Glorot initialization is still appropriate because the `2` correction factor does not apply.

**Q: Did the paper invent ResNet?**  
A: No. ResNet (also by Kaiming He and colleagues) came later, in 2015. This paper showed that deep rectifier nets *could* be trained with proper initialization; ResNet then showed how to train even deeper nets by adding identity skip connections.

## Common misconceptions

1. **"He init is just Xavier init with a factor of 2."**  
   The factor is a consequence of the ReLU non-linearity, but the derivation is different: He init explicitly computes `E[x^2]` for a rectified distribution rather than assuming zero-mean activations.
2. **"PReLU causes huge overfitting."**  
   The paper reports the opposite. The extra parameter count is negligible (one scalar per channel or per layer), so it does not meaningfully increase model capacity.
3. **"This paper is only about ImageNet."**  
   ImageNet was the benchmark, but the two main ideas — PReLU and He init — are used across vision, speech, NLP, and scientific ML.

## Real citations

* He, K., Zhang, X., Ren, S., & Sun, J. (2015). *Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification*. arXiv:1502.01852 [cs.CV].
* Glorot, X., & Bengio, Y. (2010). *Understanding the difficulty of training deep feedforward neural networks*. AISTATS 2010. (cited as the Xavier baseline).
* Russakovsky, O., et al. (2015). *ImageNet Large Scale Visual Recognition Challenge*. IJCV. (cited for the 5.1% human-level top-5 error benchmark.)
* Szegedy, C., et al. (2015). *Going Deeper with Convolutions*. CVPR 2015. (GoogLeNet, the 6.66% prior state of the art.)
* Nair, V., & Hinton, G. E. (2010). *Rectified Linear Units Improve Restricted Boltzmann Machines*. ICML 2010. (early ReLU usage.)
