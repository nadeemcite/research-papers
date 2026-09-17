# ResNet — Deep Residual Learning for Image Recognition

**arXiv:** [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

**Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft Research)

## Summary

Residual networks (ResNets) introduced a simple but powerful idea: instead of asking a stack of layers to learn a direct mapping \(H(x)\), train them to learn the *residual* mapping \(F(x) = H(x) - x\) and add the input \(x\) back with a shortcut (skip) connection. By reformulating deep layers as residual functions, the paper showed that networks with hundreds or even thousands of layers can be optimized reliably, whereas plain (stacked) networks suffer from degradation—accuracy saturates and then drops as depth increases, even when overfitting is not the cause.

The key method is the **residual block**. For an input \(x\), a block with weight layers computes \(F(x) + x\). When the dimensions of \(F(x)\) and \(x\) differ, a linear projection by \(1×1\) convolutions matches them. A deeper architecture is built by stacking these blocks, and experiments on ImageNet demonstrate that ResNets scale gracefully from 34 layers to 152 layers, winning the ILSVRC 2015 classification competition with a 3.57% top-5 test error. The same principle also produced 100+ layer networks that train well on CIFAR-10.

ResNet’s influence is enormous: it replaced VGG-style plain stacks as the default backbone for vision, enabled much deeper architectures (ResNeXt, DenseNet, EfficientNet, transformers), and the skip-connection idea is now used in nearly every modern network, including attention models and diffusion backbones.

## What problem does it solve

Imagine you are trying to send a message across a very long chain of whispering friends. With every extra friend, the message gets a little more garbled. Adding more friends does not help—it actually makes the final message worse, even though each friend is trying hard. ResNet solves the "long chain" problem for deep neural networks.

In a normal deep network, every new layer has to rewrite the entire signal from scratch. As the network gets deeper, this becomes harder and harder, so accuracy stops improving and even drops. ResNet gives each layer an *express lane*—a shortcut that carries the original signal forward. A layer only has to learn the small change (the "residual") it wants to make to that signal. If a layer is not useful, it can simply let the signal pass through. Because the original information is never lost, the network can become much deeper without getting confused, much like letting every friend both whisper a small correction and pass the original message on unchanged.
