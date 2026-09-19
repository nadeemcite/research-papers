# Show and Tell: A Neural Image Caption Generator

**Authors:** Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan (Google)  
**Published:** CVPR 2015 (arXiv: 18 Nov 2014, last revised 19 Apr 2015)  
**arXiv:** https://arxiv.org/abs/1411.4555

## Summary

This paper introduces **NIC (Neural Image Caption)**, an end-to-end neural network that automatically generates natural-language descriptions of images. The architecture is a straightforward but powerful combination of two mature building blocks: a **pretrained Convolutional Neural Network (CNN)** as an image encoder and a **Long Short-Term Memory (LSTM)** network as a language decoder. The CNN (a GoogLeNet/Inception-based model trained on ImageNet) processes an image and produces a fixed-length feature vector. This vector is fed into the LSTM as the initial input, after which the LSTM generates a caption one word at a time — each word conditioned on the image encoding and all previously generated words. The entire system is trained jointly to maximise the likelihood p(S|I) of producing the correct caption S given image I, using stochastic gradient descent.

On the Pascal dataset, NIC achieved a BLEU-1 score of 59 (vs. previous state-of-the-art 25, human ~69). On Flickr30k it improved from 56 to 66, on SBU from 19 to 28, and on the newly released MS COCO dataset it achieved a BLEU-4 of 27.7 — the state of the art at the time. The model was later open-sourced in TensorFlow and won (tied for first place) the MS COCO 2015 Image Captioning Challenge.

### Core idea

The central insight is that the image captioning problem can be cast as a **sequence-to-sequence** problem — the same framework that had just revolutionised machine translation (Sutskever et al., 2014; Cho et al., 2014). Instead of translating English to French, we "translate" an image into a sentence:

1. **Image encoding:** A pretrained CNN extracts a rich feature vector from the image. The last hidden layer before classification becomes the image embedding.
2. **Sentence decoding:** An LSTM, the same architecture used for language modeling and machine translation, takes the image embedding as its first input and then generates words one at a time. Special `<START>` and `<END>` tokens delimit the sentence.
3. **Training objective:** Maximise log p(S|I) = Σ log p(S_t | I, S_0, …, S_{t−1}), using teacher forcing (feeding ground-truth words during training).
4. **Inference:** Beam search produces the highest-probability caption.

### Key method details relevant to the code

- The CNN encoder is pretrained on ImageNet; only the LSTM and embedding layers are trained from scratch (in the original paper, CNN weights are frozen).
- Words are represented as one-hot vectors of dimension equal to the vocabulary size, then embedded into a continuous space via a learned embedding matrix W_e.
- 512 dimensions are used for both the word embeddings and the LSTM hidden state.
- Vocabulary is built by keeping words that appear at least 5 times in the training set.
- Beam search at inference time improves caption quality over greedy decoding.
- The LSTM is unrolled over the sequence length, with the image embedding injected as x_{−1} (before the first word).
- Dropout and ensembling provide a few BLEU points of improvement; the notebook uses dropout.

### Influence

This paper is one of the foundational works in **vision-to-language generation**. It demonstrated that a single end-to-end neural network could generate fluent, accurate image descriptions — eliminating the complex pipeline of object detection + template-based generation that preceded it. The architecture (CNN encoder + RNN decoder) became the standard template for image captioning and inspired Show, Attend and Tell (Xu et al., 2015), which added visual attention. The model was open-sourced by Google in TensorFlow (the "im2txt" repository) and tied for first in the MS COCO 2015 captioning challenge. The paper has been cited over 11,000 times and is a staple in deep learning curricula worldwide.

## What problem does it solve?

Imagine you have a friend who is blind, and you want to show them photos on your phone. You'd have to describe every picture out loud: "That's a dog catching a frisbee in the park," or "There's a pizza on a table with some drinks." But what if you had thousands of photos? You can't describe them all yourself.

Before this paper, computers could *sort of* do this, but the approach was clunky. They would first try to list all the objects they saw ("dog," "frisbee," "grass") and then stitch those words together into a sentence using fixed templates — like filling in blanks in a form. The result was often robotic and wrong, like "A dog is on a frisbee" when the dog is actually *catching* the frisbee.

This paper solved that by teaching a computer to do what a human does: look at the whole picture and just *describe it naturally*. It used a "vision brain" (CNN) that has looked at millions of images and learned to understand what's in a picture, connected to a "language brain" (LSTM) that has read lots of captions and learned to write sentences. Together, they could look at a new photo and write a real, natural sentence about it — sometimes even writing descriptions they'd never seen before, by combining things they'd learned separately.
