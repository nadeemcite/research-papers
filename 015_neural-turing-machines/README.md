# Neural Turing Machines

**Paper:** Graves, A., Wayne, G., & Danihelka, I. (2014). Neural Turing Machines. arXiv:1410.5401 [cs.NE].  
**Kaggle:** [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/nadymsazad/neural-turing-machines)
**Link:** https://arxiv.org/abs/1410.5401  
**Authors:** Alex Graves, Greg Wayne, Ivo Danihelka (Google DeepMind, London)  
**Published:** 20 October 2014 (v1); 10 December 2014 (v2)

## Summary

Neural Turing Machines (NTMs) couple a neural network controller to an external memory matrix, creating a differentiable computing architecture analogous to a Von Neumann machine or a Turing machine. The controller reads from and writes to the memory bank via "heads" that produce normalised weightings over memory locations — enabling blurry, attention-based access rather than hard addressing. Because every component (reading, writing, addressing) is differentiable, the entire system can be trained end-to-end with gradient descent. The paper demonstrates that NTMs can infer simple algorithms — copying, repeat copying, associative recall, dynamic N-grams, and priority sorting — purely from input/output examples, and generalise far beyond their training regime (e.g., a network trained on sequences of length 1–20 can copy sequences of length 100).

The core idea is that by giving a neural network an explicit, addressable external memory with content-based and location-based addressing, the network can learn algorithmic procedures rather than mere pattern mappings. The architecture combines a controller (LSTM or feedforward), a memory matrix M of size N×M, and read/write heads that emit weightings, key vectors, erase vectors, and add vectors. Content addressing uses cosine similarity with a key-strength parameter; location addressing uses interpolation gates, circular-convolution shifts, and a sharpening exponent.

## What Problem Does It Solve

Imagine you're trying to teach a robot to repeat back a list of numbers you tell it — like a phone number. You say "3, 7, 2, 9, 5" and the robot should repeat "3, 7, 2, 9, 5." A regular neural network is like a person with no notepad — they try to hold everything in their head at once. For a short list that works, but give them 50 numbers and they get confused and start making mistakes.

A Neural Turing Machine is like giving that person a notepad (the memory) and a pen (the write head) and the ability to look back at what they wrote (the read head). Now they can write down each number as they hear it, then read them back one by one. The clever part is that the notepad system is "soft" — instead of writing in one exact spot, the robot writes a little bit across several spots (like writing lightly across a few lines), and reads by blending what's at several spots. This "softness" means the whole system can be trained with regular AI learning methods (gradient descent), because everything is smooth and differentiable. The result: the robot can learn to copy, sort, and recall sequences on its own — and can even handle lists much longer than anything it practiced on.

## Key Method Details

- **Memory matrix** M_t: N rows × M columns, where N is the number of memory locations and M is the vector size per location.
- **Read heads** produce weightings w_t (normalised, sum to 1) over N locations; the read vector r_t = Σ w_t(i) · M_t(i) — a convex combination of memory rows.
- **Write heads** decompose writing into erase (M̃_t(i) ← M_{t-1}(i) ⊙ [1 − w_t(i)·e_t]) then add (M_t(i) ← M̃_t(i) + w_t(i)·a_t).
- **Content-based addressing**: cosine similarity K[u,v] = u·v / (‖u‖·‖v‖) between key k_t and each memory row, scaled by key strength β_t, softmax-normalised.
- **Location-based addressing**: interpolation gate g_t blends content weighting with previous weighting; shift weighting s_t performs circular convolution (rotation); sharpening exponent γ_t ≥ 1 sharpens the result.
- **Controller**: can be LSTM (recurrent, has internal registers) or feedforward (more transparent, limited by number of heads).
- **Training**: cross-entropy loss with logistic sigmoid outputs; gradient clipping to (-10, 10); optimised with RMSProp; learning rates 1e-4 to 3e-5.
- **Copy task**: input is a sequence of 8-bit random binary vectors (length 1–20) followed by a delimiter; target is the same sequence without the delimiter. NTM generalises to length 100+ while LSTM degrades beyond 20.

## Influence

NTMs are foundational to the entire family of memory-augmented neural networks. They directly inspired Differentiable Neural Computers (DNCs, Graves et al. 2016), Memory Networks (Weston et al. 2014), End-to-End Memory Networks (Sukhbaatar et al. 2015), and the broader programme of neural architectures with external memory. The differentiable read/write paradigm influenced attention mechanisms in transformers (which can be viewed as performing content-based addressing over a "memory" of key-value pairs). The paper has been cited thousands of times (2000+ on Google Scholar) and is considered a landmark in combining neural networks with algorithmic/programmatic computation.
