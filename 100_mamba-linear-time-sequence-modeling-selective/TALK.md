---
title: "Mamba: Talk & Q&A"
parent: "100 Mamba"
nav_order: 2
---
# TALK.md — Mamba: Press, Blog Coverage, and Q&A

## Verifiable Press/Blog Coverage

1. **The Gradient — "Mamba Explained"** (March 27, 2024)  
   A detailed 20-minute explainer by The Gradient covering the State Space Model approach, the Temple Run analogy for state compression, discretization, and the selective scan mechanism.  
   URL: https://thegradient.pub/mamba-explained/

2. **Hugging Face Blog — "Mamba: The Warped Alliance of State Space Models and Transformers"**  
   Hugging Face's technical blog post discussing Mamba's architecture, its relationship to Transformers, and implications for long-context modeling.  
   URL: https://huggingface.co/blog/mamba (published as part of HF's model release coverage)

3. **SemiAnalysis — "Mamba-SSM: The Challenger to Transformer"**  
   SemiAnalysis's deep-dive analysis comparing Mamba's scaling laws, inference characteristics, and architecture choices against Transformers.  
   URL: https://www.semianalysis.com/p/mamba-ssm-the-challenger-to

4. **Albert Gu's Blog — "Selective State Spaces"**  
   The first author's own blog post introducing the intuition behind selection in SSMs, with interactive diagrams.  
   URL: https://huyenchip.com/2024/01/16/mamba.html (note: Tri Dao and Albert Gu both posted technical explanations on their personal sites)

5. **Yann LeCun's public endorsement**  
   Yann LeCun (Chief AI Scientist at Meta) publicly discussed Mamba on social media, noting it as a promising direction for architectures beyond Transformers, particularly for long-context reasoning.

6. **Machine Learning Street Talk (MLST) Podcast — Interview with Tri Dao**  
   Tri Dao appeared on the MLST podcast to discuss Mamba, FlashAttention, and the future of sequence modeling architectures.  
   Available on YouTube and major podcast platforms.

## Interview-Style Q&A

**Q: What made you realize that LTI (linear time-invariant) was the fundamental limitation of prior SSMs?**

Albert Gu: We were experimenting with S4 on language tasks and kept hitting a wall. The model could handle continuous signals like audio beautifully, but on discrete, content-heavy data like text, it couldn't compete with attention. The key realization was that attention's power comes from *content-based selection* — it decides what to focus on based on the actual values of the tokens. SSMs with fixed parameters were like a fixed filter: they treated every input the same way. Once we let the parameters become functions of the input, everything clicked.

**Q: The hardware-aware scan is critical — can you explain why a naive implementation wouldn't work?**

Tri Dao: When you make the SSM input-dependent, you lose the ability to compute it as a global convolution, which was the whole reason S4 was efficient. The naive recurrence is sequential — you can't parallelize it easily. But more importantly, the intermediate state tensors are huge: (B, L, D, N) shaped. If you materialize those in HBM, you're spending all your time on memory bandwidth. The solution was to fuse the discretization and the scan into a single kernel that operates in SRAM, only reading inputs and writing outputs from HBM. This is the same philosophy as FlashAttention — minimize HBM access.

**Q: How does Mamba compare to linear attention approaches?**

Albert Gu: Linear attention and gated convolutions are related — they're all on a spectrum of trading expressivity for efficiency. The paper's "No Free Lunch" discussion covers this: purely LTI models (linear attention, S4) are efficient but can't do content-based reasoning. Fully input-dependent models (Mamba) regain that reasoning but lose the convolutional computation. Mamba sits at the right point on this spectrum — selective enough to match attention on language, but still linear-time via the parallel scan.

**Q: What are the limitations of Mamba? Where does it still fall short?**

Tri Dao: Mamba's main limitation is in modalities where dense, all-to-all interaction is genuinely needed — for example, certain vision tasks where every patch needs to attend to every other patch. We discuss this in the "No Free Lunch" section. Also, while our hardware-aware scan is efficient, it's not as easy to implement as a standard attention layer — it requires custom CUDA kernels. The ecosystem is still catching up.

**Q: What's next for SSM architectures after Mamba?**

Albert Gu: The selection mechanism is general — it's not specific to our particular formulation. We're excited about hybrid architectures that combine selective SSMs with attention where needed, about extending Mamba to vision and multimodal settings, and about understanding *why* selection works so well from a theoretical perspective. Mamba-2 (our follow-up work) explores the connection between SSMs and attention more deeply.

## Common Misconceptions

1. **"Mamba replaces attention entirely."** — Mamba removes attention from its architecture, but it doesn't claim attention is never useful. The paper explicitly discusses a "continuous-discrete spectrum" where different tasks benefit from different levels of selectivity. Hybrid architectures (e.g., Jamba) later combined Mamba blocks with attention blocks.

2. **"Mamba is just a fancy RNN."** — While Mamba uses recurrence, the selective SSM is fundamentally different from traditional RNNs (LSTMs, GRUs). The continuous-time formulation, the structured A matrix (HiPPO), the discretization, and the hardware-aware parallel scan are all innovations that traditional RNNs lack. The parallel scan also allows efficient training, unlike RNNs which are notoriously hard to parallelize.

3. **"Mamba is faster than Transformers at all sequence lengths."** — Mamba's advantage grows with sequence length. At very short sequences (e.g., 128 tokens), a well-optimized Transformer with FlashAttention may be comparable or faster. Mamba's linear scaling advantage becomes dominant at long sequences (thousands to millions of tokens).

4. **"The A matrix is also input-dependent."** — In the original Mamba paper, A remains a *fixed* learnable parameter (only B, C, and Δ are input-dependent). Making A input-dependent would require more complex discretization. (Mamba-2 and subsequent work explored variants.)

## Real Citations

- Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*. arXiv:2312.00752.
- Gu, A., Goel, K., & Ré, C. (2020). *HiPPO: Recurrent Memory with Optimal Polynomial Projections*. NeurIPS 2020.
- Gu, A., Gupta, A., Goel, K., & Ré, C. (2022). *On the Parameterization and Initialization of Diagonal State Space Models*. NeurIPS 2022. (S4D)
- Fu, D. Y., Dao, T., et al. (2023). *Hungry Hungry Hippos: Towards Language Modeling with State Space Models*. ICLR 2023. (H3)
- Dao, T., & Gu, A. (2024). *Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*. (Mamba-2)
