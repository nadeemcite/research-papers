---
title: "GPT-3: Talk & Q&A"
parent: "080 GPT-3"
nav_order: 2
---
# TALK — GPT-3: Language Models are Few-Shot Learners

## Verifiable Press/Blog Coverage

1. **The New York Times** — [Meet GPT-3. It Has Learned to Code (and Blog and Argue).](https://www.nytimes.com/2020/11/24/science/artificial-intelligence-ai-gpt3.html) (November 2020, Cade Metz) — a long feature on what people built with GPT-3 (code, tweets, poems, chat) and where it still fails, such as biased and toxic language.

2. **The Guardian** — [A robot wrote this entire article. Are you scared yet, human?](https://www.theguardian.com/commentisfree/2020/sep/08/robot-wrote-this-article-gpt-3) (September 2020) — GPT-3 was used to write an op-ed, one of the first instances of AI-generated content in a major newspaper.

3. **MIT Technology Review** — [OpenAI's new language generator GPT-3 is shockingly good—and completely mindless](https://www.technologyreview.com/2020/07/20/1005454/openai-machine-learning-language-generator-gpt-3-nlp/) (July 2020) — an early hands-on look at what GPT-3 gets right and where it makes no sense.

4. **Wired** — [AI Text Generator GPT-3 Is Learning Our Language—Fitfully](https://www.wired.com/story/ai-text-generator-gpt-3-learning-language-fitfully/) (2020; shared as "Did a Person Write This Headline, or a Machine?") — how far the program has come and how far it still has to go.

5. **VentureBeat** — [OpenAI launches an API to commercialize its research](https://venturebeat.com/business/openai-launches-an-api-to-commercialize-its-research) (June 2020) — the launch of the OpenAI API, the first way outside users could try GPT-3.

## Q&A (answers from the paper)

**Q: What is the key insight of GPT-3?**
A (from the paper, Abstract and Section 1): Scaling up a language model, here to 175 billion parameters, makes it much better at in-context learning: it can do a new task from a task description and a few examples in the prompt, with no weight updates. The authors also find that the gap between zero-, one- and few-shot performance often grows with model size.

**Q: How does few-shot learning differ from fine-tuning?**
A (from the paper, Abstract and Section 2): In fine-tuning, the model's weights are updated by gradient descent on task-specific data. In the few-shot setting, the weights stay frozen: the model reads a text prompt with examples and does the task by predicting the next tokens. The abstract says: "For all tasks, GPT-3 is applied without any gradient updates or fine-tuning."

**Q: What are the three evaluation settings?**
A (from the paper, Section 2): Zero-shot (a task description only), one-shot (one example) and few-shot (as many examples as fit in the 2,048-token context window, typically 10 to 100). For most tasks, performance scales fairly smoothly with model size in all three settings.

**Q: What are some limitations noted in the paper?**
A (from the paper, Sections 4, 5 and 6): Long GPT-3 samples can repeat themselves, lose coherence or contradict themselves. GPT-3 does little better than chance on some "comparison" tasks (WiC, ANLI) and on some reading comprehension tasks such as QuAC and RACE. Some test sets may have leaked into the web training data (Section 4). The authors also discuss misuse, bias and energy use (Section 6).

## Common Misconceptions

1. **"GPT-3 can learn new tasks."** — Not exactly. GPT-3 doesn't *learn* in the gradient-descent sense. It uses in-context learning: the "knowledge" is already in the pretrained weights; the prompt merely directs the model to the right output pattern.

2. **"GPT-3 was the first to do few-shot learning."** — In-context learning at scale was popularized by GPT-3, but the concept of meta-learning and few-shot learning predates it (e.g., MAML, matching networks). GPT-3's contribution was showing that *scale alone* can achieve this without specialized architectures.

3. **"Bigger is always better."** — Not for every task. In the word-scrambling tests (Section 3.9.2), scores on anagrams and the other tasks do improve with model size, but no model of any size can reverse the letters of a word. The paper also discusses the societal risks of scale. Later work (Chinchilla) showed that, for a fixed compute budget, the number of training tokens matters as much as the parameter count.

## Real Citations

- Brown, T. B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., et al. (2020). "Language Models are Few-Shot Learners." arXiv:2005.14165. NeurIPS 2020.
- Kaplan, J., et al. (2020). "Scaling Laws for Neural Language Models." arXiv:2001.08361. — The scaling law paper that motivated GPT-3's design.
- Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners." (GPT-2 paper) — The predecessor that established the autoregressive LM approach.
- Hoffmann, J., et al. (2022). "Training Compute-Optimal Large Language Models." arXiv:2203.15556. (Chinchilla) — Later showed GPT-3 was undertrained relative to its size.
