---
title: "AlphaZero: Talk & Q&A"
parent: "061 AlphaZero"
nav_order: 2
---
# AlphaZero — Press Coverage, Q&A, Misconceptions, Citations

## Verifiable Press & Blog Coverage

1. **The Guardian** — [AlphaZero AI beats champion chess program after teaching itself in four hours](https://www.theguardian.com/technology/2017/dec/07/alphazero-google-deepmind-ai-beats-champion-program-teaching-itself-to-play-four-hours) (December 7, 2017). Samuel Gibbs reported on AlphaZero's win over Stockfish, stressing that it learned chess from the rules alone through self-play.

2. **Wired** — [Alphabet's Latest AI Show Pony Has More Than One Trick](https://www.wired.com/story/alphabets-latest-ai-show-pony-has-more-than-one-trick/) (December 2017). Wired called AlphaZero "the first multi-skilled AI board-game champ": one program that taught itself chess, shogi and Go.

3. **MIT Technology Review** — [Alpha Zero's "Alien" Chess Shows the Power, and the Peculiarity, of AI](https://www.technologyreview.com/2017/12/08/147199/alpha-zeros-alien-chess-shows-the-power-and-the-peculiarity-of-ai/) (December 8, 2017). Will Knight covered Demis Hassabis's talk at NIPS 2017 about AlphaZero's unusual playing style.

4. **BBC News** — [Google's 'superhuman' DeepMind AI claims chess crown](https://www.bbc.com/news/technology-42251535) (December 2017). The BBC reported the 100-game match against Stockfish (28 wins, 72 draws, 0 losses) and asked grandmaster Peter Heine Nielsen for his reaction.

5. **Chess.com** — [Google's AlphaZero Destroys Stockfish In 100-Game Match](https://www.chess.com/news/view/google-s-alphazero-destroys-stockfish-in-100-game-match) (December 2017). Game analysis and reactions, including Garry Kasparov: "It's a remarkable achievement, even if we should have expected it after AlphaGo."

6. **Chess.com** — [AlphaZero Chess: Reactions From Top GMs, Stockfish Author](https://www.chess.com/news/view/alphazero-reactions-from-top-gms-stockfish-author) (December 2017). First reactions from top grandmasters and a full statement from Stockfish co-author Tord Romstad on the match conditions.

7. **DeepMind blog** — [AlphaZero: Shedding new light on chess, shogi, and Go](https://deepmind.google/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/) (December 2018). DeepMind's summary of the final Science paper and its re-run matches.

## Q&A (answers from the paper and linked sources)

The paper is [arXiv:1712.01815](https://arxiv.org/abs/1712.01815). It has no numbered sections, so answers point to its main text or Methods.

**Q: How can AlphaZero search a thousand times fewer positions than Stockfish and still win?**
A (from the paper, main text): AlphaZero searches about 80 thousand positions per second in chess, against 70 million for Stockfish. The paper says it "compensates for the lower number of evaluations by using its deep neural network to focus much more selectively on the most promising variations – arguably a more 'human-like' approach to search".

**Q: What did AlphaZero learn about chess that surprised grandmasters?**
A (from linked press and the paper): At NIPS 2017, Demis Hassabis said, as reported by [MIT Technology Review](https://www.technologyreview.com/2017/12/08/147199/alpha-zeros-alien-chess-shows-the-power-and-the-peculiarity-of-ai/): "It doesn't play like a human, and it doesn't play like a program. It plays in a third, almost alien, way." He described how it sometimes offers up a bishop and queen to exploit a positional advantage. Grandmaster Peter Heine Nielsen told the [BBC](https://www.bbc.com/news/technology-42251535): "I always wondered how it would be if a superior species landed on earth and showed us how they played chess. Now I know." The paper (main text, Table 2) shows that AlphaZero independently discovered and often played the 12 most common human openings.

**Q: Was the match against Stockfish fair?**
A (from the paper and linked sources): The paper says Stockfish played with 64 threads and a 1 GB hash, at one minute per move. In a statement to [Chess.com](https://www.chess.com/news/view/alphazero-reactions-from-top-gms-stockfish-author), Stockfish co-author Tord Romstad said the results were "not particularly meaningful because of the rather strange choice of time controls and Stockfish parameter settings": a fixed minute per move switches off Stockfish's time management, the version was a year old, and the hash tables were too small for that many threads.

In the same article Hikaru Nakamura said Stockfish "was basically running on what would be my laptop". For the final Science version, DeepMind re-ran the match with three hours per game plus 15 seconds per move and Stockfish on 44 CPU cores; AlphaZero won 155 games and lost 6 out of 1,000 ([DeepMind blog](https://deepmind.google/blog/alphazero-shedding-new-light-on-chess-shogi-and-go/)).

**Q: How does AlphaZero differ from AlphaGo Zero?**
A (from the paper, main text): The paper lists several differences: (1) AlphaZero estimates the expected outcome, which allows draws, instead of only win/loss; (2) it does not use board symmetries for data augmentation (chess and shogi are not symmetric like Go); (3) it keeps one network that is updated continually, instead of picking a best player after each iteration; (4) it uses the same hyperparameters for all games without game-specific tuning.

**Q: Can people run AlphaZero themselves?**
A: DeepMind did not release the AlphaZero program. The open-source [Leela Chess Zero](https://lczero.org/) project re-implemented the same approach for chess.

## Common Misconceptions

1. **"AlphaZero was trained on human chess games."**
   *Correction:* AlphaZero was trained entirely by self-play (tabula rasa), starting from random play. The only domain knowledge given was the rules of chess. It rediscovered human openings independently — the paper explicitly notes it "independently discovered and played frequently" the most common human openings.

2. **"AlphaZero is just a bigger, faster chess engine."**
   *Correction:* AlphaZero is fundamentally different from traditional engines. Stockfish uses alpha-beta search with handcrafted evaluation functions and domain-specific heuristics refined over decades. AlphaZero uses MCTS guided by a neural network trained by reinforcement learning. The search algorithm, evaluation method, and learning approach are all different.

3. **"The 4-hour training means AlphaZero is easy to reproduce."**
   *Correction:* Those 4 hours ran on 5,000 first-generation TPUs generating self-play games plus 64 second-generation TPUs training the network (paper, main text). That is roughly 20,000 TPU-hours for self-play alone, far beyond a single machine. Reproducing AlphaZero's full results needs enormous compute.

4. **"AlphaZero proved neural networks are inherently superior to alpha-beta search."**
   *Correction:* The paper says its results call into question "the widely held belief that alpha-beta search is inherently superior in these domains", but it does not prove neural networks are always better. Critics such as Tord Romstad ([Chess.com](https://www.chess.com/news/view/alphazero-reactions-from-top-gms-stockfish-author)) pointed out that AlphaZero ran on custom hardware not available to ordinary users. Today's Stockfish combines alpha-beta search with a small neural-network evaluation (NNUE).

5. **"AlphaZero can play any game."**
   *Correction:* AlphaZero requires a perfect-information, turn-based game with known rules and a well-defined terminal scoring. It cannot directly handle imperfect-information games (poker), real-time games (StarCraft, though AlphaStar extended the paradigm), or games without clear win/loss/draw outcomes. MuZero (2019) later removed even the requirement for known rules.

## Real Citations

- Silver, D., Hubert, T., Schrittwieser, J., et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and go through self-play. Science, 362(6419), 1140–1144. doi:10.1126/science.aar6404
- Silver, D., Schrittwieser, J., Hubert, T., et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero). Nature, 550, 354–359. (AlphaZero's direct predecessor)
- Silver, D., Huang, A., Maddison, C.J., et al. (2016). Mastering the game of Go with deep neural networks and tree search (AlphaGo). Nature, 529, 484–489. (Original AlphaGo)
- Schrittwieser, J., Antonoglou, I., Hubert, T., et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero). Nature, 588, 604–609. (Extends AlphaZero to learn the rules)
- Kocsis, L. & Szepesvári, C. (2006). Bandit based Monte-Carlo Planning. ECML. (UCT — the MCTS selection algorithm used by AlphaZero)
- Browne, C., Powley, E., Whitehouse, D., et al. (2012). A Survey of Monte Carlo Tree Search Methods. IEEE TCIAIG, 4(1), 1–43. (MCTS survey)
- Leela Chess Zero (lc0), community project. https://lczero.org/ (Open-source AlphaZero-style chess engine; the separate Leela Zero for Go was written by Gian-Carlo Pascutto)
