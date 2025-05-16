# Score Four

## Source
This repository is a fork of the https://github.com/Latylus/ScoreFour repository maintained by Pierre Lataillade.

## Abstract game
For a rough information page on the game itself, check out [its Wikipedia page](https://en.wikipedia.org/wiki/Score_Four).

## Implementation

### main.py
This script define the players to use then launch as many games of Score Four as you want and check your stats from here.
For reference about 3000 games per second can be completed with PlayerRandom and a standard 4 Size.

### game.py
This script define the class of the game and the principal loop game.

### player.py
This script defines all the different possible players.
A player is required to implement a game strategy i.e return a legal move from a given gamestate.
Currently, there are 4 different possible players :
* a human (PlayerHuman),
* an AI which play randomly (PlayerRandom),
* an AI which use a search tree to decide (PlayerSearchTree),
* an AI which use a neural network to decide (PlayerPPO).

### game_state.py
This script contains most of the game logic. Also contains the parameters for the grid size and the win condition size (if you want to play Score 5). Not intended to be modified except for these parameters.

### train_PPO.py
This script is used to train a PPO model to play score four. It generates a “ppo_score_four_final” file containing the model's weights.

## Contribution

Contributions are welcome !

There are two main areas for improvement:
* The search tree AI takes too long to calculate the best move. The algorithm needs to be optimized.
* The neural network AI wins 90% of the time against the random AI, but loses all the time against the search tree AI with a depth of 1, that is quite bad.

Here's a list of improvements to neural network training:

#### Curriculum Learning Enhancements
- **Progressive opponents**: Add phases against increasingly deep search-tree AIs (depth 2, 3, 4) or varied opponent pools (Random, Tree, Human).  
- **Dynamic board size**: Start training on smaller boards, then ramp up to 4×4×4.  
- **Incremental alignment length**: Begin by rewarding shorter alignments (e.g. 2 in a row), then require longer ones.  
- **Evolving opening positions**: Train from empty boards, then introduce critical opening scenarios.

#### Reward Shaping
- **Intermediate bonuses**: +0.1 reward for forming 2- or 3-piece alignments.  
- **Per-move penalty**: −0.01 per action to speed up games and discourage aimless play.

#### Native Illegal-Action Masking
- Implement a **dynamic action mask** in the policy so full columns are never proposed (e.g. via sb3-contrib’s MaskablePPO).

#### Hyperparameter Fine-Tuning
- Adjust **`ent_coef`** to balance exploration vs. exploitation.  
- Anneal **`learning_rate`** or tweak **`clip_range`** toward the end of training.  
- Experiment with different **rollout lengths** (`n_steps`) and **batch sizes**.

#### Advanced Self-Play (League Training)
- Keep a **pool of historical checkpoints** and train against a diverse set of past versions.  
- Build an **auto-challenger selection** mechanism to continually push the agent’s limits.

#### Imitation Learning / Pre-training
- Create a dataset of MinMax (depth 3–4) or human games and **pretrain the policy**.  
- Then **fine-tune with PPO** to speed up convergence.

#### Automated Evaluation & Checkpointing
- Define **validation metrics** (win rates vs Random, Tree, etc.) and **save the best model** automatically.  
- **Stop training** when performance metrics plateau.

## Contact

For any questions, please contact me at remi.dromnelle@gmail.com.