'''
This script is the main part of the program.It initializes the two players, the game, and loops through consecutive games.
It's the only script that needs to be manipulated initialize the players and play the game.
Currently, there are 4 different possible players :
- a human (PlayerHuman),
- an AI which play randomly (PlayerRandom),
- an AI which use a search tree to decide (PlayerSearchTree),
- an AI which use a neural network to decide (PlayerPPO).
'''

from Game import Game
from GameState import GameState
from Player import PlayerRandom, PlayerHuman, PlayerSearchTree, PlayerPPO

# Create the players, the class defines the strategy

#player0 = PlayerHuman(ID = 0)
player0 = PlayerRandom(ID = 0)
#player0 = PlayerSearchTree(ID = 0, depthMax = 3)
#player0 = PlayerPPO(ID = 0, model_path = "ppo_score_four_final")

#player1 = PlayerHuman(ID = 1)
#player1 = PlayerRandom(ID = 1)
#player1 = PlayerSearchTree(ID = 1, depthMax = 1)
player1 = PlayerPPO(ID = 1, model_path = "ppo_score_four_final")

numberOfGames = 100
gameLengths = [None] * numberOfGames
winners = [None] * numberOfGames

for i in range(numberOfGames):
    print(f'Game {i}')
    game = Game(player0 = player0, player1 = player1, isVerbose = False, gameState = GameState())
    game.run()
    gameLengths[i] = game.CurrentGameState.MoveCount
    winners[i] = game.CurrentGameState.getWinner()

print(f"Average game length : {sum(gameLengths)/len(gameLengths)} moves")
print(f"Player 0 won {len([x for x in winners if x == 0])} \nPlayer 1 won {len([x for x in winners if x == 1])}")
