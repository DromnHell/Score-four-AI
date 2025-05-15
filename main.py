'''
This script is the main of the Score Four program. It initializes the two players, the game, and loop accross 
epoch (several consecutive games). It is the only script that needs to be manipulated to play the game,
selecting and initializing the players.
'''

from Game import Game
from GameState import GameState
from Player import PlayerRandom, PlayerHuman, PlayerSearchTree, PlayerPPO

# Create the players, the class defines the strategy

#player0 = PlayerHuman(ID = 0)
#player0 = PlayerRandom(ID = 0)
#player0 = PlayerSearchTree(ID = 0, depthMax = 3)
player0 = PlayerPPO(ID = 0, model_path = "ppo_score_four")

#player1 = PlayerHuman(ID = 1)
player1 = PlayerRandom(ID = 1)
#player1 = PlayerSearchTree(ID = 1, depthMax = 3)
#player1 = PlayerPPO(ID = 1, model_path = "ppo_score_four")

numberOfGames = 1000
gameLengths = [None] * numberOfGames
winners = [None] * numberOfGames

for i in range(numberOfGames):
    print(f'Game {i}')
    game = Game(player0 = player0, player1 = player1, isVerbose = False, gameState = GameState())
    game.run()
    #basic statistic collection on the game once it's ended
    gameLengths[i] = game.CurrentGameState.MoveCount
    winners[i] = game.CurrentGameState.getWinner()

print(f"Average game length : {sum(gameLengths)/len(gameLengths)} moves")
print(f"Player 0 won {len([x for x in winners if x == 0])} \nPlayer 1 won {len([x for x in winners if x == 1])}")
