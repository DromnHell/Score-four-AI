'''
This script is part of the Score Four program. It contains the class of the game and the principal loop game.
'''

from GameState import GameState 
from Player import Player, PlayerPPO
import datetime

class Game:
    CurrentGameState = None
    Player0 = None
    Player1 = None

    IsVerbose = False

    def __init__(self, player0 : Player, player1 : Player , isVerbose : bool = False, gameState : GameState = GameState()) -> None:
        self.Player0 = player0
        self.Player1 = player1
        self.CurrentGameState = gameState
        self.IsVerbose = isVerbose

    def logInfo(self, message: str)-> None:
        if(self.IsVerbose):
            print(message)

    def run(self) -> None:
        while (not self.CurrentGameState.checkEnd()) :
            currentPlayer = self.Player0 if self.CurrentGameState.IsPlayerZeroTurn else self.Player1
            start_time = datetime.datetime.now()
            move = currentPlayer.strategy(self.CurrentGameState)
            #print(f'Player {currentPlayer.ID}, Move : {move}')
            end_time = datetime.datetime.now()
            time = (end_time - start_time).total_seconds()
            #print(f'Time = {time}')
            self.logInfo(f"Move {self.CurrentGameState.MoveCount} : player {0 if self.CurrentGameState.IsPlayerZeroTurn else 1} playing move {move}")
            self.CurrentGameState.playLegalMove(move)

        if self.CurrentGameState.getWinner() is not None:
            self.logInfo(f"END GAME : Player {self.CurrentGameState.getWinner()} wins")
        else :
            self.logInfo("END GAME : draw!")
