'''
This script contains all the classes of the different possible players :
- a human (PlayerHuman),
- an AI which play randomly (PlayerRandom),
- an AI which use a search tree to decide (PlayerSearchTree),
- an AI which use a neural network to decide (PlayerPPO).
'''

from abc import ABC, abstractmethod
from GameState import GameState, SIZE, WIN_SIZE
import random
import math
import numpy as np
from stable_baselines3 import PPO


class Player(ABC):
    
    def __init__(self, ID) -> None:
        self.ID = ID
        if ID not in (0, 1, "helper"):
            print("Error : the ID of the players has to be be 0 or 1.")
            quit()

    @abstractmethod
    def strategy(self, gameState : GameState) -> tuple: #return a move : a legal triplet of coordinates in the grid
        pass


class PlayerHuman(Player) : 

    def __init__(self, ID) -> None:
        super().__init__(ID)
        self.name = "HUMAN"

    def strategy(self, gameState: GameState) -> tuple:
        possible_moves = gameState.getPossibleMoves()
        nb_possible_moves = len(possible_moves)
        valid_input = False
        while not valid_input:
            print(f"Possible moves pick a number between 0 and {nb_possible_moves - 1} : \n {possible_moves}")
            try:
                chosen_move = int(input())
            except ValueError:
                print(f'The input has to be an integer. Please try again.')
            else:
                if chosen_move in range(nb_possible_moves):
                    valid_input = True
                else:
                    print(f'This move is not valid. Please try again.')
        return possible_moves[chosen_move][0]


class PlayerRandom(Player):

    def __init__(self, ID) -> None:
        super().__init__(ID)
        self.name = "RNDAI"

    def strategy(self, gameState: GameState) -> tuple:
        return random.choice(gameState.getPossibleMoves())[0]


class PlayerSearchTree(Player) :

    def __init__(self, ID, depthMax = 0, epsilon = None) -> None:
        super().__init__(ID)
        self.name = f"STAI{depthMax}"
        self.depthMax = depthMax
        self.epsilon = epsilon
        self.alignment_score_table = dict()
        self.build_alignment_score_table()
        self.state_score_table = dict()
    
    def build_alignment_score_table(self):
        '''
        Build the alignment score table.
        The higher the ratio "pawn /empty square" of the window, the higher the score of the window.
        The distribution of scores follows an exponential function.
        '''
        for i in range(WIN_SIZE+1):
            if WIN_SIZE-i > 0:
                self.alignment_score_table[(WIN_SIZE-i, i)] = int((WIN_SIZE-i)*math.exp(WIN_SIZE-i))
        
    def compute_alignments_score(self, player, other_player, window) -> int:
        '''
        Compute the alignements score windows by subtracting the score of player 0 (or 1) from that of player 1 (or 0 respectively).
        '''
        return self.alignment_score_table.get((window.count(player), window.count(None)), 0) - \
                    self.alignment_score_table.get((window.count(other_player), window.count(None)), 0)
    
    def compute_last_move_score(self, gameState: GameState) -> int:
        '''
        Calculates the score of the last move by calculating the score of the 13 segments that intersect in its coordinate,
        and in the top coordinate.
        '''
        player = self.ID
        other_player = 1-self.ID
        lastMove = gameState.LastMove 
        grid = gameState.Grid
        dif_size = SIZE - WIN_SIZE
        score = 0
        # If it's the first move, return a null score
        if lastMove is None:
            return 0
        else:
            x, y, z = lastMove
        # From the coordinate of the last movement and its z+1 coordinate, record the elements of ...
        list_z = [z] if z+1 == SIZE else [z, z+1]
        for z in list_z:
            segments = []
            segments.extend([
                # ... the X layer column, and of ...
                [grid[x][y][j] for j in range(SIZE)],
                # ... the X layer row, and of ...
                [grid[x][j][z] for j in range(SIZE)],
                # ... the 2 X layer diagonals, and of ...
                [grid[x][i][j] for i in range(y - 1, -1, -1) for j in range(z - 1, -1, -1) if i - y == j - z][::-1] + \
                    [grid[x][i][j] for i in range(y, SIZE) for j in range(z, SIZE) if i - y == j - z],
                [grid[x][i][j] for i in range(y, -1, -1) for j in range(z, SIZE) if i - y == z - j][::-1] + \
                    [grid[x][i][j] for i in range(y, SIZE) for j in range(z - 1, -1, -1) if i - y == z - j],
                # ... the Y layer row, and of ...
                [grid[j][y][z] for j in range(SIZE)],
                # ... the 2 Y layer diagonals, and of ...
                [grid[i][y][j] for i in range(x - 1, -1, -1) for j in range(z - 1, -1, -1) if i - x == j - z][::-1] + \
                    [grid[i][y][j] for i in range(x, SIZE) for j in range(z, SIZE) if i - x == j - z],
                [grid[i][y][j] for i in range(x, -1, -1) for j in range(z, SIZE) if i - x == z - j][::-1] + \
                    [grid[i][y][j] for i in range(x, SIZE) for j in range(z - 1, -1, -1) if i - x == z - j],
                # ... the 2 Z layer diagonals, and of ...
                [grid[i][j][z] for i in range(x - 1, -1, -1) for j in range(y - 1, -1, -1) if i - x == j - y][::-1] + \
                    [grid[i][j][z] for i in range(x, SIZE) for j in range(y, SIZE) if i - x == j - y],
                [grid[i][j][z] for i in range(x, -1, -1) for j in range(y, SIZE) if i - x == y - j][::-1] + \
                    [grid[i][j][z] for i in range(x, SIZE) for j in range(y - 1, -1, -1) if i - x == y - j],
                # ... the 4 diagonals that cross the X, the Y and the Z layers.
                [grid[i][j][k] for i in range(x - 1, -1, -1) for j in range(y - 1, -1, -1) for k in range(z - 1, -1, -1) if i - x == j - y == k - z][::-1] + \
                    [grid[i][j][k] for i in range(x, SIZE) for j in range(y, SIZE) for k in range(z, SIZE) if i - x == j - y == k - z],
                [grid[i][j][k] for i in range(x - 1, -1, -1) for j in range(y - 1, -1, -1) for k in range(z, SIZE)if x - i == y - j == k - z][::-1] + \
                    [grid[i][j][k] for i in range(x, SIZE) for j in range(y, SIZE) for k in range(z, -1, -1)if i - x == j - y == z - k],
                [grid[i][j][k] for i in range(x - 1, -1, -1)for j in range(y, SIZE) for k in range(z - 1, -1, -1)if x - i == j - y == z - k][::-1] + \
                    [grid[i][j][k] for i in range(x, SIZE) for j in range(y, -1, -1) for k in range(z, SIZE)if i - x == y - j == k - z],
                [grid[i][j][k] for i in range(x - 1, -1, -1)for j in range(y, SIZE) for k in range(z, SIZE)if x - i == j - y == k - z][::-1] + \
                    [grid[i][j][k] for i in range(x, SIZE) for j in range(y, -1, -1)for k in range(z, -1, -1)if i - x == y - j == z - k],
                ])
            # By moving a window of size "win_size" in those 13 segments, compute their alignments scores.
            for segment in segments:
                for i in range(dif_size + 1):
                    window = segment[i:i + SIZE]
                    score += self.compute_alignments_score(player, other_player, window)
        # Return the final score
        return score
    
    def MinMaxAlphaBetaPruning(self, gameState: GameState, depth, alpha, beta, maximizingPlayer) -> int:
        '''
        Min max algorithm with alpha beta pruning.
        '''
        # Terminating condition
        if depth == 0 or gameState.checkEnd():
            # Use a transposition table to save the already computed game state
            if str(gameState.Grid) in self.state_score_table:
                return self.state_score_table[str(gameState.Grid)]
            else:
                score = self.compute_last_move_score(gameState)
                self.state_score_table[str(gameState.Grid)] = score
            return score
        # Maximizing player block
        if maximizingPlayer:
            bestValue = -float('inf')
            # Recur on all children
            for move in gameState.getPossibleMoves():
                new_gameState = gameState.copy()
                new_gameState.playLegalMove(move[0])
                value = self.MinMaxAlphaBetaPruning(new_gameState, depth - 1, alpha, beta, False)
                bestValue = max(bestValue, value)
                alpha = max(alpha, bestValue)
                # Alpha Beta Pruning
                if beta <= alpha:
                    break
            return bestValue
        # Minimizing player block
        else:
            bestValue = float('inf')
            # Recur on all children
            for move in gameState.getPossibleMoves():
                new_gameState = gameState.copy()
                new_gameState.playLegalMove(move[0])
                value = self.MinMaxAlphaBetaPruning(new_gameState, depth - 1, alpha, beta, True)
                bestValue = min(bestValue, value)
                beta = min(beta, bestValue)
                # Alpha Beta Pruning
                if beta <= alpha:
                    break
            return bestValue
        
    def random_max_index(self, values):
        '''
        Choose randomly the index betwen the max identical ones.
        '''
        max_index = []
        max_value = max(values)
        for i, value in enumerate(values):
            if value == max_value:
                max_index.append(i)
        return random.choice(max_index)

    def strategy(self, gameState: GameState) -> tuple:
        '''
        Evaluates each movement recursively according to a given depth.
        '''
        # Allow the AI to play randomly. Only usefull to help RL training
        if self.epsilon != None and random.random() < self.epsilon:
            return random.choice(gameState.getPossibleMoves())[0]
        else:
            values = []
            # For each possibles moves,  compute its value
            for move in gameState.getPossibleMoves():
                new_gameState = gameState.copy()
                new_gameState.playLegalMove(move[0])
                value = self.MinMaxAlphaBetaPruning(new_gameState, self.depthMax, -float("inf"), float("inf"), False)
                #print(move, value)
                values.append(value)
            # Choose a random state among those with the highest values
            random_max_index = self.random_max_index(values)
            bestMove = gameState.getPossibleMoves()[random_max_index][0]
            return bestMove


class PlayerPPO(Player):
    def __init__(self, ID: int, model_path: str):
        super().__init__(ID)
        self.name = "PPOAI"
        self.model = PPO.load(model_path)

    def strategy(self, gameState):
        '''
        Let the model predict the best next move.
        '''
        obs = self._encode(gameState)
        action, _ = self.model.predict(obs, deterministic = True)

        possible = gameState.getPossibleMoves()
        orig_indices = [orig for (_move, orig) in possible]

        # If the move is illegal, fallback on random move.
        # In the future, we'd like to use MaskablePPO from sb3-contrib to directly mask illegal moves.
        if action not in orig_indices:
            move = random.choice(possible)[0]
            return move
        else:
            move = next(m for (m, orig) in possible if orig == action)
            return move

    def _encode(self, gameState):
        grid = gameState.Grid
        obs = np.zeros((2 * SIZE * SIZE * SIZE,), dtype = np.int8)
        for x in range(SIZE):
            for y in range(SIZE):
                for z in range(SIZE):
                    v = grid[x][y][z]
                    if v is not None:
                        idx = v * (SIZE**3) + x * (SIZE**2) + y * SIZE + z
                        obs[idx] = 1
        return obs