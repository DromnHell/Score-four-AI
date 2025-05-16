'''
This script is used to train a PPO model to play score four. It generates a “ppo_score_four_final” file containing the model's weights.
For the time being, the model is trained against the random player for 1,000,000 actions (alternately as first and second player)
and then in self-play against the same PPO model but with a stochastic policy for 1,000,000 actions (alternately as first and second player).
'''

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from GameState import GameState, SIZE
from Player import PlayerRandom, PlayerSearchTree
import random


class ScoreFourEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, opponent_type = 'random'):
        super().__init__()
        self.opponent_type = opponent_type
        self.mixed_options = ['random', 'searchTree1', 'selfplay']
        self.mixed_weights = [0.333333] * 3
        self.observation_space = gym.spaces.Box(low = 0, high = 1,
                                                shape = (2 * SIZE**3,),
                                                dtype = np.int8)
        self.action_space = gym.spaces.Discrete(SIZE * SIZE)
        self.agent_id = 0
        self.model = None
        self.state = None
        self.opponent = None
        self.reset()

    def reset(self, **kwargs):
        self.state = GameState()
        # Randomize starting player
        self.agent_id = random.choice([0, 1])
        # Select opponent for this episode
        opt = self.opponent_type
        if opt == 'mixed':
            opt = random.choices(self.mixed_options, weights = self.mixed_weights)[0]
        if opt == 'random':
            self.opponent = PlayerRandom(1)
        elif opt == 'searchTree1':
            self.opponent = PlayerSearchTree(1, depthMax = 1)
        elif opt == 'searchTree3':
            self.opponent = PlayerSearchTree(1, depthMax = 3)
        elif opt == 'selfplay':
            self.opponent = None
        else:
            self.opponent = PlayerRandom(1)

        # If PPO is player1 (starts second), let rival play first
        if self.agent_id == 1:
            # choose opponent's move
            possible = self.state.getPossibleMoves()
            if self.opponent is None:
                opp_idx, _ = self.model.predict(self._get_obs(), deterministic = False)
            else:
                opp_move = self.opponent.strategy(self.state)
                orig_map = {tuple(m): orig for (m, orig) in possible}
                opp_idx = orig_map.get(tuple(opp_move), random.choice(possible)[1])
            self._execute_move(opp_idx, 1 - self.agent_id)

        return self._get_obs(), {}

    def step(self, action_agent):

        # Penalization of illegal moves.
        # In the future, we'd like to use MaskablePPO from sb3-contrib to directly mask illegal moves.
        possible = self.state.getPossibleMoves()  # [ ((x,y,z), orig_idx), … ]
        orig_indices = [orig for (_m, orig) in possible]
        if action_agent not in orig_indices:
            return self._get_obs(), -1.0, True, False, {'winner': None, 'illegal_action': True}

        # Agent move execution
        reward_agent, done_agent = self._execute_move(action_agent, self.agent_id)
        obs_after_agent = self._get_obs()
        if done_agent:
            return obs_after_agent, reward_agent, done_agent, False, {'winner': self.agent_id, 'illegal_action': False}

        # Rival move execution (Random rival, Search Tree rival or self-play with the same politic but with stochastic choices)
        possible = self.state.getPossibleMoves()
        if self.opponent is None:
            # Self-play rival with the same politic but with stochastic choices
            rival_action, _ = self.model.predict(obs_after_agent, deterministic = False)
        else:
            # Random or Search Tree rivals requires to convert the action tuple (x,y,z) in origin index
            rival_action_tuple = self.opponent.strategy(self.state)
            orig_map = {tuple(m): orig for (m, orig) in possible}
            rival_action = orig_map.get(tuple(rival_action_tuple))
            if rival_action is None:
                print("ERROR")
        reward_rival, done_rival = self._execute_move(rival_action, 1 - self.agent_id)
        obs_after_rival = self._get_obs()
        if done_rival:
            return obs_after_rival, -reward_rival, done_rival, False, {'winner': 1 - self.agent_id, 'illegal_action': False}

        return obs_after_rival, reward_agent, False, False, {'winner': None, 'illegal_action': False}

    def _execute_move(self, action, player):

        # Illegal action => instant defeat
        possible = self.state.getPossibleMoves()
        if action >= len(possible):
            return (-1, True) if player == self.agent_id else (1, True)

        move = possible[action][0]
        self.state.playLegalMove(move)
        done = self.state.checkEnd()
        if done:
            winner = self.state.getWinner()
            return (1, True) if winner == self.agent_id else (-1, True)

        return 0, False

    def _get_obs(self):
        grid = self.state.Grid
        obs = np.zeros((2 * SIZE**3,), dtype = np.int8)
        for x in range(SIZE):
            for y in range(SIZE):
                for z in range(SIZE):
                    v = grid[x][y][z]
                    if v is not None:
                        idx = v * (SIZE**3) + x * (SIZE**2) + y * SIZE + z
                        obs[idx] = 1
        return obs

    def render(self, mode = 'human'):
        print(self.state.Grid)


class IllegalMoveCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.illegal = 0
        self.total = 0

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if info.get("illegal_action", False):
                self.illegal += 1
            self.total += 1

        if self.total > 0 and self.num_timesteps % self.model.n_steps == 0:
            ratio = self.illegal / self.total
            print(f"→ Illegal logging ratio : {ratio:.3f} ({self.illegal}/{self.total})")

        return True


if __name__ == "__main__":

    env = ScoreFourEnv(opponent_type = 'random')
    model = PPO("MlpPolicy", env, verbose = 1, device = "cuda")
    env.model = model

    phases = [
        ('random', 1_000_000),
        ('selfplay', 1_000_000),
    ]

    '''The search tree calculations are too long at the moment. The search tree need to be reworked for use as a training rival to the PPO
    phases = [
        ('random', 200_000),
        ('searchTree1', 200_000),
        ('selfplay', 200_000),
        ('mixed', 200_000),
        ('searchTree3', 200_000),
    ]'''

    # Curriculum learning
    for name, timesteps in phases:
        print(f"--- Phase: {name} ({timesteps} timesteps) ---")
        env = ScoreFourEnv(opponent_type = name)
        model.set_env(env)
        env.model = model
        model.learn(total_timesteps = timesteps,
                    reset_num_timesteps = False,
                    callback = IllegalMoveCallback())
        model.save(f"ppo_score_four_{name}")

    model.save('ppo_score_four_final')
