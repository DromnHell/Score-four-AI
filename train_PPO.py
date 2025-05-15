import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from GameState import GameState, SIZE

class ScoreFourEnv(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self):
        super().__init__()
        # Observation: two channels (agent vs opponent), each channel is a flatten 4*4*4
        self.observation_space = gym.spaces.Box(low = 0, high = 1, shape = (2 * SIZE**3,), dtype = np.int8)
        # Action: select one of the (x,y) columns from SIZE*SIZE
        self.action_space = gym.spaces.Discrete(SIZE * SIZE)
        self.agent_id = 0
        self.model = None
        self.state = None
        self.reset()

    def reset(self, **kwargs):
        self.state = GameState()
        return self._get_obs(), {}

    def step(self, action_agent):

        # Penalization of illegal moves
        possible = self.state.getPossibleMoves()  # [ ((x,y,z), orig_idx), … ]
        orig_indices = [orig for (_m, orig) in possible]
        if action_agent not in orig_indices:
            return self._get_obs(), -1.0, True, False, {'winner': None, 'illegal_action': True}

        # Agent move execution
        reward_agent, done_agent = self._execute_move(action_agent, self.agent_id)
        obs_after_agent = self._get_obs()
        if done_agent:
            return obs_after_agent, reward_agent, done_agent, False, {'winner': self.agent_id, 'illegal_action': False}

        # Rival move execution (self-play with the same politic, but with stochastic choices)
        rival_action, _ = self.model.predict(obs_after_agent, deterministic = False)
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
    env = ScoreFourEnv()
    model = PPO("MlpPolicy", env, verbose = 1)
    env.model = model

    model.learn(total_timesteps = 1000000, callback = IllegalMoveCallback())

    model.save("ppo_score_four")