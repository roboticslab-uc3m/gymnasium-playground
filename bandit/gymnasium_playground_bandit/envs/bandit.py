import gymnasium as gym
import numpy as np
from gymnasium import spaces


class BanditEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        pass

    def _get_obs(self):
        pass

    def _get_info(self):
        pass

    def reset(self, seed=None, options=None):
        pass

    def step(self, action):
        pass

    def render(self):
        pass

    def close(self):
        pass
