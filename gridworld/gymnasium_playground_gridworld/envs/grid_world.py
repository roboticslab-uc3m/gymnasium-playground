import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        # Remember: X points down, Y points right, thus Z points outwards.
        # hard-coded vars (begin)
        inFileStr = 'map1.csv'
        initX = 2
        initY = 2
        goalX = 7
        goalY = 2
        # hard-coded vars (end)
        self.inFile = np.genfromtxt(inFileStr, delimiter=',')
        self.inFile[goalX][goalY] = 3 # The goal (3) is fixed, so we paint it, but the robot (2) moves, so done at render().
        nrow, ncol = self.inFile.shape
        self.observation_space = spaces.Discrete(nrow * ncol)

        # We have 4 actions, corresponding to "right", "up", "left", "down".
        self.action_space = spaces.Discrete(4)

    def _get_obs(self):
        pass

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random.
        super().reset(seed=seed)

    def step(self, action):
        pass

    def render(self):
        pass

    def close(self):
        pass
