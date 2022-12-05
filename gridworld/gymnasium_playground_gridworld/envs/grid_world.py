import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

# X points down (rows)(v), Y points right (columns)(>), Z would point outwards.
LEFT = 0  # < Decrease Y (column)
DOWN = 1  # v Increase X (row)
RIGHT = 2 # > Increase Y (column)
UP = 3    # ^ Decrease X (row)
CORNER_UP_RIGHT=4
CORNER_UP_LEFT=5
CORNER_DOWN_RIGHT=6
CORNER_DOWN_LEFT=7

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

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

        self.nrow, self.ncol = nrow, ncol = self.inFile.shape
        self.nS = nrow * ncol # nS: number of states
        self.observation_space = spaces.Discrete(self.nS)

        self.nA = 8 # nA: number of actions
        self.action_space = spaces.Discrete(self.nA)

        self.P = {s : {a : [] for a in range(self.nA)} for s in range(self.nS)} # transitions (*), filled in at the for loop below.

        def _to_s(row, col):
            return row*ncol + col

        self.initial_state = _to_s(initX, initY)

        def _inc(row, col, a): # Assures we will not go off limits.
            if a == LEFT:
                col = max(col-1,0)
            elif a == DOWN:
                row = min(row+1,nrow-1)
            elif a == RIGHT:
                col = min(col+1,ncol-1)
            elif a == UP:
                row = max(row-1,0)
            elif a == CORNER_UP_RIGHT:
                row = max(row-1,0)
                col = min(col+1,ncol-1)
            elif a == CORNER_UP_LEFT:
                row = max(row-1,0)
                col = max(col-1,0)
            elif a == CORNER_DOWN_RIGHT:
                row = min(row+1,nrow-1)
                col = min(col+1,ncol-1)
            elif a == CORNER_DOWN_LEFT:
                row = min(row+1,nrow-1)
                col = max(col-1,0)
            return (row, col)

        for row in range(nrow): # Fill in P[s][a] transitions and rewards
            for col in range(ncol):
                s = _to_s(row, col)
                for a in range(self.nA):
                    li = self.P[s][a] # In Python this is not a deep copy, therefore we are appending to actual P[s][a] !!
                    tag = self.inFile[row][col]
                    if tag == 3: # goal
                        li.append((1.0, s, 1.0, True)) # (probability, nextstate, reward, done)
                    elif tag == 1: # wall
                        li.append((1.0, s, -0.5, True)) # (probability, nextstate, reward, done) # Some algorithms fail with reward -float('inf')
                    else: # e.g. tag == 0
                        newrow, newcol = _inc(row, col, a)
                        newstate = _to_s(newrow, newcol)
                        li.append((1.0, newstate, 0.0, False)) # (probability, nextstate, reward, done)


    def _get_obs(self):
        return self.s

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random.
        super().reset(seed=seed)

        self.s = self.initial_state

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        pass

    def render(self):
        pass

    def close(self):
        pass
