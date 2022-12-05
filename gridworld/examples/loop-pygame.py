#!/usr/bin/env python

import gymnasium as gym
import gymnasium_playground_gridworld

import numpy as np 
import time

# X points down (rows)(v), Y points right (columns)(>), Z would point outwards.
LEFT = 0  # < Decrease Y (column)
DOWN = 1  # v Increase X (row)
RIGHT = 2 # > Increase Y (column)
UP = 3    # ^ Decrease X (row)
CORNER_UP_RIGHT=4
CORNER_UP_LEFT=5
CORNER_DOWN_RIGHT=6
CORNER_DOWN_LEFT=7

SIM_PERIOD_MS = 500.0

env = gym.make('gymnasium_playground/GridWorld-v0', render_mode='pygame')
state = env.reset()
print("state: "+str(state))
env.render()
time.sleep(0.5)

for i in range(4):
    observation, reward, terminated, truncated, info = env.step(CORNER_DOWN_RIGHT)
    env.render()
    print("observation: "+str(observation)+", reward: "+str(reward)+", terminated: "+str(terminated))
    time.sleep(SIM_PERIOD_MS/1000.0)
