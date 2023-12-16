#!/usr/bin/env python

import gymnasium as gym
import gymnasium_playground_line

import numpy as np
import random as rng
import time

env = gym.make('gymnasium_playground/Line-v0')
observation, info = env.reset()
print("observation: "+str(observation)+", info: "+str(info))
env.render()

for i in range(10):
    ac = rng.uniform(-1,1)
    observation, reward, terminated, truncated, info = env.step(ac)
    env.render()
    print("observation: " + str(observation)+", reward: " + str(reward) + ", terminated: " +
          str(terminated) + ", truncated: " + str(truncated) + ", info: " + str(info))
