#!/usr/bin/env python

import gymnasium as gym
import gymnasium_playground_ring

import numpy as np
import random as rng
import time
import math

env = gym.make('gymnasium_playground/Ring-v0', render_mode=None, random_init=False,  init_angle=2*math.pi -5*0.1, goal_angle=0.3)
observation, info = env.reset()
print("observation: "+str(observation)+", info: "+str(info))
env.render()

for i in range(10):
    #ac = rng.uniform(-1,1)
    ac = 1
    observation, reward, terminated, truncated, info = env.step(ac)
    env.render()
    print("observation: " + str(observation)+", reward: " + str(reward) + ", terminated: " +
          str(terminated) + ", truncated: " + str(truncated) + ", info: " + str(info))
