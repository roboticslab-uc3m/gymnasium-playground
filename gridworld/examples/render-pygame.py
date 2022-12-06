#!/usr/bin/env python

import gymnasium as gym
import gymnasium_playground_gridworld

env = gym.make('gymnasium_playground/GridWorld-v0', render_mode='pygame',
               inFileStr='map1.csv', initX=2, initY=2, goalX=7, goalY=2)
env.reset()
env.render()
