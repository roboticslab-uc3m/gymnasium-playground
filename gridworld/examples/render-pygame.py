#!/usr/bin/env python

import gymnasium as gym
import gymnasium_playground_gridworld

env = gym.make('gymnasium_playground/GridWorld-v0', render_mode='pygame')
env.reset()
env.render()
