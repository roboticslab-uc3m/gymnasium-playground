import gymnasium as gym
import gymnasium_playground_grippers

from stable_baselines3.common.env_checker import check_env

# check environnment
env = gym.make('gymnasium_playground/Grippers-v0')
check_env(env, warn=True)