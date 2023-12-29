import gymnasium as gym
import gymnasium_playground_relative_ring
import math

from stable_baselines3 import PPO # DQN coming soon

from stable_baselines3.common.callbacks import BaseCallback

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback,self).__init__(verbose)

        self.reward =0

    def _on_step(self) -> bool:
        self.logger.record("step_reward", self.training_env.get_attr('current_reward')[0])
        return True

# check environnment
env = gym.make('gymnasium_playground/RelativeRing-v0', random_init=True, goal_difference=math.pi)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/",learning_rate=0.01, gamma=0.99).learn(100000,callback=TensorboardCallback())

n_steps = 20
n_epis = 5

for epi in range(n_epis):
    obs, _ = env.reset()
    env.render()

    for step in range(n_steps):
        action, _ = model.predict(obs, deterministic=True)
        print("Step {}".format(step + 1))
        print("Action: ", action)
        obs, reward, done, trucated, info = env.step(action)
        env.render()
        print("reward: ",reward )

        if done:
            print("*_*_*_*_*_*_*_*_*_*_*_*_")
            print("Goal reached!", "reward=", reward)
            print("*_*_*_*_*_*_*_*_*_*_*_*_")
            break