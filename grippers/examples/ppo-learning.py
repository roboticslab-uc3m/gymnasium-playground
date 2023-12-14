import gymnasium as gym
import gymnasium_playground_grippers

from stable_baselines3 import PPO # DQN coming soon

# check environnment
env = gym.make('gymnasium_playground/Grippers-v0')

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/",learning_rate=0.01).learn(10000)

obs, _ = env.reset()
env.render()

n_steps = 10

for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print("Step {}".format(step + 1))
    print("Action: ", action)
    obs, reward, done, trucated, info = env.step(action)
    env.render()
    print("reward: ",reward )

    if done:
        print("Goal reached!", "reward=", reward)
        break
