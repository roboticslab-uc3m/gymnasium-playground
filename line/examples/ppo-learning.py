import gymnasium as gym
import gymnasium_playground_line

from stable_baselines3 import PPO # DQN coming soon

# check environnment
env = gym.make('gymnasium_playground/Line-v0', random_init=True)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./tensorboard/",learning_rate=0.01).learn(10000)

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
