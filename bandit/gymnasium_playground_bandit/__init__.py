from gymnasium.envs.registration import register

register(
    id="gymnasium_playground/Bandit-v0",
    entry_point="gymnasium_playground_bandit.envs:BanditEnv",
)
