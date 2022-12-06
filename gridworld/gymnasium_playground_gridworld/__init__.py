from gymnasium.envs.registration import register

register(
    id="gymnasium_playground/GridWorld-v0",
    entry_point="gymnasium_playground_gridworld.envs:GridWorldEnv",
)
