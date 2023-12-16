from gymnasium.envs.registration import register

register(
    id="gymnasium_playground/Line-v0",
    entry_point="gymnasium_playground_line.envs:LineEnv",
)
