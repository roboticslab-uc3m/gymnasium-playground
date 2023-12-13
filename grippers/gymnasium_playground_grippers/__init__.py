from gymnasium.envs.registration import register

register(
    id="gymnasium_playground/Grippers-v0",
    entry_point="gymnasium_playground_grippers.envs:GrippersEnv",
)
