from gymnasium.envs.registration import register

register(
    id="gymnasium_playground/Ring-v0",
    entry_point="gymnasium_playground_ring.envs:RingEnv",
)
