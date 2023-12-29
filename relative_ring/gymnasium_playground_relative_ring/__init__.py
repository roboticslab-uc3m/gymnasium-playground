from gymnasium.envs.registration import register

register(
    id="gymnasium_playground/RelativeRing-v0",
    entry_point="gymnasium_playground_relative_ring.envs:RelativeRingEnv",
)
