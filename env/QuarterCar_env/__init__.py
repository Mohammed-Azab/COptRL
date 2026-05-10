from gymnasium.envs.registration import register

register(
    id="QuarterCar_env/QuarterCar-v0",
    entry_point="QuarterCar_env.envs:QuarterCarEnv",
)
