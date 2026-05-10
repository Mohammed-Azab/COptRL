import gymnasium as gym


class RewardScaler(gym.RewardWrapper):
    """Scale reward by a constant."""

    def __init__(self, env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward: float) -> float:
        return reward * self.scale
