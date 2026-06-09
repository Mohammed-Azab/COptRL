import gymnasium as gym


class RewardScaler(gym.RewardWrapper):
    # scales every reward by a fixed constant

    def __init__(self, env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward: float) -> float:
        return reward * self.scale
