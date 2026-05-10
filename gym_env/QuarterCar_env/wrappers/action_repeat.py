import gymnasium as gym


class ActionRepeat(gym.Wrapper):
    """Repeat action n_repeat times, accumulate reward."""

    def __init__(self, env, n_repeat: int = 2):
        super().__init__(env)
        self.n_repeat = n_repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.n_repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info
