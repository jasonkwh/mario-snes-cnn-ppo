import gymnasium as gym

class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_lives = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_lives = info.get("lives", 0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_lives = info.get("lives", self.prev_lives)

        if self.is_lost_life(current_lives):
            reward -= 15.0 # Death penalty
            
        self.prev_lives = current_lives

        return obs, reward, terminated, truncated, info

    def is_lost_life(self, current_lives):
        return current_lives < self.prev_lives
