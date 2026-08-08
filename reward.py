import gymnasium as gym

class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_x = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_x = info.get("x", 0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        print(obs)
        print(reward)
        print(terminated)
        print(truncated)
        print(info)

        current_x = info.get("x", 0)
        x_reward = current_x - self.prev_x
        self.prev_x = current_x

        custom_reward = x_reward * 0.1
        if terminated:
            custom_reward -= 15.0  # Death penalty

        return obs, custom_reward, terminated, truncated, info
