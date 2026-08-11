import gymnasium as gym
from .reward_types import RewardState

class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_state: RewardState | None = None
        self.max_x = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_state = RewardState.from_info(info) if "x" in info else None
        self.max_x = info.get("x", 0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        cur_state = RewardState.from_info(info)

        if self.prev_state is None:
            self.prev_state = cur_state
            self.max_x = cur_state.x
            return obs, 0.0, terminated, truncated, info

        if self.prev_state.is_lost_life(cur_state):
            reward -= 5.0 # Death penalty
            terminated = True
        else:
            if cur_state.x > self.max_x:
                self.max_x = cur_state.x
                reward += 0.05 # Progress reward

        if self.prev_state.has_increased_score(cur_state):
            if self.prev_state.has_collected_coin(cur_state):
                reward += 0.2 # Coin reward
            else:
                reward += 0.05 # Score reward

        if self.prev_state.is_times_up(cur_state):
            reward -= 10.0 # Times up penalty
            terminated = True
        elif cur_state.timer < 100:
            reward -= 0.02 # hurry up mario!
        else:
            reward -= 0.01 # Timer penalty every step

        if self.prev_state.is_level_completed(cur_state):
            reward += 100.0 # Level completion reward
            terminated = True

        self.prev_state = cur_state

        return obs, reward, terminated, truncated, info
