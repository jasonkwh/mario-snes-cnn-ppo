import gymnasium as gym
from .reward_types import RewardState
from .reward_rules import (
    CompletionRule,
    FailurePenaltyRule,
    ScoreEventRule,
    ProgressRule,
)

class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_state: RewardState | None = None
        self.rules = [
            # order matters
            CompletionRule(),
            FailurePenaltyRule(),
            ScoreEventRule(),
            ProgressRule(),
        ]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_state = RewardState.from_info(info) if "x" in info else None
        
        for rule in self.rules:
            rule.reset(self.prev_state)

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        cur_state = RewardState.from_info(info)

        if self.prev_state is None:
            self.prev_state = cur_state

            for rule in self.rules:
                rule.reset(self.prev_state)

            return obs, 0.0, terminated, truncated, info

        for rule in self.rules:
            reward_delta, rule_terminated = rule.apply(self.prev_state, cur_state)
            reward += reward_delta

            if rule_terminated:
                terminated = True
                break

        self.prev_state = cur_state

        return obs, reward, terminated, truncated, info
