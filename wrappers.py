import gymnasium as gym
import numpy as np

class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.set_previous_state_values()
        self.has_previous_state = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.set_previous_state_values(
            info.get("lives", 0),
            info.get("coins", 0),
            info.get("score", 0),
            info.get("x", 0),
            info.get("level_end_timer", 0),
        )
        self.has_previous_state = "x" in info

        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_lives = info.get("lives", self.prev_lives)
        current_coins = info.get("coins", self.prev_coins)
        current_score = info.get("score", self.prev_score)
        current_x = info.get("x", self.prev_x)  
        current_level_end_timer = info.get("level_end_timer", self.prev_level_end_timer)

        if not self.has_previous_state:
            self.set_previous_state_values(
                current_lives, 
                current_coins, 
                current_score, 
                current_x,
                current_level_end_timer,
            )
            self.has_previous_state = True
            return obs, 0.0, terminated, truncated, info

        if self.is_lost_life(current_lives):
            reward -= 5.0 # Death penalty
        else:
            reward += self.x_changed(current_x) * 0.05 # Progress reward/penalty

        if self.increase_score(current_score):
            if self.has_collected_coin(current_coins):
                reward += 0.2 # Coin reward
            else:
                reward += 0.05 # Score reward

        if self.level_completed(current_level_end_timer):
            reward += 100.0 # Level completion reward
            terminated = True

        self.set_previous_state_values(
            current_lives, 
            current_coins, 
            current_score, 
            current_x,
            current_level_end_timer,
        )

        # print(f"reward: {reward}")
        # print(f"info: {info}")
        # print(f"obs: {obs}")

        return obs, reward, terminated, truncated, info

    def is_lost_life(self, current_lives):
        return current_lives < self.prev_lives

    def has_collected_coin(self, current_coins):
        return current_coins > self.prev_coins

    def increase_score(self, current_score):
        return current_score > self.prev_score

    def x_changed(self, current_x):
        return current_x - self.prev_x

    def set_previous_state_values(
        self, 
        lives=0, 
        coins=0, 
        score=0, 
        x=0,
        level_end_timer=0,
    ):
        self.prev_lives = lives
        self.prev_coins = coins
        self.prev_score = score
        self.prev_x = x
        self.prev_level_end_timer = level_end_timer

    def level_completed(self, current_level_end_timer):
        return self.prev_level_end_timer==0 and current_level_end_timer>0

class ActionWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.combos = [
            [], # do nothing
            ["LEFT"], # move left
            ["RIGHT"],
            ["B"], # jump
            ["A"], # spin jump
            ["DOWN"], # crouching
            ["RIGHT", "A"], # right spin jump
            ["LEFT", "A"], # left spin jump
            ["RIGHT", "B"], # right jump
            ["LEFT", "B"], # left jump
            ["Y"], # pick up item
            ["Y", "DOWN"], # pick up item + crouch
            ["Y", "B"], # jump + pick up item
            ["RIGHT", "Y"],    # run right, fire right
            ["LEFT", "Y"],    # run left, fire left
            ["RIGHT", "Y", "B"],    # run right + jump + fire
            ["LEFT", "Y", "B"],    # run left + jump + fire
            ["RIGHT", "Y", "A"],    # run right + spin jump + fire
            ["LEFT", "Y", "A"],    # run left + spin jump + fire
            # ["UP"], # up
            # ["UP", "B"], # jumping out of water
            # ["UP", "A"], # spin jump out of water / Yoshi dismount
            # ["UP", "Y"],
        ]
        buttons = env.unwrapped.buttons
        self.actions = []

        for combo in self.combos:
            action = np.zeros(env.action_space.n, dtype=np.uint8)
            for button in combo:
                action[buttons.index(button)] = 1
            self.actions.append(action)
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action):
        return self.actions[action].copy()