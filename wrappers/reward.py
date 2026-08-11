import gymnasium as gym

class RewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.set_previous_state_values()
        self.has_previous_state = False
        self.max_x = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.set_previous_state_values(
            info.get("lives", 0),
            info.get("coins", 0),
            info.get("score", 0),
            info.get("x", 0),
            info.get("level_end_timer", 0),
            self.get_current_time(info),
        )
        self.has_previous_state = "x" in info
        self.max_x = info.get("x", 0)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_lives = info.get("lives", self.prev_lives)
        current_coins = info.get("coins", self.prev_coins)
        current_score = info.get("score", self.prev_score)
        current_x = info.get("x", self.prev_x)  
        current_level_end_timer = info.get("level_end_timer", self.prev_level_end_timer)
        current_timer = self.get_current_time(info)

        if not self.has_previous_state:
            self.set_previous_state_values(
                current_lives, 
                current_coins, 
                current_score, 
                current_x,
                current_level_end_timer,
                current_timer,
            )
            self.has_previous_state = True
            return obs, 0.0, terminated, truncated, info

        if self.is_lost_life(current_lives):
            reward -= 5.0 # Death penalty
            terminated = True
        else:
            if current_x > self.max_x:
                self.max_x = current_x
                reward += 0.05 # Progress reward

        if self.increase_score(current_score):
            if self.has_collected_coin(current_coins):
                reward += 0.2 # Coin reward
            else:
                reward += 0.05 # Score reward

        if self.times_up(current_timer):
            reward -= 10.0 # Times up penalty
            terminated = True
        elif current_timer < 100:
            reward -= 0.02 # hurry up mario!
        else:
            reward -= 0.01 # Timer penalty every step

        if self.level_completed(current_level_end_timer):
            reward += 100.0 # Level completion reward
            terminated = True

        self.set_previous_state_values(
            current_lives, 
            current_coins, 
            current_score, 
            current_x,
            current_level_end_timer,
            current_timer,
        )

        return obs, reward, terminated, truncated, info

    def is_lost_life(self, current_lives):
        return current_lives < self.prev_lives

    def has_collected_coin(self, current_coins):
        return current_coins > self.prev_coins

    def increase_score(self, current_score):
        return current_score > self.prev_score

    def get_current_time(self, info):
        hundreds = info.get("timer_hundreds", 0)
        tens = info.get("timer_tens", 0)
        ones = info.get("timer_ones", 0)
        return hundreds * 100 + tens * 10 + ones

    def set_previous_state_values(
        self, 
        lives=0, 
        coins=0, 
        score=0, 
        x=0,
        level_end_timer=0,
        timer=0,
    ):
        self.prev_lives = lives
        self.prev_coins = coins
        self.prev_score = score
        self.prev_x = x
        self.prev_level_end_timer = level_end_timer
        self.prev_timer = timer

    def times_up(self, current_timer):
        return self.prev_timer > 0 and current_timer == 0

    def level_completed(self, current_level_end_timer):
        return self.prev_level_end_timer==0 and current_level_end_timer>0
