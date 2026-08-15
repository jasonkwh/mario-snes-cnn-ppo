from dataclasses import dataclass


@dataclass(frozen=True)
class RewardState:
    lives: int
    coins: int
    score: int
    x: int
    level_end_timer: int
    timer: int
    powerup: int

    @classmethod
    def from_info(cls, info: dict) -> "RewardState":
        return cls(
            lives=info.get("lives", 0),
            coins=info.get("coins", 0),
            score=info.get("score", 0),
            x=info.get("x", 0),
            level_end_timer=info.get("level_end_timer", 0),
            timer=cls._get_current_time(info),
            powerup=info.get("powerup", 0),
        )

    @staticmethod
    def _get_current_time(info: dict) -> int:
        hundreds = info.get("timer_hundreds", 0)
        tens = info.get("timer_tens", 0)
        ones = info.get("timer_ones", 0)
        return hundreds * 100 + tens * 10 + ones

    def is_times_up(self, current: "RewardState") -> bool:
        return self.timer > 0 and current.timer == 0

    def is_level_completed(self, current: "RewardState") -> bool:
        return self.level_end_timer == 0 and current.level_end_timer > 0

    def is_lost_life(self, current: "RewardState") -> bool:
        return current.lives < self.lives

    def has_collected_coin(self, current: "RewardState") -> bool:
        return current.coins > self.coins

    def has_increased_score(self, current: "RewardState") -> bool:
        return current.score > self.score

    def becomes_big_mario(self, current: "RewardState") -> bool:
        return current.powerup != 0 and self.powerup == 0

    def becomes_small_mario(self, current: "RewardState") -> bool:
        return current.powerup == 0 and self.powerup != 0
