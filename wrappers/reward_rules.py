from abc import ABC, abstractmethod
from .reward_types import RewardState


class RewardRule(ABC):
    def reset(self, prev_state: RewardState | None = None) -> None:
        """Reset the rule state."""
        pass

    @abstractmethod
    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        """Return (reward_delta, terminated)."""

    @property
    def name(self) -> str:
        return type(self).__name__.removesuffix("Rule")

    def get_metrics(self) -> dict[str, float]:
        return {}


class CompletionRule(RewardRule):
    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        if prev_state.is_level_completed(cur_state):
            return 100.0, True
        return 0.0, False


class TimeUpPenaltyRule(RewardRule):
    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        if prev_state.is_times_up(cur_state):
            return -10.0, True
        return 0.0, False


class LifeLostPenaltyRule(RewardRule):
    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        if prev_state.is_lost_life(cur_state):
            return -5.0, True
        return 0.0, False


class ScoreEventRule(RewardRule):
    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        if prev_state.has_increased_score(cur_state):
            if prev_state.has_collected_coin(cur_state):
                return 0.2, False
            return 0.05, False
        return 0.0, False


class PowerUpRule(RewardRule):
    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        if prev_state.becomes_big_mario(cur_state):
            return cur_state.powerup / 10, False
        if prev_state.becomes_small_mario(cur_state):
            if prev_state.is_lost_life(cur_state):
                return 0.0, False
            return -0.1, False
        return 0.0, False


class ProgressRule(RewardRule):
    def __init__(self):
        self.max_x = 0

    def reset(self, prev_state: RewardState | None = None) -> None:
        self.max_x = prev_state.x if prev_state is not None else 0

    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        reward_delta = 0.0

        # time based penalty, every step
        if cur_state.timer < 100:
            reward_delta -= 0.02
        else:
            reward_delta -= 0.01

        # progress reward, if x increased
        if cur_state.x > self.max_x:
            self.max_x = cur_state.x
            reward_delta += 0.05

        return reward_delta, False

    def get_metrics(self) -> dict[str, float]:
        return {
            "max_x": float(self.max_x),
        }
