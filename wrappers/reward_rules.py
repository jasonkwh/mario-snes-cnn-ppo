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

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the rule."""


class CompletionRule(RewardRule):
    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        if prev_state.is_level_completed(cur_state):
            return 100.0, True
        return 0.0, False

    def get_name(self) -> str:
        return "completion"


class TimeUpPenaltyRule(RewardRule):
    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        if prev_state.is_times_up(cur_state):
            return -10.0, True
        return 0.0, False

    def get_name(self) -> str:
        return "time_up"


class LifeLostPenaltyRule(RewardRule):
    def apply(
        self,
        prev_state: RewardState,
        cur_state: RewardState,
    ) -> tuple[float, bool]:
        if prev_state.is_lost_life(cur_state):
            return -5.0, True
        return 0.0, False

    def get_name(self) -> str:
        return "lost_life"


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

    def get_name(self) -> str:
        return "score_event"


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

    def get_name(self) -> str:
        return "progress"
