from abc import ABC, abstractmethod
from .reward_types import RewardState


class ProgressEstimator(ABC):
    def estimate(self, state: RewardState | None = None) -> float:
        if state is None:
            return 0.0

        return self._estimate(state)

    @abstractmethod
    def _estimate(self, state: RewardState) -> float:
        pass


class XProgressEstimator(ProgressEstimator):
    def _estimate(self, state: RewardState) -> float:
        return float(state.x)


class YProgressEstimator(ProgressEstimator):
    def _estimate(self, state: RewardState) -> float:
        return float(state.y)
