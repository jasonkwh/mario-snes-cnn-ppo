from uuid import uuid4

import gymnasium as gym
import stable_retro
from wrappers import ActionWrapper, RewardWrapper
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.monitor import Monitor

from config import (
    VIDEO_DIR, RECORD_VIDEO_EVERY,
    FRAME_SKIP, OBSERVATION_SHAPE, VIDEO_RENDER_FPS,
    MAX_EPISODE_STEPS,
)

class MarioEnvironment(gym.Wrapper):
    def __init__(
        self,
        game_name,
        state_name,
        monitor_filename=None,
        record_video=False,
    ):
        env = stable_retro.make(
            game=game_name,
            state=state_name,
            use_restricted_actions=stable_retro.Actions.ALL,
            render_mode="rgb_array",
        )

        env = ActionWrapper(env)
        env = RewardWrapper(env)
        env = MaxAndSkipEnv(env, skip=FRAME_SKIP)
        env = gym.wrappers.ResizeObservation(env, shape=OBSERVATION_SHAPE)
        env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)
        env = gym.wrappers.TimeLimit(
            env,
            max_episode_steps=MAX_EPISODE_STEPS,
        )

        if record_video:
            env = gym.wrappers.RecordVideo(
                env,
                video_folder=VIDEO_DIR,
                episode_trigger=lambda episode_id: episode_id % RECORD_VIDEO_EVERY == 0,
                fps=VIDEO_RENDER_FPS,
                name_prefix=f"model-{uuid4().hex}",
            )

        super().__init__(
            Monitor(
                env, 
                filename=monitor_filename if monitor_filename else None,
            )
        )
