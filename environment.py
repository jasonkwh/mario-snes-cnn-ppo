import gymnasium as gym
import stable_retro
from reward import RewardWrapper
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.monitor import Monitor

from config import (
    RECORD_VIDEO, RECORD_VIDEO_EVERY,
    FRAME_SKIP, OBSERVATION_SHAPE, VIDEO_RENDER_FPS,
)

def make_env(
    game_name,
    state_name,
    video_folder="./mario_videos/",
    monitor_filename="./monitor.csv"):
    env = stable_retro.make(
        game=game_name,
        state=state_name,
        use_restricted_actions=stable_retro.Actions.DISCRETE,
        render_mode="rgb_array",
    )

    env = RewardWrapper(env)
    env = MaxAndSkipEnv(env, skip=FRAME_SKIP)
    env = gym.wrappers.ResizeObservation(env, shape=OBSERVATION_SHAPE)
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)

    if RECORD_VIDEO:
        env.metadata["render_fps"] = VIDEO_RENDER_FPS

        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda episode_id: episode_id % RECORD_VIDEO_EVERY == 0 # Records every RECORD_VIDEO_EVERYth episode
        )

    env = Monitor(env, filename=monitor_filename)

    return env
