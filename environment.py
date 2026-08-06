import gymnasium as gym
import stable_retro
from reward import RewardWrapper
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
from stable_baselines3.common.monitor import Monitor

def make_env(
    game_name,
    state_name,
    record_video=False,
    video_folder="./mario_videos",
    monitor_filename="./monitor.csv"):
    env = stable_retro.make(
        game=game_name,
        state=state_name,
        use_restricted_actions=stable_retro.Actions.DISCRETE,
        render_mode="rgb_array",
    )

    env = RewardWrapper(env)
    env = MaxAndSkipEnv(env, skip=4)
    env = gym.wrappers.ResizeObservation(env, shape=(84, 96))
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=True)

    # Add RecordVideo as the final wrapper
    if record_video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            episode_trigger=lambda episode_id: episode_id % 100 == 0 # Records every 100th episode
        )

    env = Monitor(env, filename=monitor_filename)

    return env