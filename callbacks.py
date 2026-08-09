from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from environment import MarioEnvironment
from config import (
    FRAME_STACK,
    GAME_NAME,
    STATE_NAME,
    SEED,
)

class RecordVideoAtBestModelCallback(BaseCallback):
    # when the new best is found
    def _on_step(self) -> bool:
        video_env = DummyVecEnv([
            lambda: MarioEnvironment(
                GAME_NAME,
                STATE_NAME,
                monitor_filename=None,
                record_video=True,
            )
        ])
        video_env = VecFrameStack(
            video_env, 
            n_stack=FRAME_STACK, 
            channels_order="last",
        )
        video_env = VecTransposeImage(video_env)
        video_env.seed(SEED + 20_000) # different seed for video environment

        try:
            obs = video_env.reset()
            done = [False]
            while not done[0]:
                action, _ = self.model.predict(
                    obs,
                    deterministic=True,
                )
                obs, _, done, _ = video_env.step(action)
        finally:
            video_env.close()

        return True
