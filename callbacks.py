from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)
from environment import MarioEnvironment
from config import (
    BEST_MODEL_SAVE_DIR,
    FRAME_STACK,
    GAME_NAME,
    STATE_NAME,
    SEED,
)


class SaveVecNormalizeAtBestModelCallback(BaseCallback):
    def _on_step(self) -> bool:
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is None:
            return True

        save_path = Path(BEST_MODEL_SAVE_DIR)
        save_path.mkdir(parents=True, exist_ok=True)
        vec_normalize.save(save_path / "best_model_vecnormalize.pkl")
        return True


class RecordVideoAtBestModelCallback(BaseCallback):
    # when the new best is found
    def _on_step(self) -> bool:
        video_env = SubprocVecEnv(
            [
                lambda: MarioEnvironment(
                    GAME_NAME,
                    STATE_NAME,
                    monitor_filename=None,
                    record_video=True,
                    seed=SEED + 20_000,
                )
            ],
            start_method="spawn",
        )
        video_env = VecFrameStack(
            video_env,
            n_stack=FRAME_STACK,
            channels_order="last",
        )
        video_env = VecTransposeImage(video_env)

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
