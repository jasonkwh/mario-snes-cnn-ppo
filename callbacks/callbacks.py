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


class TensorboardLogExtensionCallback(BaseCallback):
    def _on_step(self) -> bool:
        for env_index, info in enumerate(self.locals.get("infos", [])):
            for key, value in info.items():
                if key in ("powerup", "item"):
                    self.logger.record(
                        f"env_{env_index}/state/{key}",
                        int(value),
                    )

                elif key in ("x", "y", "timer"):
                    self.logger.record(f"env_{env_index}/state/{key}", float(value))
                    self.logger.record_mean(f"state/{key}", float(value))

                elif key.startswith("reward/") or key.startswith("termination/"):
                    self.logger.record(f"env_{env_index}/{key}", float(value))
                    self.logger.record_mean(key, float(value))

        return True


class SaveVecNormalizeCallback(BaseCallback):
    def __init__(
        self,
        save_dir: Path | str = Path(BEST_MODEL_SAVE_DIR),
        name_prefix: str = "best_model",
    ):
        super().__init__()
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = save_dir / f"{name_prefix}_vecnormalize.pkl"

    def _on_step(self) -> bool:
        vec_normalize = self.model.get_vec_normalize_env()
        if vec_normalize is None:
            return True

        vec_normalize.save(self.save_path)
        return True


class RecordVideoCallback(BaseCallback):
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
