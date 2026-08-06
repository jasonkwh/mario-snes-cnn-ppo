import sys
import subprocess
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback
from cleanup import close_env
from environment import make_env

from config import (
    GAME_NAME, STATE_NAME, MODEL_NAME, VIDEO_DIR,
    CHECKPOINT_DIR, MONITOR_FILENAME,
)

def main():
    env = None

    try:
        result = subprocess.run(
            [sys.executable, "-m", "stable_retro.import", "."],
            check=True,
            capture_output=True,
            text=True,
        )

        print(result.stdout)

        env = DummyVecEnv([lambda: make_env(GAME_NAME, STATE_NAME, VIDEO_DIR, MONITOR_FILENAME)])
        env = VecFrameStack(env, n_stack=4, channels_order="last")
        env = VecTransposeImage(env)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = PPO(
            policy="CnnPolicy",
            env=env,
            device=device,
            verbose=1,
            learning_rate=0.0001,
            n_steps=2048,
            batch_size=64,
            ent_coef=0.01,
        )

        print(f"Training {GAME_NAME} Agent on {device.upper()}...")
        model.learn(
            total_timesteps=1000000,
            callback=CheckpointCallback(
                save_freq=50000,
                save_path=CHECKPOINT_DIR,
                name_prefix=MODEL_NAME,
                verbose=2
            )
        )

        model.save(MODEL_NAME)
        print(f"Model saved successfully as '{MODEL_NAME}.zip'")
    finally:
        if env is not None:
            close_env(env)

if __name__ == "__main__":
    main()
