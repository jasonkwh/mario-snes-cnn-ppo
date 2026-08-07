import sys
import subprocess
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from cleanup import close_env
from environment import make_env

from config import (
    GAME_NAME, STATE_NAME, MODEL_NAME, VIDEO_DIR, BEST_MODEL_SAVE_DIR, TENSORBOARD_LOG_DIR,
    CHECKPOINT_DIR, MONITOR_FILENAME, MONITOR_EVALUATION_FILENAME, LEARNING_RATE, N_STEPS, 
    BATCH_SIZE, ENT_COEF, FRAME_STACK, TOTAL_TIMESTEPS, EVAL_FREQ, N_EVAL_EPISODES, 
    CHECKPOINT_EVERY,
)

def main():
    env = None
    eval_env = None

    try:
        result = subprocess.run(
            [sys.executable, "-m", "stable_retro.import", "."],
            check=True,
            capture_output=True,
            text=True,
        )

        print(result.stdout)

        env = DummyVecEnv([lambda: make_env(GAME_NAME, STATE_NAME, VIDEO_DIR, MONITOR_FILENAME)])
        env = VecFrameStack(env, n_stack=FRAME_STACK, channels_order="last")
        env = VecTransposeImage(env)

        eval_env = SubprocVecEnv([
            lambda: make_env(
                GAME_NAME,
                STATE_NAME,
                video_folder=f"{VIDEO_DIR}/evaluation",
                monitor_filename=MONITOR_EVALUATION_FILENAME,
            )
        ], start_method="spawn")
        eval_env = VecFrameStack(eval_env, n_stack=FRAME_STACK, channels_order="last")
        eval_env = VecTransposeImage(eval_env)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = PPO(
            policy="CnnPolicy",
            env=env,
            device=device,
            verbose=1,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            ent_coef=ENT_COEF,
            tensorboard_log=TENSORBOARD_LOG_DIR,
        )

        print(f"Training {GAME_NAME} Agent on {device.upper()}...")

        callbacks = [
            CheckpointCallback(
                save_freq=CHECKPOINT_EVERY,
                save_path=CHECKPOINT_DIR,
                name_prefix=MODEL_NAME,
                verbose=2,
            ),
            EvalCallback(
                eval_env,
                best_model_save_path=BEST_MODEL_SAVE_DIR,
                log_path=BEST_MODEL_SAVE_DIR,
                eval_freq=EVAL_FREQ,
                n_eval_episodes=N_EVAL_EPISODES,
                deterministic=True,
                verbose=2,
                render=False,
            ),
        ]

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
        )

        model.save(MODEL_NAME)
        print(f"Model saved successfully as '{MODEL_NAME}.zip'")
    finally:
        if env is not None:
            close_env(env)
        if eval_env is not None:
            close_env(eval_env)

if __name__ == "__main__":
    main()
