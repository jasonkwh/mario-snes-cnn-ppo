import sys
import subprocess
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from environment import MarioEnvironment
from helpers import get_checkpoint_path, get_best_model_path, get_n_envs
from setup import setup_game
from callbacks import RecordVideoAtBestModelCallback

from config import (
    GAME_NAME, STATE_NAME, MODEL_NAME, BEST_MODEL_SAVE_DIR, TENSORBOARD_LOG_DIR,
    CHECKPOINT_DIR, MONITOR_FILENAME, MONITOR_EVALUATION_FILENAME, LEARNING_RATE, N_STEPS, 
    BATCH_SIZE, ENT_COEF, FRAME_STACK, TOTAL_TIMESTEPS, EVAL_FREQ, N_EVAL_EPISODES, 
    CHECKPOINT_EVERY, SEED,
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

        env = SubprocVecEnv([
            lambda rank=rank: MarioEnvironment(
                GAME_NAME,
                STATE_NAME,
                monitor_filename=MONITOR_FILENAME.replace(".csv", f"-{rank}.csv"),
                seed=SEED + rank,
            )
            for rank in range(get_n_envs())
        ])
        env = VecFrameStack(env, n_stack=FRAME_STACK, channels_order="last")
        env = VecTransposeImage(env)

        eval_env = SubprocVecEnv([
            lambda: MarioEnvironment(
                GAME_NAME,
                STATE_NAME,
                monitor_filename=MONITOR_EVALUATION_FILENAME,
                seed=SEED + 10_000,
            )
        ], start_method="spawn")
        eval_env = VecFrameStack(eval_env, n_stack=FRAME_STACK, channels_order="last")
        eval_env = VecTransposeImage(eval_env)

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # load checkpoint if it exists
        checkpoint_path = get_checkpoint_path()

        if checkpoint_path is not None:
            model = PPO.load(
                checkpoint_path,
                env=env,
                device=device,
                tensorboard_log=TENSORBOARD_LOG_DIR,
            )
            reset_num_timesteps = False
        else:
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
                seed=SEED,
            )
            reset_num_timesteps = True

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
                callback_on_new_best=RecordVideoAtBestModelCallback(),
            ),
        ]

        if model.num_timesteps < TOTAL_TIMESTEPS:
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS - model.num_timesteps,
                callback=callbacks,
                reset_num_timesteps=reset_num_timesteps,
                progress_bar=True,
            )
        else:
            print(f"Model has already reached {TOTAL_TIMESTEPS} timesteps. Skipping training.")

        best_model_path = get_best_model_path()

        if best_model_path is not None:
            best_model = PPO.load(best_model_path, device=device)
            best_model.save(MODEL_NAME)
            print(f"Best evaluated model saved as '{MODEL_NAME}.zip'")
        else:
            model.save(MODEL_NAME)
            print(
                f"No evaluated best model was created; "
                f"last model saved as '{MODEL_NAME}.zip'"
            )
    finally:
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()

if __name__ == "__main__":
    setup_game()
    main()
