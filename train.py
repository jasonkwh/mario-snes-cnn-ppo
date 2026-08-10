import sys
import subprocess
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from environment import MarioEnvironment
from helpers import get_checkpoint_path, get_best_model_path, get_n_envs
from setup import setup_game
from callbacks import RecordVideoAtBestModelCallback

from config import (
    GAME_NAME, STATE_NAME, MODEL_NAME, BEST_MODEL_SAVE_DIR, TENSORBOARD_LOG_DIR,
    CHECKPOINT_DIR, MONITOR_FILENAME, MONITOR_EVALUATION_FILENAME, LEARNING_RATE, N_STEPS, 
    BATCH_SIZE, ENT_COEF, FRAME_STACK, TOTAL_TIMESTEPS, EVAL_EVERY, N_EVAL_EPISODES, 
    CHECKPOINT_EVERY, SEED, N_EVAL_EPISODES_FINAL,
)

def final_evaluation(model, eval_env):
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=N_EVAL_EPISODES_FINAL,
        deterministic=True,
    )
    print(f"Final evaluation: {mean_reward:.2f} +/- {std_reward:.2f}")

def get_model_to_save(model, device):
    best_model_path = get_best_model_path()

    if best_model_path is not None:
        best_model = PPO.load(best_model_path, device=device)
        print("Using the best model selected during training for final evaluation.")
        return best_model
    else:
        print("No evaluated best model was created. Using the last trained model for final evaluation.")
        return model

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
        ], start_method="spawn")
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

        if model.num_timesteps < TOTAL_TIMESTEPS:
            model.learn(
                total_timesteps=TOTAL_TIMESTEPS - model.num_timesteps,
                callback=[
                    CheckpointCallback(
                        save_freq=max(CHECKPOINT_EVERY // get_n_envs(), 1),
                        save_path=CHECKPOINT_DIR,
                        name_prefix=MODEL_NAME,
                        verbose=2,
                    ),
                    EvalCallback(
                        eval_env,
                        best_model_save_path=BEST_MODEL_SAVE_DIR,
                        log_path=BEST_MODEL_SAVE_DIR,
                        eval_freq=max(EVAL_EVERY // get_n_envs(), 1),
                        n_eval_episodes=N_EVAL_EPISODES,
                        deterministic=True,
                        verbose=2,
                        render=False,
                        callback_on_new_best=RecordVideoAtBestModelCallback(),
                    ),
                ],
                reset_num_timesteps=reset_num_timesteps,
            )
        else:
            print(f"Model has already reached {TOTAL_TIMESTEPS} timesteps. Skipping training.")

        # Evaluate and save the same policy: the best checkpoint when available.
        final_model = get_model_to_save(model, device)
        final_evaluation(final_model, eval_env)

        # save the final model
        final_model.save(MODEL_NAME)
        print(f"Final evaluated model saved as '{MODEL_NAME}.zip'")
    finally:
        if env is not None:
            env.close()
        if eval_env is not None:
            eval_env.close()

if __name__ == "__main__":
    setup_game()
    main()
