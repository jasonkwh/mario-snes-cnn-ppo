GAME_NAME = "SuperMarioWorld-Snes-v0"
STATE_NAME = "YoshiIsland1"
MODEL_NAME = "mario_ppo"
RECORD_VIDEO = True
RECORD_VIDEO_EVERY = 5
VIDEO_RENDER_FPS = 60
CHECKPOINT_DIR = "./mario_checkpoints/"
BEST_MODEL_SAVE_DIR = "./mario_best_model/"
VIDEO_DIR = "./mario_videos/"
LOG_DIR = "./mario_logs/"
TENSORBOARD_LOG_DIR = f"{LOG_DIR}tensorboard/"
MONITOR_FILENAME = f"{LOG_DIR}monitor.csv"
MONITOR_EVALUATION_FILENAME = f"{LOG_DIR}monitor_evaluation.csv"

# Training parameters
FRAME_SKIP = 4 # Repeat each action for 4 frames to reduce decisions and speed up training
OBSERVATION_SHAPE = (84, 96) # Resize observations to height x width for the CNN input
LEARNING_RATE = 0.00025 # Learning rate for the optimizer
N_STEPS = 2_048 # Number of steps to run for each environment per update (per policy rollout)
BATCH_SIZE = 64
ENT_COEF = 0.01 # Entropy coefficient for the loss calculation
FRAME_STACK = 4 # Number of frames to stack
TOTAL_TIMESTEPS = 1_000_000
EVAL_FREQ = 20_000 # Evaluate every 20,000 training timesteps
N_EVAL_EPISODES = 5 # Number of episodes to run during each evaluation
MAX_EVAL_EPISODE_STEPS = 2_500 # Maximum number of steps per evaluation episode
CHECKPOINT_EVERY = 50_000 # Save a checkpoint every 50,000 training timesteps
