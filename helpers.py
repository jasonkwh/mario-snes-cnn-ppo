from pathlib import Path
import os
from config import (
    CHECKPOINT_DIR, MODEL_NAME, BEST_MODEL_SAVE_DIR, 
    RESUME_FROM, RESERVED_CPU_CORES,
)

def get_n_envs() -> int:
    return max(1, len(os.sched_getaffinity(0)) - RESERVED_CPU_CORES)


def get_checkpoint_path() -> Path | None:
    match RESUME_FROM:
        case "latest":
            return get_latest_checkpoint_path()
        case "best":
            return get_best_model_path()
        case "none":
            return None
        case _:
            raise ValueError(f"Unknown RESUME_FROM value: {RESUME_FROM}")

def get_latest_checkpoint_path() -> Path | None:
    checkpoints = list(Path(CHECKPOINT_DIR).glob(f"{MODEL_NAME}_*_steps.zip"))
    
    return max(
        checkpoints,
        key=lambda path: int(path.stem.rsplit("_", 2)[-2]),
        default=None,
    )

def get_best_model_path() -> Path | None:
    path = Path(BEST_MODEL_SAVE_DIR) / "best_model.zip"
    return path if path.exists() else None
