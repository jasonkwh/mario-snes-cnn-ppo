from pathlib import Path
from config import CHECKPOINT_DIR, MODEL_NAME, BEST_MODEL_SAVE_DIR

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
