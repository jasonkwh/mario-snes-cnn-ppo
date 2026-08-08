from pathlib import Path
from config import CHECKPOINT_DIR, MODEL_NAME

def get_latest_checkpoint_path() -> Path | None:
    checkpoints = list(Path(CHECKPOINT_DIR).glob(f"{MODEL_NAME}_*_steps.zip"))
    
    return max(
        checkpoints,
        key=lambda path: int(path.stem.rsplit("_", 2)[-2]),
        default=None,
    )
