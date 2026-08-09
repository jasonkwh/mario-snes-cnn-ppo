from pathlib import Path
import stable_retro
import json
from config import GAME_NAME

def setup_game():
    retro_dir = Path(stable_retro.__file__).parent
    matches = list(retro_dir.rglob(f"{GAME_NAME}/data.json"))
    data_path = matches[0]

    print(f"Updating {data_path}")

    data = json.loads(data_path.read_text())

    data["info"].update({
        "x": {
            "address": 8257684,
            "type": "<u2",
        },
        # powerup
        # valid values: 00 small, 01 big, 02 cape, 03 fire
        "powerup": {
            "address": 8257561,
            "type": "<u1",
        },
    })

    data_path.write_text(json.dumps(data, indent=2) + "\n")

    print(f"Updated {data_path}")

if __name__ == "__main__":
    setup_game()
