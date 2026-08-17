from __future__ import annotations
from pathlib import Path

from pred_runner import run_prediction

# This gets the directory where THIS script lives
script_dir = Path(__file__).resolve().parent 

# This goes up two levels from src/prediction to the project root
# where experiment_config.yaml actually lives
CONFIG_PATH = script_dir.parent.parent / "experiment_config.yaml"
INPUT_ROOT = script_dir.parent.parent / "artifacts/generation"  # This is where the generated trajectories are located

def main():

    model_name = "timesfm"

    run_prediction(
        model_name=model_name,
        exp_config_path=CONFIG_PATH,
        root_dir=INPUT_ROOT
    )


if __name__ == "__main__":
    main()