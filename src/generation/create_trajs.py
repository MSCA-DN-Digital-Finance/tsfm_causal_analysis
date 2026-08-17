# create_data.py
from __future__ import annotations
from pathlib import Path
from generators import GENERATOR_REGISTRY

from sweep_builder import build_sweep
from sweep_runner import run_sweep

# This gets the directory where THIS script lives
script_dir = Path(__file__).resolve().parent 

# This goes up two levels from src/prediction to the project root
# where experiment_config.yaml actually lives
CONFIG_PATH = script_dir.parent.parent / "experiment_config.yaml"
OUT_ROOT = script_dir.parent.parent / "artifacts/generation"  # This is where the generated trajectories will be saved
def main():
    
    sweep = build_sweep(CONFIG_PATH)
    gen_registry = GENERATOR_REGISTRY

    run_sweep(sweep, gen_registry, out_root=OUT_ROOT)


if __name__ == "__main__":
    main()

