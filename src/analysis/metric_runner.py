# This file contains the functions required for the metric to compute the parameter statistics for a given run.

import json
import numpy as np
import sys
from pathlib import Path

# Calculate the project root (2 levels up from src/prediction/)
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)


from src.utils import get_exp_id_from_meta


def _process_single_run(
    file_path: Path,
    analysis_dir: Path,
    input_root: Path,
    param_registry: dict,
    exp_config: dict,
    load_strategy: str
):
    """Helper function to process an individual prediction or trajectory run."""
    # 1. Check if analysis directory already exists
    if analysis_dir.exists():
        print(f"Skipping {file_path} as analysis directory {analysis_dir} already exists.")
        return
    
    print(f"Parameter estimates for {file_path} will be saved to {analysis_dir}")
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # 2. Extract experiment ID from meta.json in the generation folder
    hash_val = file_path.parts[-3] if load_strategy == "prediction" else file_path.parts[-2]
    gen_dir = input_root.parent / "generation" / hash_val
    exp_id = get_exp_id_from_meta(gen_dir)

    exp_id_str = str(exp_id).strip() if exp_id is not None else ""

    # 3. Safely extract the param_estimate_name from config
    param_estimate_name = None
    for exp in exp_config.get('experiments', []):
        exp_id_candidate = str(str(exp.get('id')).strip()) 
        
        # Compare string against string using exp_id_str
        if exp_id_candidate == exp_id_str:
            param_estimate_name = exp.get('analysis', {}).get('param_estimate')
            break

    if not param_estimate_name:
        raise KeyError(
            f"Could not find an experiment matching ID '{exp_id}' in your configuration file. "
            f"Available IDs: {[e.get('id') for e in exp_config.get('experiments', [])]}. If ID is there, error must be raised due to other issues in retrieving the parameter estimate name."
        )

    param_estimate_func = param_registry.get(param_estimate_name)

    # 4. Load and preprocess data based on strategy
    with np.load(file_path) as data:
        if load_strategy == "prediction":
            key = 'trajectory' if 'trajectory' in data.files else data.files[0]
            trajectory = np.squeeze(data[key])
            if trajectory.ndim > 1:
                trajectory = np.mean(trajectory, axis=0)
            trajectory = np.asarray(trajectory, dtype=np.float64)
            print(f"Using key {key} to retrieve trajectory from {file_path}.")
        elif load_strategy == "trajectory":
            trajectory = data['x']
        else:
            raise ValueError(f"Unknown load strategy: {load_strategy}")

    # 5. Run metric function and save results
    try:
        param_estimate_value = param_estimate_func(trajectory)
    except Exception as e:
        print(f"Error computing {param_estimate_name} for {file_path}: {e}")
        param_estimate_value = None

    results = {param_estimate_name: param_estimate_value}
    
    param_estimates_file = analysis_dir / "param_estimates.json"
    with open(param_estimates_file, 'w') as f:
        json.dump(results, f, indent=4)


def run_metric(INPUT_ROOT: str, OUTPUT_ROOT: str, param_registry: dict, exp_config: dict):
    """
    Iterate over prediction and trajectory run directories and calculate parameter estimates.
    """
    input_root = Path(INPUT_ROOT)
    output_root = Path(OUTPUT_ROOT)

    # --- Process Predictions ---
    pred_file_paths = list(input_root.rglob("*/predictions.npz"))
    print(f"Found {len(pred_file_paths)} predictions.npz files in {input_root}")

    for pred_file_path in pred_file_paths:
        hash_val = pred_file_path.parts[-3]
        model = pred_file_path.parts[-2]
        analysis_dir = output_root / hash_val / model
        
        _process_single_run(
            file_path=pred_file_path,
            analysis_dir=analysis_dir,
            input_root=input_root,
            param_registry=param_registry,
            exp_config=exp_config,
            load_strategy="prediction"
        )

    # --- Process Trajectories ---
    traj_file_paths = list(input_root.parent.rglob("*/trajectory.npz"))
    print(f"Found {len(traj_file_paths)} trajectory.npz files in {input_root.parent}")

    for traj_file_path in traj_file_paths:
        hash_val = traj_file_path.parts[-2]
        analysis_dir = output_root / hash_val / "trajectory"

        _process_single_run(
            file_path=traj_file_path,
            analysis_dir=analysis_dir,
            input_root=input_root,
            param_registry=param_registry,
            exp_config=exp_config,
            load_strategy="trajectory"
        )