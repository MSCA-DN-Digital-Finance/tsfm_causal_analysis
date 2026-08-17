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




def run_metric(INPUT_ROOT:str, OUTPUT_ROOT:str, param_registry:dict, exp_config:dict):
    """
    Iterate over each run directory in the given path and call calculate_param_stats for each run.

    Args:
        INPUT_ROOT (str): Path to the directory containing predictions.
        OUTPUT_ROOT (str): Path to the directory where output files will be saved.
        param_registry (dict): A dictionary mapping parameter statistic names to their corresponding functions.
        exp_config (dict): The experiment configuration.

    Returns:
        None

    """

    INPUT_ROOT = Path(INPUT_ROOT)
    OUTPUT_ROOT = Path(OUTPUT_ROOT)

    # get all paths for predictions.npz files in the INPUT_ROOT directory
    pred_file_paths = list(INPUT_ROOT.rglob("*/predictions.npz"))
    print(f"Found {len(pred_file_paths)} predictions.npz files in {INPUT_ROOT}")

    # for each predictions.npz file path
    for pred_file_path in pred_file_paths:

        hash = pred_file_path.parts[-3]
        model = pred_file_path.parts[-2]

        # 1. create the corresponding run directory in the OUTPUT_ROOT by chopping off the first two and the last part of the path and appending it to OUTPUT_ROOT
        analysis_dir = OUTPUT_ROOT / hash / model
        
        # if the analysis_dir already exists, skip this run
        if analysis_dir.exists():
            print(f"Skipping {pred_file_path} as analysis directory {analysis_dir} already exists.")
            continue
        print(f"Parameter statistics for {pred_file_path} will be saved to {analysis_dir}")

        # create the analysis directory if it doesn't exist
        analysis_dir.mkdir(parents=True, exist_ok=True)



        # 2. get experiment_id from artifacts/trajectories/hash/meta.json for a given run path
        gen_dir = INPUT_ROOT.parent / "generation" / hash # information about the experiment_id is stored in the generation folder, not in the prediction folder
        exp_id = get_exp_id_from_meta(gen_dir)

        # 3. Safely extract the param_stat_name from the experiments list
        param_stat_name = None

        for exp in exp_config.get('experiments', []):
            # Convert both IDs to strings to ensure matching type ('1' == '1')
            if str(exp.get('id')) == str(exp_id):
                param_stat_name = exp.get('analysis', {}).get('param_stat')
                break

        # Handle edge case where lookup fails
        if not param_stat_name:
            raise KeyError(
                f"Could not find an experiment matching ID '{exp_id}' in your configuration file. "
                f"Available IDs: {[e.get('id') for e in exp_config.get('experiments', [])]}"
    )

        # 4. select parameter statistic function from param_registry
        param_stat_func = param_registry.get(param_stat_name)

        # 5. load predictions.npz file and extract the 'trajectory' array safely
        with np.load(pred_file_path) as data:
            # Fallback to the first available key if 'trajectory' isn't explicitly defined
            key = 'trajectory' if 'trajectory' in data.files else data.files[0]
            trajectory = data[key]
            
            # CRITICAL: If the foundation model outputs multiple paths/samples (2D array),
            # convert it to a 1D point forecast by taking the mean across samples.
            if trajectory.ndim > 1:
                # Assuming shape is (num_samples, sequence_length) -> reduces to (sequence_length,)
                trajectory = np.mean(trajectory, axis=0)
        
        print(f"Using key {key} to retrieve trajectory from {pred_file_path}.")


        # 6. run the parameter statistic function on the loaded data
        try:
            param_stat_value = param_stat_func(trajectory)
        except Exception as e:
            print(f"Error computing {param_stat_name} for {pred_file_path}: {e}")
            param_stat_value = None

        # 7. create a results dictionary to store the computed parameter statistics
        results = {param_stat_name: param_stat_value}

        # 8. save the results dictionary to analysis_dir/param_stats.json
        param_stats_file = analysis_dir / "param_stats.json"
        with open(param_stats_file, 'w') as f:
            json.dump(results, f, indent=4)

    # Do the same for trajectories.npz files in the INPUT_ROOT.parent / "generation" directory
    traj_file_paths = list(INPUT_ROOT.parent.rglob("*/trajectory.npz"))
    print(f"Found {len(traj_file_paths)} trajectory.npz files in {INPUT_ROOT.parent}")

    for traj_file_path in traj_file_paths:
        hash = traj_file_path.parts[-2]


        # 1. create the corresponding run directory in the OUTPUT_ROOT by chopping off the first two and the last part of the path and appending it to OUTPUT_ROOT
        analysis_dir = OUTPUT_ROOT / hash / "trajectory"

        # if the analysis_dir already exists, skip this run
        if analysis_dir.exists():
            print(f"Skipping {traj_file_path} as analysis directory {analysis_dir} already exists.")
            continue
        print(f"Parameter statistics for {traj_file_path} will be saved to {analysis_dir}")

        # create the analysis directory if it doesn't exist
        analysis_dir.mkdir(parents=True, exist_ok=True)

        # 2. get experiment_id from artifacts/trajectories/hash/meta.json for a given run path
        gen_dir = Path(INPUT_ROOT).parent / "generation" / hash # information about the experiment_id is stored in the generation folder, not in the prediction folder
        exp_id = get_exp_id_from_meta(gen_dir)

        # 3. Safely extract the param_stat_name from the experiments list
        param_stat_name = None

        for exp in exp_config.get('experiments', []):
            # Convert both IDs to strings to ensure matching type ('1' == '1')
            if str(exp.get('id')) == str(exp_id):
                param_stat_name = exp.get('analysis', {}).get('param_stat')
                break

        # Handle edge case where lookup fails
        if not param_stat_name:
            raise KeyError(
                f"Could not find an experiment matching ID '{exp_id}' in your configuration file. "
                f"Available IDs: {[e.get('id') for e in exp_config.get('experiments', [])]}"
    )

        # 4. select parameter statistic function from param_registry
        param_stat_func = param_registry.get(param_stat_name)

        # 5. load trajectories.npz file and extract the 'x' array
        with np.load(traj_file_path) as data:
            trajectory = data['x']

        # 6. run the parameter statistic function on the loaded data
        try:
            param_stat_value = param_stat_func(trajectory)
        except Exception as e:
            print(f"Error computing {param_stat_name} for {traj_file_path}: {e}")
            param_stat_value = None

        # 7. create a results dictionary to store the computed parameter statistics
        results = {param_stat_name: param_stat_value}

        # 8. save the results dictionary to analysis_dir/param_stats.json
        param_stats_file = analysis_dir / "param_stats.json"
        with open(param_stats_file, 'w') as f:
            json.dump(results, f, indent=4)


    

    

