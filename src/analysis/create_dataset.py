"""
This file contains the function to create an aggregated dataset of all the metrics for each run in the artifacts/analysis directory.

The final dataset should have the following columns:

- run_id: The unique hash identifier for the run.
- experiment_id: The experiment ID associated with the run.
- generator_name: The name of the generator used.
- intervention_param: The parameter intervened on.
- parameter_value: The value of the intervened parameter.
- param_stat_name: The name of the parameter statistic.
- param_stat_of: The  model name or "trajectory" that the parameter statistic was calculated on. 
- param_stat_value: The computed value of the parameter statistic for the generator's output.

"""

from pathlib import Path
import sys
import pandas as pd
import json

# Calculate the project root (2 levels up from src/prediction/)
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import load_config

ANALYSIS_DIR = Path(project_root) / "artifacts" / "analysis"
OUTPUT_DIR = Path(project_root) / "artifacts" / "dataset"


def create_dataset(analysis_dir: str, output_dir: str, exp_config: dict):

    """
    Creates an aggregated dataset of all the metrics for each run in the artifacts/analysis directory.

    Args:
        analysis_dir (str): Path to the artifacts/analysis directory containing the metrics.
        output_dir (str): Path to the directory where the aggregated dataset will be saved.
        exp_config (dict): Configuration dictionary containing experiment settings.

    Returns:
        None

    """
    # convert analysis_dir and output_dir into Path

    analysis_dir = Path(analysis_dir)
    output_dir = Path(output_dir)

   

    # check if output_dir exists, if not create it
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create experiments map to retrieve experimental config information via experiment id
    exp_map = {str(e.get("id")): e for e in exp_config.get("experiments", [])}

    # Create a records list to append data to
    records = []
    # Get list of param_stats.json file paths in analysis_dir
    param_stat_file_paths = list(analysis_dir.rglob("*/param_stats.json"))
    print(f"Found {len(param_stat_file_paths)} param_stats.json files in {analysis_dir}!")

    print("Starting to process files...")
    for count, param_stat_file_path in enumerate(param_stat_file_paths):

        
    # From analysis/hash/../param_stat.json

        # Get run_id from path
        run_id = param_stat_file_path.parts[-3]

        # Get param_stat_of from path

        param_stat_of = param_stat_file_path.parts[-2]

        # Get param_stat_name from file
        with open(param_stat_file_path, "r") as f:
            param_stats_file_content = json.load(f)

        param_stat_name = next(iter(param_stats_file_content))
        
        # Get param_stat_value from file
        param_stat_value = param_stats_file_content[param_stat_name]
        
    # From generation/hash/meta.json

        # Create file path

        meta_path = analysis_dir.parent / "generation" / run_id / "meta.json"

        with open(meta_path, "r") as f:
            meta_file_content = json.load(f)

        # Get experiment_id 
        experiment_id = str(meta_file_content["sweep_config"]["experiment_id"])

        # Get generator_name
        generator_name = meta_file_content["sweep_config"]["generator"]

    
    # With experiment_id, from exp_map 

        # Get intervention param
        sweep_dict = exp_map[experiment_id]["generator"]["sweep"]
        intervention_param = str(next(iter(sweep_dict)))

    # With intervention_param, from generation/hash/meta.json

        # Get parameter_value
        parameter_value = meta_file_content["sweep_config"]["params"][intervention_param]



    # Create a single dictionary row
        row = {
            "run_id": run_id,
            "experiment_id": experiment_id,
            "generator_name": generator_name,
            "intervention_param": intervention_param,
            "parameter_value": parameter_value,
            "param_stat_name": param_stat_name,
            "param_stat_of": param_stat_of,
            "param_stat_value": param_stat_value
        }
        
        # Append values to records
        records.append(row)

        if (count+1) % 100 == 0:
            print(f"{count+1} out of {len(param_stat_file_paths)} files completed...")

    # Create dataframe from records
    df = pd.DataFrame(records, columns=[
            "run_id", "experiment_id", "generator_name", "intervention_param", 
            "parameter_value", "param_stat_name", "param_stat_of", "param_stat_value"
        ])
    
    # Save the DataFrame as a CSV file in the output_dir
    output_path = f"{output_dir}/aggregated_dataset.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved aggregated dataset to {output_path}.")

    

if __name__ == "__main__":

    # Load the experiment configuration
    exp_config_path = Path(project_root) / "experiment_config.yaml"
    exp_config = load_config(exp_config_path)

    # Create the aggregated dataset
    create_dataset(ANALYSIS_DIR, OUTPUT_DIR, exp_config)

