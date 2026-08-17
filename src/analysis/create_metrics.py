# This file creates the parameter statistics for each run

import sys
from pathlib import Path

# Calculate the project root (2 levels up from src/prediction/)
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import load_config
from src.analysis.param_stats import PARAM_STATS_REGISTRY
from src.analysis.metric_runner import run_metric

INPUT_ROOT = Path(project_root) / "artifacts" / "prediction"
OUTPUT_ROOT = Path(project_root) / "artifacts" / "analysis"
EXP_CONFIG_PATH = Path(project_root) / "experiment_config.yaml"

def create_metrics(input_root, output_root, param_stat_registry, exp_config_path):
    """
    Creates the parameter statistics for each run by calling the run_metric function.

    Args:
        input_root (Path): Path to the directory containing predictions.
        output_root (Path): Path to the directory where output files will be saved.
        param_stat_registry (dict): Registry containing parameter statistics.
        exp_config_path (Path): Path to the experiment configuration file.

    Returns:
        None
    """

    # Load the experiment configuration
    exp_config = load_config(exp_config_path)

    # Call the run_metric function to compute parameter statistics
    run_metric(input_root, output_root, param_stat_registry, exp_config)

if __name__ == "__main__":
    create_metrics(INPUT_ROOT, OUTPUT_ROOT, PARAM_STATS_REGISTRY, EXP_CONFIG_PATH)

    
