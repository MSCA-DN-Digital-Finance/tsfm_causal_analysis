import sys
from pathlib import Path


root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from utils import load_config


def build_sweep(config_path="config.yaml"):
    """
    Orchestrates the creation of an experiment sweep by combining global
    defaults with specific experiment parameters and expanding sweeps.
    
    This function replaces hard-coded loops by iterating over the 
    defined experiments in the config file.
    
    Returns:
        list: A list of dictionaries, where each dict is a single configuration
              ready to be passed to a generator or model.
    """
    config = load_config(config_path)
    glob = config['global']
    sweep = []

    # Calculate seeds once based on global offset and count
    noise_seeds = [glob['seed_offset'] + i for i in range(glob['n_seeds'])]

    # Iterate through each experiment block defined in the YAML
    for exp in config['experiments']:
        
        # Identify the variable being 'swept' (e.g., 'mu' or 'phi')
        # We assume there is exactly one sweep key per experiment
        sweep_param_name = list(exp['generator']['sweep'].keys())[0]
        sweep_values = exp['generator']['sweep'][sweep_param_name]

        # Nested loops to create the Cartesian product of (Seeds x Sweep Values)
        for seed in noise_seeds:
            for val in sweep_values:
                # Construct the flat dictionary for this specific run
                entry = {
                    "experiment_id": exp['id'],
                    "generator": exp['generator']['type'],
                    "params" : {**exp['generator']['params'], sweep_param_name: val, "T": glob['T'], "seed_noise": int(seed)}
                }
                sweep.append(entry)
                
    return sweep
