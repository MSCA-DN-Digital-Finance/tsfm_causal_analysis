
"""
This file contains unit tests for the run_metric function in the `analysis.metric_runner` module.

"""

import json
import yaml
import numpy as np
from pathlib import Path
import os
import sys

# Get the absolute path to the 'src' directory
# This looks up two levels from the current test file
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.append(src_path)


from analysis.metric_runner import run_metric

def test_run_metric_orchestration(tmp_path):
    # 1. Setup Mock Directories
    input_root = tmp_path / "prediction"
    output_root = tmp_path / "analysis"
    gen_root = tmp_path / "generation"
    
    # Create path structures for a mock run: 'hash_123'
    run_hash = "hash_123"
    model_name = "mock_model"
    
    pred_dir = input_root / run_hash / model_name
    pred_dir.mkdir(parents=True)
    gen_dir = gen_root / run_hash
    gen_dir.mkdir(parents=True)
    
    # 2. Populate Mock Files
    # Save dummy data arrays
    np.savez(pred_dir / "predictions.npz", trajectory=np.array([1.0, 2.0, 3.0]))
    np.savez(gen_dir / "trajectory.npz", x=np.array([1.0, 2.0, 3.0]))
    
    # Save mock experiment metadata
    meta_data = {"experiment_id": 1}
    (gen_dir / "meta.json").write_text(json.dumps(meta_data))
    
    # 3. Setup Config and Registry Inputs
    exp_config = {
        "experiments": [
            {"id": 1, "analysis": {"param_stat": "mock_stat"}}
        ]
    }
    
    # Mock registry metric function (simply returns mean)
    param_registry = {"mock_stat": lambda arr: float(np.mean(arr))}
    
    # 4. Run Orchestrator
    run_metric(
        INPUT_ROOT=str(input_root),
        OUTPUT_ROOT=str(output_root),
        param_registry=param_registry,
        exp_config=exp_config
    )
    
    # 5. Assertions: Verify outputs exist and calculations are correct
    pred_analysis_file = output_root / run_hash / model_name / "param_stats.json"
    traj_analysis_file = output_root / run_hash / "trajectory" / "param_stats.json"
    
    assert pred_analysis_file.exists()
    assert traj_analysis_file.exists()
    
    # Verify JSON content matches expected metric calculation outputs
    with open(pred_analysis_file, "r") as f:
        pred_res = json.load(f)
        assert pred_res["mock_stat"] == 2.0  # Mean of [1, 2, 3]