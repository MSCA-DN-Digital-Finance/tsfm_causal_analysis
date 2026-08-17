import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path
import json
import yaml
import os
import sys

# 1. PATH SETUP
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if src_path not in sys.path:
    sys.path.append(src_path)

from prediction.pred_runner import run_prediction, get_exp_id_from_meta, get_prediction_params

############### TESTS FOR get_exp_id_from_meta ##############

def test_get_exp_id_from_meta_success(tmp_path):
    """Test successfully extracting experiment_id from meta.json."""
    # Setup: Create a directory with a valid meta.json
    meta_data = {"experiment_id": "exp_01", "other_info": "data"}
    (tmp_path / "meta.json").write_text(json.dumps(meta_data))
    
    exp_id = get_exp_id_from_meta(tmp_path)
    assert exp_id == "exp_01"

def test_get_exp_id_from_meta_missing_file(tmp_path):
    """Test behavior when meta.json does not exist."""
    # tmp_path is empty by default
    exp_id = get_exp_id_from_meta(tmp_path)
    assert exp_id is None

def test_get_exp_id_from_meta_corrupted_json(tmp_path):
    """Test behavior when meta.json contains invalid JSON."""
    # Setup: Write invalid JSON text
    (tmp_path / "meta.json").write_text("{ 'invalid': json }")
    
    exp_id = get_exp_id_from_meta(tmp_path)
    assert exp_id is None

def test_get_exp_id_from_meta_missing_key(tmp_path):
    """Test behavior when file exists but key 'experiment_id' is missing."""
    meta_data = {"wrong_key": "oops"}
    (tmp_path / "meta.json").write_text(json.dumps(meta_data))
    
    exp_id = get_exp_id_from_meta(tmp_path)
    assert exp_id is None



############## TESTS FOR get_prediction_params ##############

def test_get_prediction_params_success():
    """Test retrieving valid prediction params."""
    mock_cfg = {
        'experiments': [
            {'id': 1, 'prediction': {'input_length': 200, 'output_length': 1, 'gap': 1}},
            {'id': 2, 'prediction': {'input_length': 100, 'output_length': 5, 'gap': 0}}
        ]
    }
    
    # We patch the 'load_config' wherever it is imported in your pred_runner file
    with patch('prediction.pred_runner.load_config', return_value=mock_cfg):
        params = get_prediction_params(Path("fake_path.yaml"), exp_id=2)
        assert params['input_length'] == 100
        assert params['output_length'] == 5

def test_get_prediction_params_missing_section():
    """Test error when ID exists but 'prediction' block is missing."""
    mock_cfg = {
        'experiments': [{'id': 'no_pred', 'name': 'Missing Section'}]
    }
    
    with patch('prediction.pred_runner.load_config', return_value=mock_cfg):
        with pytest.raises(ValueError, match="found but missing 'prediction' section"):
            get_prediction_params(Path("fake.yaml"), exp_id="no_pred")

def test_get_prediction_params_not_found():
    """Test error when experiment ID does not exist."""
    mock_cfg = {'experiments': [{'id': 1}]}
    
    with patch('prediction.pred_runner.load_config', return_value=mock_cfg):
        with pytest.raises(ValueError, match="not found in configuration"):
            get_prediction_params(Path("fake.yaml"), exp_id=999)



############### TESTS FOR run_prediction ORCHESTRATION ##############

def test_run_prediction_orchestration(tmp_path):
    """
    Tests discovery, execution, and skipping logic with patched inference.
    """
    # --- 1. SETUP CONFIG ---
    config_path = tmp_path / "experimental_config.yaml"
    config_data = {
        'experiments': [
            {
                'id': 1,
                'prediction': {'input_length': 24, 'output_length': 24}
            }
        ]
    }
    config_path.write_text(yaml.dump(config_data))

    # --- 2. SETUP DIRECTORIES & DATA ---
    run_1 = tmp_path / "run_01"
    run_2 = tmp_path / "run_02"
    
    for r in [run_1, run_2]:
        r.mkdir()
        np.savez(r / "trajectory.npz", x=np.arange(100, dtype=float))
        meta_data = {"experiment_id": 1, "seed": 42}
        (r / "meta.json").write_text(json.dumps(meta_data))

    # Pre-fill run_02 to trigger skip logic in new structural directory
    run_1_hash = run_1.name  # Extracts "run_01"
    run_2_hash = run_2.name  # Extracts "run_02"
    
    pred_dir_2 = tmp_path.parent / "prediction" / run_2_hash / "mock_model"
    pred_dir_2.mkdir(parents=True, exist_ok=True)
    (pred_dir_2 / "predictions.npz").write_text("already exists")

    # --- 3. EXECUTION WITH PATCHES ---
    fake_preds = np.array([4, 5, 6])
    mock_registry = {"mock_model": {"some": "spec"}}
    
    with patch("prediction.pred_runner.MODEL_REGISTRY", mock_registry), \
         patch("prediction.pred_runner.inference_pipeline", return_value=fake_preds) as mock_pipe:
        
        run_prediction(
            model_name="mock_model", 
            exp_config_path=config_path,
            root_dir=tmp_path
        )

        # --- 4. ASSERTIONS ---
        assert mock_pipe.call_count == 1
        
        args, kwargs = mock_pipe.call_args
        assert args[0] == {"some": "spec"}
        assert "data" in kwargs
        assert kwargs["data"].ndim >= 2 
        assert kwargs["horizon"] == 24

    # --- 5. VERIFY OUTPUT FILES (CRITICAL STRUCTURAL FIX HERE) ---
    out_file = tmp_path.parent / "prediction" / run_1_hash / "mock_model" / "predictions.npz"
    assert out_file.exists()
    
    with np.load(out_file) as d:
        np.testing.assert_array_equal(d["trajectory"], fake_preds)

