import pytest
import yaml

import sys
import os

# Get the absolute path to the 'src' directory
# This looks up two levels from the current test file
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.append(src_path)

# Now Python can see the generation folder
from generation.sweep_builder import build_sweep

@pytest.fixture
def mock_config_file(tmp_path):
    """
    Creates a temporary YAML config file for testing.
    'tmp_path' is a built-in pytest fixture that provides a temporary directory.
    """
    config_data = {
        "global": {
            "T": 100,
            "n_seeds": 2,
            "seed_offset": 10
        },
        "experiments": [
            {
                "id": 1,
                "name": "Test Exp",
                "generator": {
                    "type": "test_gen",
                    "params": {"x": 1},
                    "sweep": {"mu": [0.1, 0.2]}
                },
            }
        ]
    }
    
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    
    return str(config_file)

def test_build_sweep_length(mock_config_file):
    """
    Verifies that the number of generated entries is correct.
    Calculated as: (number of seeds) * (number of sweep values).
    In our mock: 2 seeds * 2 mu values = 4 entries.
    """
    sweep = build_sweep(mock_config_file)
    assert len(sweep) == 4

def test_build_sweep_content(mock_config_file):
    """
    Verifies that the dictionary keys and values are correctly merged.
    """
    sweep = build_sweep(mock_config_file)
    first_entry = sweep[0]
    
    assert first_entry["experiment_id"] == 1
    assert "mu" in first_entry["params"]
    assert first_entry["params"]["T"] == 100
    assert first_entry["params"]["seed_noise"] == 10  # seed_offset (10) + first index (0)

def test_invalid_path():
    """
    Verifies that the function raises a FileNotFoundError for non-existent files.
    """
    with pytest.raises(FileNotFoundError):
        build_sweep("non_existent_file.yaml")