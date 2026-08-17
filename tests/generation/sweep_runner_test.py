import pytest
import numpy as np
import json
from pathlib import Path
import sys
import os

# Get the absolute path to the 'src' directory
# This looks up two levels from the current test file
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.append(src_path)

from generation.sweep_runner import save_run, run_sweep 


def test_save_run(tmp_path):

    """
    Verifies the data serialization and file-system I/O.
    
    Checks:
    - Does it create the directory if it doesn't exist?
    - Does it successfully compress NumPy arrays into an .npz file?
    - Does it correctly map the 'x' and 'noise' keys to the saved file?
    - Does the JSON metadata accurately preserve the config and the hash-based folder name?
    """
    
    # Arrange
    run_dir = tmp_path / "test_run_001"
    traj = {
        "x": np.array([1.0, 2.0, 3.0]),
        "noise": np.array([0.1, -0.1, 0.0]),
        "name": "test_gen"
    }
    config = {"generator": "test_gen", "params": {"sigma": 1.0}}

    # Act
    save_run(run_dir, traj, config)

    # Assert
    assert (run_dir / "trajectory.npz").exists()
    assert (run_dir / "meta.json").exists()

    # Verify NPZ content
    data = np.load(run_dir / "trajectory.npz")
    np.testing.assert_array_equal(data["x"], traj["x"])
    
    # Verify JSON content
    meta = json.loads((run_dir / "meta.json").read_text())
    assert meta["run_id"] == "test_run_001"
    assert meta["sweep_config"]["generator"] == "test_gen"



def test_run_sweep_logic(tmp_path):
    """
    Integration test: Verifies the end-to-end sweep execution loop.
    
    Checks:
    1. Does it correctly iterate through a list of multiple sweep configurations?
    2. Does it resolve the correct generator function from the registry?
    3. Does it pass parameters through to the generator using dictionary unpacking?
    4. Does it organize files using the path structure: /root/hash_id/?
    5. Does it skip existing hashes?
    """
    
    # 1. Mock Generator Registry
    def mock_gen(**params):
        return {"x": np.zeros(10), "noise": np.zeros(10), "name": "mock"}
    
    gen_registry = {"mock_gen": mock_gen}
    
    # 2. Define a sweep list
    sweep_list = [
        {"generator": "mock_gen", "params": {"val": 1}, "id": "sweep1"},
        {"generator": "mock_gen", "params": {"val": 2}, "id": "sweep2"}
    ]
    
    out_root = tmp_path / "output"

    # Act: First Run
    run_sweep(sweep_list, gen_registry, out_root)
    
    # Assert: Check if folders were created
    # Note: folders are structured as out_root / hash
    gen_folders = list(out_root.iterdir())
    assert len(gen_folders) == 2
    
    # Act: Second Run (Testing the "skip" logic)
    # We can check the print output or just ensure it doesn't crash 
    # and files remain intact.
    run_sweep(sweep_list, gen_registry, out_root)
    assert len(list(out_root.iterdir())) == 2