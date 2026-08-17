import pytest
import numpy as np
import sys
import os

# Get the absolute path to the 'src' directory
# This looks up two levels from the current test file
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.append(src_path)

# Now Python can see the generation folder
from generation.generators import GENERATOR_REGISTRY



# Sample parameters for testing
TEST_CONFIGS = [
    ("rw_drift", {"T": 100, "mu": 0.1, "sigma": 1.0, "seed_noise": 42}),
    ("ar1", {"T": 50, "beta": 0.9, "sigma": 0.5, "seed_noise": 42}),
    ("harmonic", {"T": 200, "A": 2.0, "sigma": 0.1, "seed_noise": 42}),
    ("regime", {"T":200, "dwell_time": 10, "slopes": [1.0, -1.0], "seed_noise": 42})
]

@pytest.mark.parametrize("gen_name, params", TEST_CONFIGS)
def test_generator_output_structure(gen_name, params):
    """Check if all generators return the correct dictionary keys and types."""
    gen_func = GENERATOR_REGISTRY[gen_name]
    result = gen_func(**params)
    
    assert isinstance(result, dict)
    assert "x" in result
    assert "noise" in result
    assert "T" in result
    assert result["T"] == params["T"]
    assert len(result["x"]) == params["T"]
    assert len(result["noise"]) == params["T"]

@pytest.mark.parametrize("gen_name, params", TEST_CONFIGS)
def test_generator_reproducibility(gen_name, params):
    """Check if the same seeds produce the exact same trajectory."""
    gen_func = GENERATOR_REGISTRY[gen_name]
    
    run1 = gen_func(**params)
    run2 = gen_func(**params)
    
    np.testing.assert_array_almost_equal(run1["x"], run2["x"])
    np.testing.assert_array_almost_equal(run1["noise"], run2["noise"])

def test_rw_drift_logic():
    """Specific check for RW Drift: x = x0 + mu*t + noise."""
    T = 10
    mu = 0.5
    x0 = 10.0
    res = GENERATOR_REGISTRY["rw_drift"](T=T, x0=x0, mu=mu, sigma=0.0, seed_noise=42)
    
    # With sigma=0, x should be exactly x0 + mu*t
    expected_signal = np.array([10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5])
    np.testing.assert_array_almost_equal(res["x"], expected_signal)

def test_ar1_stability_check():
    """Ensure AR(1) raises error on unstable beta if we chose to enforce it."""
    with pytest.raises(ValueError, match="beta should be within"):
        GENERATOR_REGISTRY["ar1"](T=10, beta=2.0)

def test_negative_t_error():
    """Ensure all generators fail gracefully with non-positive T."""
    for gen_name in GENERATOR_REGISTRY:
        with pytest.raises(ValueError):
            GENERATOR_REGISTRY[gen_name](T=-1)

