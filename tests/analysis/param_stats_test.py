"""
This file contains unit tests for the param_stats functions in the `analysis.param_stats` module.
"""

import os
import sys
import numpy as np
import pytest
from fbm import fbm

# Get the absolute path to the 'src' directory
# This looks up two levels from the current test file
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.append(src_path)

from analysis.param_stats import (
    estimated_mean,
    estimated_beta,
    estimated_wavelength,
    estimated_dwell_time,
    estimated_threshold,
    estimated_hurst_exponent
)

################### Tests for estimated_mean function ###################
estimated_mean_testdata = [
    # Case 0: Increasing values
    (
        np.arange(1, 1000, 1), # input run
        1.0                    # expected probability
    ),
    # Case 1: Decreasing values
    (
        np.arange(0, -99, -1), # input run
        -1.0                    # expected probability
    ),
    # Case 2: Mixed values sampled from a normal distribution
    (
        np.random.normal(0, 1, 1000), # input run
        0.0                           # expected probability
    ),
    # Case 3: Empty trajectory
    (
        np.array([]), # input run
        ValueError    # raises ValueError due to empty input
    ),
    # Case 4: Wrong data type (string instead of numeric)
    (
        ["a", "b", "c"], # input run
        ValueError       # raises ValueError due to invalid data type
    ),
    # Case 5: Single-value input (cannot compute changes)
    (
        np.array([42.0]),
        ValueError       # raises ValueError (results in NaN mean of empty slice)
    ),
    # Case 6: Single-value nested input (proves dimension handling)
    (
        np.array([[42.0]]),
        ValueError       # raises ValueError
    )
]

@pytest.mark.parametrize("input_run,expected_output", estimated_mean_testdata)
def test_estimated_mean(input_run, expected_output):
    """
    Tests the `estimated_mean` function to ensure it correctly computes 
    the mean of the trajectory.
    """
    if expected_output == ValueError or isinstance(expected_output, ValueError):
        with pytest.raises(ValueError):
            estimated_mean(input_run)
    else:
        assert np.isclose(estimated_mean(input_run), expected_output, atol=1e-1), f"Expected {expected_output}, got {estimated_mean(input_run)}"


#################### Tests for estimated_beta function ###################

def generate_ar1_series(n_steps=500, beta=0.7, sigma=1.0, seed=42):
    """
    Generates an AR(1) time series: x_{t+1} = beta * x_t + epsilon_t
    """
    rng = np.random.default_rng(seed)
    series = np.zeros(n_steps)
    epsilon = rng.normal(loc=0.0, scale=sigma, size=n_steps)
    series[0] = epsilon[0]
    for t in range(1, n_steps):
        series[t] = beta * series[t-1] + epsilon[t]
    return np.asarray(series)

estimated_beta_testdata = [
    # Case 0: Positive correlation with beta=0.7
    (
        generate_ar1_series(n_steps=500, beta=0.7, sigma=1.0),
        0.7 # expected beta
    ),
    # Case 1: Positive correlation with beta=1.0
    (
        generate_ar1_series(n_steps=500, beta=1.0, sigma=1.0),
        1.0 # expected beta
    ),
    # Case 2: Empty trajectory
    (
        np.array([]),
        ValueError
    ),
    # Case 3: Negative correlation with beta=-0.5
    (
        generate_ar1_series(n_steps=500, beta=-0.5, sigma=1.0),
        -0.5 # expected beta
    ),
    # Case 4: Single-value input (needs > 1 observation for autoregression)
    (
        np.array([42.0]),
        ValueError
    ),
    # Case 5: Single-value nested input (proves dimension handling)
    (
        np.array([[42.0]]),
        ValueError
    )
]

@pytest.mark.parametrize("input_run,expected_output", estimated_beta_testdata)
def test_estimated_beta(input_run, expected_output):
    """
    Tests the `estimated_beta` function to ensure it correctly computes 
    the model-implied AR(1) beta.
    """
    if expected_output == ValueError or isinstance(expected_output, ValueError):
        with pytest.raises(ValueError):
            estimated_beta(input_run)
    else:
        assert np.sign(estimated_beta(input_run)) == np.sign(expected_output), f"Expected sign {np.sign(expected_output)}, got {np.sign(estimated_beta(input_run))}"


##################### Tests for estimated_wavelength function ###################

n = 1000 # number of samples for the test signals
estimated_wavelength_testdata = [
    # Case 0: Simple sinusoidal signal with  frequency of 0.1 Hz
    (
        np.sin(2 * np.pi * 0.1 * np.arange(n)),
        10.0 # expected wavelength (1/frequency)
    ),
    # Case 1: Simple sinusoidal signal with frequency of 0.05 Hz
    (
        np.sin(2 * np.pi * 0.05 * np.arange(n)),
        20.0 # expected wavelength (1/frequency)
    ),
    # Case 2: Constant signal of ones (should return 0.0 as dominant frequency)
    (
        np.ones(n),
        float('inf') # expected wavelength (1/frequency)
    ),
    # Case 3: Empty trajectory
    (
        np.array([]),
        ValueError
    ),
    # Case 4: Single-value input (fails signal.welch length requirements)
    (
        np.array([42.0]),
        ValueError
    ),
    # Case 5: Single-value nested input (proves dimension handling)
    (
        np.array([[42.0]]),
        ValueError
    )
]

@pytest.mark.parametrize("input_run,expected_output", estimated_wavelength_testdata)
def test_estimated_wavelength(input_run, expected_output):
    """
    Tests the `estimated_wavelength` function to ensure it correctly computes 
    the estimated wavelength of the trajectory.
    """
    if expected_output == ValueError or isinstance(expected_output, ValueError):
        with pytest.raises(ValueError):
            estimated_wavelength(input_run)
    else:
        wavelength = estimated_wavelength(input_run)
        assert np.isclose(wavelength, expected_output, atol=0.5), f"Expected {expected_output}, got {wavelength}"


##################### Tests for estimated_dwell_time function ###################

def generate_regime_series(dwell_time=20, num_regimes=4, slopes=[2.0, -2.0], sigma=0.01, seed=42):
    """
    Generates a realistic multi-regime cumulative trend path with a known uniform 
    dwell time and a small amount of noise.
    """
    rng = np.random.default_rng(seed)
    T = dwell_time * num_regimes
    indices = np.arange(T)
    regimes = (indices // dwell_time) % len(slopes)
    slope_array = np.array(slopes)[regimes]
    noise = rng.normal(loc=0.0, scale=sigma, size=T)
    return np.cumsum(slope_array + noise)

estimated_dwell_time_testdata = [
    # Case 0: Clean step shifts with perfect uniform dwell time of 30 steps
    (
        generate_regime_series(dwell_time=30, num_regimes=10, slopes=[5.0, -5.0]),
        30.0 # expected average dwell time
    ),
    # Case 1: Clean step shifts with uniform dwell time of 15 steps
    (
        generate_regime_series(dwell_time=15, num_regimes=10, slopes=[10.0, -10.0]),
        15.0 # expected average dwell time
    ),
    # Case 2: Insufficient observation sequence length (length <= 2)
    (
        np.array([1.0, 2.0]),
        ValueError
    ),
    # Case 3: Empty trajectory
    (
        np.array([]),
        ValueError
    ),
    # Case 4: High dimensional matrix instead of 1D array
    (
        np.ones((2, 50)),
        ValueError
    ),
    # Case 5: Single element array (should raise ValueError due to length <= 2)
    (
        np.array([42.0]),
        ValueError
    ),
    # Case 6: Single-value nested input (proves dimension handling)
    (
        np.array([[42.0]]),
        ValueError
    )
]

@pytest.mark.parametrize("input_run,expected_output", estimated_dwell_time_testdata)
def test_estimated_dwell_time(input_run, expected_output):
    """
    Tests the `estimated_dwell_time` function to ensure it correctly estimates
    the average sequence length between trend switches using the PELT algorithm.
    """
    if expected_output == ValueError or isinstance(expected_output, ValueError):
        with pytest.raises(ValueError):
            estimated_dwell_time(input_run)
    else:
        estimated_dwell = estimated_dwell_time(input_run, penalty=1.5)
        assert np.isclose(estimated_dwell, expected_output, atol=1e-1), f"Expected average dwell time {expected_output}, got {estimated_dwell}"


##################### Tests for estimated_threshold function ###################

def generate_energy_release_series(T=100, threshold=10.0, mu=2.0, sigma=0.01, seed=42):
    """
    Generates a realistic energy-release trajectory with a deterministic build-up 
    and crisp resets when crossing the specified threshold.
    """
    rng = np.random.default_rng(seed)
    series = np.zeros(T, dtype=float)
    current_energy = 0.0
    noise = rng.normal(0, sigma, T)
    for t in range(T):
        increment = mu + np.abs(noise[t])
        current_energy += increment
        if current_energy >= threshold:
            series[t] = current_energy
            current_energy = 0.0
        else:
            series[t] = current_energy
    return series

estimated_threshold_testdata = [
    # Case 0: Clean sawtooth resets with a deterministic threshold of 10.0
    (
        generate_energy_release_series(T=100, threshold=10.0, mu=2.0),
        10.0  # expected average peak height near threshold
    ),
    # Case 1: Clean sawtooth resets with a deterministic threshold of 25.0
    (
        generate_energy_release_series(T=100, threshold=25.0, mu=5.0),
        25.0  # expected average peak height near threshold
    ),
    # Case 2: No reset points. Should fallback to global max.
    (
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        5.0  # expected fallback value (np.max)
    ),
    # Case 3: Single element array (returns the single value)
    (
        np.array([42.0]),
        42.0
    ),
    # Case 4: Single element nested array (proves dimension handling and returns value)
    (
        np.array([[42.0]]),
        42.0
    ),
    # Case 5: Empty trajectory
    (
        np.array([]),
        ValueError
    ),
    # Case 6: High dimensional matrix instead of 1D array
    (
        np.ones((2, 50)),
        ValueError
    )
]

@pytest.mark.parametrize("input_run,expected_output", estimated_threshold_testdata)
def test_estimated_threshold(input_run, expected_output):
    """
    Tests the `estimated_threshold` function to ensure it correctly identifies
    reset peaks and computes their average height, with correct fallbacks and exceptions.
    """
    if expected_output == ValueError or isinstance(expected_output, ValueError):
        with pytest.raises(ValueError):
            estimated_threshold(input_run)
    else:
        est_threshold = estimated_threshold(input_run)
        assert np.isclose(est_threshold, expected_output, atol=0.5), (
            f"Expected threshold estimation near {expected_output}, got {est_threshold}"
        )

##################### Tests for estimated_hurst_exponent function ###################

def generate_fractal_series(N=500, target_h=0.5, seed=42):
    """
    Generates an exact fractional Brownian motion trajectory using the 'fbm' library.
    - target_h = 0.5 generates standard Brownian motion
    - target_h > 0.5 generates persistent, trending paths
    - target_h < 0.5 generates anti-persistent, mean-reverting paths
    """
    # Fix: Set the numpy seed so the underlying random state is reproducible
    if seed is not None:
        np.random.seed(seed)
        
    # Setting n=N-1 makes the output array length exactly N (due to 0-index inclusion)
    return fbm(n=N-1, hurst=target_h, length=1, method='daviesharte')


estimated_hurst_exponent_testdata = [
    # Case 0: Persistent series (Target H = 0.75)
    (
        generate_fractal_series(N=500, target_h=0.75),
        0.75
    ),
    # Case 1: Standard random walk (Target H = 0.50)
    (
        generate_fractal_series(N=500, target_h=0.50),
        0.50
    ),
    # Case 2: Anti-persistent series (Target H = 0.25)
    (
        generate_fractal_series(N=500, target_h=0.25),
        0.25
    ),
    # Case 3: Too short array (below our estimator limit of 100 observations)
    (
        np.arange(50, dtype=float),
        ValueError
    ),
    # Case 4: Single element array
    (
        np.array([42.0]),
        ValueError
    ),
    # Case 5: Single element nested array
    (
        np.array([[42.0]]),
        ValueError
    ),
    # Case 6: Empty trajectory
    (
        np.array([]),
        ValueError
    ),
    # Case 7: High dimensional matrix instead of 1D array
    (
        np.ones((2, 100)),
        ValueError
    )
]


@pytest.mark.parametrize("input_run,expected_output", estimated_hurst_exponent_testdata)
def test_estimated_hurst_exponent(input_run, expected_output):
    """
    Tests the `estimated_hurst_exponent` function to ensure it correctly identifies
    exact fractional scaling behavior using true fBm trajectories.
    """
    if expected_output == ValueError or isinstance(expected_output, ValueError):
        with pytest.raises(ValueError):
            estimated_hurst_exponent(input_run)
    else:
        est_hurst = estimated_hurst_exponent(input_run)

        # Verify H stays within mathematical bounds [0.0, 1.0]
        assert 0.0 <= est_hurst <= 1.0, f"Hurst exponent {est_hurst} is outside valid bounds [0, 1]"

        # A tolerance of 0.15 accommodates statistical variance of the R/S 
        # algorithm on finite sample sizes (N=500).
        assert np.isclose(est_hurst, expected_output, atol=0.15), (
            f"Expected Hurst exponent near {expected_output}, got {est_hurst}"
        )