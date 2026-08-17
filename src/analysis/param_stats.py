"""
This file contains functions used to compute the parameter statistics for each generator.
""" 

from typing import Any
import numpy as np
from scipy import signal
from statsmodels.tsa.ar_model import AutoReg
import ruptures as rpt
from hurst import compute_Hc


def _ensure_1d_trajectory(trajectory: Any) -> np.ndarray:
    """
    Safely converts input to a 1D float NumPy array.
    Prevents 1-element inputs from collapsing into 0D arrays after squeezing.
    """
    if trajectory is None:
        raise ValueError("Trajectory cannot be None")
        
    arr = np.asarray(trajectory, dtype=float).squeeze()
    
    # If a 1-element array got squeezed down to a 0D scalar, restore it to 1D
    if arr.ndim == 0:
        arr = np.atleast_1d(arr)
        
    if arr.ndim > 1:
        raise ValueError(f"Expected 1D trajectory array, got shape {arr.shape}")
        
    return arr


def estimated_mean(trajectory: np.ndarray) -> float:
    """
    Computes the estimated mean change between consecutive time step values 
    in the trajectory to be used as an intervention parameter.

    Args:
      - trajectory: sequence of time series values (1D array)

    Returns:
      - float: The average magnitude and direction of changes in the trajectory.
    """
    try:
        arr = _ensure_1d_trajectory(trajectory)
    except (TypeError, ValueError) as e:
        raise ValueError("Invalid input data type for 'trajectory' or empty array") from e

    if len(arr) <= 1:
        raise ValueError("Invalid input data type for 'trajectory' or empty array")

    # Calculate the actual differences between consecutive steps
    differences = arr[1:] - arr[:-1]
    
    # Compute the mean of those differences
    mean_change = float(np.mean(differences))
    
    if np.isnan(mean_change):
        raise ValueError("Invalid input data type for 'trajectory'")

    return mean_change


def estimated_beta(trajectory: np.ndarray) -> float:
    """
    Computes the model-implied AR(1) beta for a given run.
    
    Args:
      - trajectory  : forecasted trajectory

    Returns:
        - float: The estimated AR(1) beta coefficient.  
    """
    arr = _ensure_1d_trajectory(trajectory)
        
    if len(arr) <= 1:
        raise ValueError("Input arrays must contain more than 1 observation to fit AR(1).")
    
    # Fit an AR model with a lag of 1
    model = AutoReg(arr, lags=1, trend='c').fit()

    # Extract the beta coefficient for the first lag
    beta = float(model.params[1]) 

    return beta


def estimated_wavelength(trajectory: np.ndarray) -> float:
    """
    Computes the dominant wavelength of the trajectory.

    Args:
      - trajectory  : forecasted trajectory

    Returns:
        - float: The estimated wavelength of the trajectory.
    """
    arr = _ensure_1d_trajectory(trajectory)

    if len(arr) <= 1:
        raise ValueError("Input arrays must contain more than 1 observation to calculate wavelength.")

    # Compute power spectrum density using Welch's method
    frequencies, power = signal.welch(arr, fs=1)
    dominant_freq = float(frequencies[np.argmax(power)])

    # Convert frequency to wavelength (assuming unitary sampling rate)
    estimated_wavelength = 1.0 / dominant_freq if dominant_freq != 0 else float('inf')

    return estimated_wavelength


def estimated_dwell_time(trajectory: np.ndarray, penalty: float = 1.5) -> float:
    """
    Detects change points in a noisy trend using PELT and 
    calculates the average dwell time.

    Args:
      - trajectory : sequence of time series values (1D array)
      - penalty    : PELT penalty complexity parameter (sensitivity)

    Returns:
      - float: The average dwell time.
    """
    arr = _ensure_1d_trajectory(trajectory)

    if len(arr) <= 2:
        raise ValueError("Input arrays must contain more than 2 observations to calculate differences and dwell times.")
    
    # 1. Convert trend to differences (slopes)
    # This turns 'slope changes' into 'mean changes'
    signal_diff = np.diff(arr)
    
    # 2. Configure PELT
    # 'l2' (Least Squares) is best for shifts in the mean
    algo = rpt.Pelt(model="l2", jump=1).fit(signal_diff)
    
    # 3. Predict change points
    # The 'pen' value is the sensitivity. 
    result = algo.predict(pen=penalty)
    
    # 4. Calculate Dwell Times
    change_points = [0] + result
    dwell_times = np.diff(change_points)
    
    avg_dwell = float(np.mean(dwell_times))
        
    return avg_dwell


def estimated_threshold(trajectory: np.ndarray) -> float:
    """
    Finds the reset points (peaks) in an Integrate-and-Fire series and returns 
    their average height. Falls back to global max if no drops are found.

    Args:
      - trajectory: sequence of time series values (1D array)

    Returns:
      - float: The estimated threshold parameter.
    """
    arr = _ensure_1d_trajectory(trajectory)
    
    if len(arr) == 0:
        raise ValueError("Input arrays must not be empty.")
    if len(arr) == 1:
        return float(arr[0])

    # Find indices where the value drops significantly (the reset)
    diffs = np.diff(arr)
    
    # A large negative jump indicates a reset
    peak_indices = np.where(diffs < -0.5 * np.max(arr))[0]
    
    if len(peak_indices) == 0:
        return float(np.max(arr)) # Fallback to global max
        
    # The peak is the value right before the jump
    peaks = arr[peak_indices]
    return float(np.mean(peaks))





def estimated_hurst_exponent(trajectory: np.ndarray) -> float:
    """
    Estimates the Hurst exponent (H) of a trajectory using the 'hurst' library.
    H close to 0.5 is a standard random walk.
    H > 0.5 indicates persistent (trending) behavior.
    H < 0.5 indicates anti-persistent (mean-reverting) behavior.

    Args:
      - trajectory: sequence of time series values (1D array)

    Returns:
      - float: The estimated Hurst exponent clamped between 0 and 1.
    """
    arr = _ensure_1d_trajectory(trajectory)
    
    # The hurst package requires a minimum number of points to perform 
    # linear regression across sub-intervals. 
    if len(arr) < 100:
        raise ValueError("Trajectory is too short to accurately estimate the Hurst exponent (need N >= 100).")

    try:
        # kind='random_walk' expects a series representing a cumulative sum / integrated path
        H, _, _ = compute_Hc(arr, kind='random_walk', simplified=True)
    except Exception as e:
        raise ValueError(f"Could not compute Hurst exponent: {str(e)}")

    # Safety clamp to keep it within mathematical bounds [0, 1]
    return float(np.clip(H, 0.0, 1.0))


# Set up parameter statistics registry
PARAM_STATS_REGISTRY = {
    "estimated_mean": estimated_mean,
    "estimated_beta": estimated_beta,
    "estimated_wavelength": estimated_wavelength,
    "estimated_dwell_time": estimated_dwell_time,
    "estimated_threshold": estimated_threshold,
    "estimated_hurst_exponent": estimated_hurst_exponent
}