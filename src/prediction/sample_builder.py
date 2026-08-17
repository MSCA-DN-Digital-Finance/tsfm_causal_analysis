import numpy as np

def build_samples(
    traj: np.ndarray
) -> np.ndarray:
    """
    Constructs input-output pairs for time series prediction.
    
    Parameters:
    - traj: 1D trajectory of length input_length.

    
    Returns:
    - x: NumPy array of shape (n_samples = 1, input_length, dim = 1) containing the input sequences.

    """
    x = np.expand_dims(traj, axis=(0, 2))  # Shape: (1, input_length, 1)

    return x