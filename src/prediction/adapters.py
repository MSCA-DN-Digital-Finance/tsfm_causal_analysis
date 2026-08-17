import numpy as np
import pandas as pd



#################### CHRONOS ADAPTERS ####################

def chronos_input_adapter(
    input_data: np.ndarray
        
):

    """
    Converts a 3D NumPy array (Samples, Time Steps, Dimensions) into a Pandas DataFrame
    formatted for Chronos time series forecasting.
    
    Parameters:
    - input_data: A 3D NumPy array of shape (n_samples, time_steps, n_features).
    
    Returns:
    - A Pandas DataFrame with columns ['item_id', 'timestamp', 'target'] where:
      - 'item_id' is a unique identifier for each sample (e.g., 0,1,2,...).
      - 'timestamp' is a datetime index starting from "1750-01-01" with hourly frequency.
      - 'target' contains the values from the input array.
    
    Note:
    The function assumes that the input data has at least one sample and one time step.
    It does not perform bounds checking or handle edge cases.
    """

    n_samples, time_steps, n_features = input_data.shape

    # Create a DataFrame to hold the results
    df_list = []
    for i in range(n_samples):
        timestamps = pd.date_range("1750-01-01", periods=time_steps, freq="h")
        for j in range(time_steps):
            # Start with the base columns
            row = {
                "item_id": i,
                "timestamp": timestamps[j],
                "target": input_data[i, j, 0]
            }
            
                
            df_list.append(row)
            
    return pd.DataFrame(df_list)


def chronos_output_adapter(pred_df: pd.DataFrame) -> np.ndarray:
    """
    Converts a Chronos prediction DataFrame back into a 3D NumPy array format (Samples, Time Steps, Dimensions).
    
    Parameters:
    - pred_df: A Pandas DataFrame with columns ['item_id', 'timestamp', 'target_name', 'predictions'] and possibly additional feature columns.
    
    Returns:
    - A 3D NumPy array of shape (n_samples, time_steps, n_features) where:
      - n_samples is the number of unique item_ids.
      - time_steps is the number of unique timestamps per item_id.
      - n_features is 1.
    
    Note:
    The function assumes that the input DataFrame is properly formatted and does not perform extensive error checking.
    """

# 1. Extract the unique IDs to find N
    n_samples = pred_df['item_id'].nunique()
    
    # 2. Extract only the prediction values
    # We sort by item_id then timestamp to ensure the flat array 
    # matches the (N, T) order perfectly.
    values = pred_df.sort_values(['item_id', 'timestamp'])['predictions'].values
    
    # 3. Reshape into (N, T, 1)
    # -1 tells numpy to figure out the 'T' dimension automatically
    return values.reshape(n_samples, -1, 1).astype(np.float32)


##################### TIMESFM ADAPTERS ####################

def timesfm_input_adapter(input_data: np.ndarray) -> list[np.ndarray]:
    """
    Adapts a 3D multivariate batch tensor into a list of univariate sequences 
    compatible with the TimesFM model input requirements.

    This function reshapes data from a sample-centric 3D format into a 
    flattened list of 1D arrays.
    Args:
        input_data (np.ndarray): A 3D NumPy array of shape (N, T, D), where:
            - N: Number of samples (batch size).
            - T: Number of time steps (context length).
            - D: Number of features (dimensions == 1).

    Returns:
        list[np.ndarray]: A list of length (N * D), where each element is a 
            1D NumPy array of shape (T,) and type float32.
    """

    N, T, D = input_data.shape
            
    # 1. Flatten N and D into a single dimension: (N * D, T)
    # We swap axes so that the time steps (T) remain contiguous
    # Resulting shape: (N*D, T)
    flattened_series = input_data.swapaxes(1, 2).reshape(N * D, T)
    
    # 2. Convert to the list of float32 arrays TimesFM expects
    inputs_list = [row.astype(np.float32) for row in flattened_series]
    
    return inputs_list 

def timesfm_output_adapter(input_list: list[np.ndarray]) -> np.ndarray:
    """
    Adapts a list of univariate sequences from TimesFM output back into a 3D batch tensor.

    This function takes the "Long Format" list produced by the model and restores 
    the sample-centric 3D structure. While currently configured for D=1, it is 
    structured to be the inverse of the input adapter.

    Args:
        input_list (list[np.ndarray]): A list of length N, where each element 
            is a 1D NumPy array of shape (T,).
    
    Returns:
        np.ndarray: A 3D NumPy array of shape (N, T, D) where:
            - N: Number of samples (inferred from list length).
            - T: Number of time steps (inferred from array length).
            - D: 1 (Univariate dimension).
    """
    # 1. Stack the list of 1D arrays into a 2D array of shape (N, T)
    # 2. Reshape to (N, T, 1) to satisfy the project's 3D tensor requirement
    N = len(input_list)
    T = input_list[0].shape[0]
    D = 1 

    return np.array(input_list).reshape(N, T, D)

