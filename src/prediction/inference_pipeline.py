import numpy as np


def inference_pipeline(model_spec: dict, data: np.ndarray, horizon: int) -> np.ndarray:
    """
    Orchestrates the end-to-end forecasting pipeline based on a provided model specification.

    This function abstracts away the differences between models (like TimesFM's 
    lists vs. Chronos' DataFrames) by executing the standardized adapters defined 
    within the passed model_spec.

    Args:
        model_spec (dict): The configuration dictionary containing the loader, 
                           adapters, and inference logic for a specific model.
        data (np.ndarray): The input tensor of shape (N, T, D).
        horizon (int): The number of future time steps to predict.

    Returns:
        np.ndarray: The predicted values as a 3D tensor of shape (N, horizon, D).
    """
    
    # 1. Model Initialization
    # Calls the 'loader' function from the spec. In a production setting, 
    # you might wrap this in a singleton/cache so it doesn't reload 
    # from disk every time.
    model = model_spec["loader"]()
    
    # 2. Input Adaptation (The "Pre-processor")
    # Converts your standard (N, T, D) array into the format the model 
    # expects (e.g., a list of 1D arrays for TimesFM or a DataFrame for Chronos)
    # as defined by the spec's input_adapter.
    formatted_in = model_spec["input_adapter"](data)
    
    # 3. Core Inference execution
    # Triggers the actual forward pass using the model and inference function 
    # provided in the spec. The output here is still in the model's native format.
    raw_out = model_spec["inference_fn"](model, formatted_in, horizon)
    
    # 4. Output Adaptation (The "Post-processor")
    # Translates the model's native output back into the project-standard 
    # 3D tensor (N, H, D) using the spec's output_adapter, ensuring 
    # consistency for downstream metrics.
    final_preds = model_spec["output_adapter"](raw_out)    
    return final_preds