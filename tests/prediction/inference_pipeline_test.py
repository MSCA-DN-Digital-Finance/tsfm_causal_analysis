import pytest
import numpy as np
import sys
import os

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
if src_path not in sys.path:
    sys.path.append(src_path)

from prediction.inference_pipeline import inference_pipeline

from prediction.adapters import chronos_input_adapter, chronos_output_adapter, timesfm_input_adapter, timesfm_output_adapter


############ INFERENCE PIPELINE TESTS ############

def loader(model_key: str = "mock model"):
    # Mock loader that returns a dummy model object
    return "mock_model"

def chronos_inference(model, formatted_input, horizon=3):
    # Mock model that adds target column as predictions column
    pred_df = formatted_input.copy()
    pred_df["predictions"] = pred_df["target"]
    return pred_df

def timesfm_inference(model, formatted_input, horizon=3):
    # Mock model that returns the input as point forecasts
     # TimesFM returns a tuple: (point_forecast, quantile_forecast)
    # point_forecast shape: (N, horizon)
  
    
    return formatted_input

testdata = [
    # Tensor is (Samples, Time Steps, Dimensions)


    # Case 0: Chronos adapters end-to-end test
    (
        np.array([[[0.], [1.], [2.], [3.], [4.]]]), # input sample
        loader,
        chronos_input_adapter,
        chronos_inference,
        chronos_output_adapter,
        np.array([[[0.0], [1.0], [2.0], [3.0], [4.0]]]) # expected output       

    ),
    # Case 1: TimesFM adapters end-to-end test
    (
        np.array([[[0.], [1.], [2.], [3.], [4.]]]), # input sample
        loader,
        timesfm_input_adapter,
        timesfm_inference,
        timesfm_output_adapter,
        np.array([[[0.], [1.], [2.], [3.], [4.]]]) # expected output       
    )

]


@pytest.mark.parametrize("input, loader, input_adapter, model, output_adapter, expected_output", testdata)
def test_inference_pipeline(input, loader, input_adapter, model, output_adapter, expected_output):

    """Tests the end-to-end prediction flow using the inference_pipeline function with mocked components."""
    
    spec = {
        "loader": loader,
        "input_adapter": input_adapter,
        "inference_fn": model,
        "output_adapter": output_adapter
    }
    actual_output = inference_pipeline(model_spec=spec, data=input, horizon=3)
    assert np.array_equal(actual_output, expected_output)


