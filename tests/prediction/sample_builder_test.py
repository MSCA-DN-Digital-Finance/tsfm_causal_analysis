import pytest
import numpy as np


import sys
import os

# Get the absolute path to the 'src' directory
# This looks up two levels from the current test file
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.append(src_path)

from prediction.sample_builder import build_samples


testdata = [
    # Tensor is (Samples, Time Steps, Dimensions)


    # Case 0: 1 sample, 1 dimension
    (
        np.array([0, 1, 2, 3, 4, 5]),
        np.array([[[0], [1], [2], [3], [4], [5]]]), # (1, 6, 1)
    ),

]


@pytest.mark.parametrize("traj,x_val", testdata)
def test_sample_creation(traj, x_val):

    x = build_samples(traj)
    assert x.shape == x_val.shape, f"Expected shape {x_val.shape}, got {x.shape}"
    assert np.array_equal(x, x_val)
