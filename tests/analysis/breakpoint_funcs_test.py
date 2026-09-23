# This file contains the test cases for functions in the breakpoint_funcs.py file in src.



import numpy as np
import pytest
import os
import sys 

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.append(src_path)

from analysis.breakpoint_funcs import *

# 1. Helper generator functions to isolate data creation
def generate_no_break():
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = x + np.random.normal(0, 0.5, size=x.shape)
    return x, y, 5.0 # last value is breakpoint lower bound input 

def generate_break_at_5():
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = np.piecewise(
        x,
        [x < 5, x >= 5],
        [lambda x: x + np.random.normal(0, 0.5, size=x.shape),
         lambda x: 2 * x - 5 + np.random.normal(0, 0.5, size=x.shape)]
    )
    return x, y, 5.0

def generate_break_at_7():
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = np.piecewise(
        x,
        [x < 7, x >= 7],
        [lambda x: x + np.random.normal(0, 0.5, size=x.shape),
         lambda x: -x + 14 + np.random.normal(0, 0.5, size=x.shape)]
    )
    return x, y, 7.0

def generate_sharp_break_at_5():
    np.random.seed(0)
    x = np.linspace(0, 10, 100)
    y = np.piecewise(
        x, 
        [x < 5.0, x >= 5.0],
        [lambda x: 2.0 * x + 0.0 + np.random.normal(0, 0.5, size=x.shape), 
         lambda x: -2.0 * x + 20.0 + np.random.normal(0, 0.5, size=x.shape)]
    )
    return x, y, 5.0




### Tests for first_seg_wald() ###

# 2. Parametrized Test Function
@pytest.mark.parametrize(
    ("dataset_fn", "expectation"),
    [
        (generate_no_break, True),
        (generate_break_at_5,True), 
        (generate_break_at_7,True),
        (generate_sharp_break_at_5,False)
    ],
    ids=["no_break", "break_at_5", "break_at_7", "sharp_break_at_5"]  # Gives clean names in test logs
    
)


def test_first_seg_wald(dataset_fn, expectation):
    "This functions tests whether first_seg_wald() returns correct boolean when breached and not"
    x, y, bp = dataset_fn()

    wald_res = first_seg_wald(x,y,bp)

    if wald_res.pvalue < 0.05:
        reject = False
    else:
        reject = True
    
    # Assert that cusum output matches expected output
    assert reject == expectation