"""
This file contains unit tests for the functions in the `analysis.create_quant` module.
"""

import os
import sys
import numpy as np
import pytest
import statsmodels.api as sm

# Get the absolute path to the 'src' directory
# This looks up two levels from the current test file
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.append(src_path)

from analysis.create_quant import *
from analysis.breakpoint_funcs import *

######################### Tests for get_bias_from_df #########################

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def base_df_builder():
    """Factory fixture to build structured DataFrames for tests."""

    def _make_df(model_values, exp_id="exp_1", gen_name="gen_A"):
        np.random.seed(42)
        x = np.linspace(10, 100, 50)

        # Ground truth
        df_traj = pd.DataFrame(
            {
                "experiment_id": exp_id,
                "generator_name": gen_name,
                "param_estimate_of": "trajectory",
                "param_estimate_value": x,
            }
        )

        # Evaluated model
        df_model = pd.DataFrame(
            {
                "experiment_id": exp_id,
                "generator_name": gen_name,
                "param_estimate_of": "model_v1",
                "param_estimate_value": model_values(x),
            }
        )

        return pd.concat([df_traj, df_model], ignore_index=True)

    return _make_df


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_empty_or_mismatched_data_returns_empty_summary():
    """Verify groups with missing baseline or unequal lengths are cleanly skipped."""
    df_missing_traj = pd.DataFrame(
        {
            "experiment_id": ["exp_1"] * 2,
            "generator_name": ["gen_A"] * 2,
            "param_estimate_of": ["model_v1", "model_v1"],
            "param_estimate_value": [1.0, 2.0],
        }
    )

    res = get_bias_from_df(df_missing_traj)
    assert res.empty


def test_unbiased_perfect_fit(base_df_builder):
    """Verify an exact y = x relationship accepts H0 and classifies as unbiased."""
    df = base_df_builder(lambda x: x)
    res = get_bias_from_df(df)

    assert len(res) == 1
    assert res.iloc[0]["Decision"] == "Accept (Unbiased)"
    assert res.iloc[0]["Bias Classification"] == "N/A"
    assert res.iloc[0]["Intercept (b0)"] == 0.0
    assert res.iloc[0]["Slope (b1)"] == 1.0


@pytest.mark.parametrize(
    "func, expected_decision, expected_bias",
    [
        # Positive Shift (b0 > 0)
        (lambda x: x + 10.0, "Reject (Biased)", "Positive Shift"),
        # Negative Shift (b0 < 0)
        (lambda x: x - 10.0, "Reject (Biased)", "Negative Shift"),
        # Attenuation (b1 < 1)
        (lambda x: 0.5 * x, "Reject (Biased)", "Attenuation"),
        # Amplification (b1 > 1)
        (lambda x: 1.5 * x, "Reject (Biased)", "Amplification"),
        # Combined: Negative Shift + Attenuation
        (lambda x: 0.5 * x - 5.0, "Reject (Biased)", "Negative Shift, Attenuation"),
    ],
)
def test_bias_classifications(
    base_df_builder, func, expected_decision, expected_bias
):
    """Parametrized test for distinct bias conditions."""
    # Add minimal gaussian noise to ensure p-value drops below 0.05 cutoff
    np.random.seed(42)
    df = base_df_builder(lambda x: func(x) + np.random.normal(0, 0.01, len(x)))

    res = get_bias_from_df(df)

    assert res.iloc[0]["Decision"] == expected_decision
    assert res.iloc[0]["Bias Classification"] == expected_bias


def test_high_r2_tolerance_override(base_df_builder):
    """Verify numerical tolerance guard catches micro-floating-point artifacts."""
    # Slight precision drift where R2 >= 0.9999 and b0 ~ 0, b1 ~ 1
    df = base_df_builder(lambda x: x + 0.0001)
    res = get_bias_from_df(df)

    assert res.iloc[0]["Decision"] == "Accept (Unbiased)"
    assert res.iloc[0]["Bias Classification"] == "N/A"
    assert res.iloc[0]["Wald Test p-value"] == "1.00e+00"


