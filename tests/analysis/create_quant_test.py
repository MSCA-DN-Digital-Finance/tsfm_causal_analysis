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
                "param_stat_of": "trajectory",
                "param_stat_value": x,
            }
        )

        # Evaluated model
        df_model = pd.DataFrame(
            {
                "experiment_id": exp_id,
                "generator_name": gen_name,
                "param_stat_of": "model_v1",
                "param_stat_value": model_values(x),
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
            "param_stat_of": ["model_v1", "model_v1"],
            "param_stat_value": [1.0, 2.0],
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


######################### Tests for constrained_piecewise #########################


def test_breakpoint_continuity():
    """Verify continuity directly at the breakpoint x = x_break."""
    x_break = 5.0
    slope_2 = 2.5

    # At x = x_break, y must equal x_break
    result = constrained_piecewise(x_break, x_break=x_break, slope_2=slope_2)
    assert np.isclose(result, x_break)


@pytest.mark.parametrize("slope_2", [0.0, 0.5, 1.0, 2.0, -1.5])
def test_segment_1_identity(slope_2):
    """Verify Segment 1 (x <= x_break) strictly returns y = x regardless of slope_2."""
    x_break = 10.0
    x_vals = np.array([-5.0, 0.0, 5.0, 10.0])

    result = constrained_piecewise(x_vals, x_break=x_break, slope_2=slope_2)
    np.testing.assert_array_equal(result, x_vals)


@pytest.mark.parametrize(
    "slope_2, expected_y2",
    [
        (0.0, 10.0),  # Flat after break: 10 + 0*(15-10) = 10
        (0.5, 12.5),  # Attenuated: 10 + 0.5*5 = 12.5
        (2.0, 20.0),  # Amplified:  10 + 2.0*5 = 20
        (-1.0, 5.0),  # Inverted:   10 - 1.0*5 = 5
    ],
)
def test_segment_2_slopes(slope_2, expected_y2):
    """Verify Segment 2 (x > x_break) correctly computes the second linear segment."""
    x_break = 10.0
    x_val = 15.0  # x - x_break = 5.0

    result = constrained_piecewise(x_val, x_break=x_break, slope_2=slope_2)
    assert np.isclose(result, expected_y2)


def test_numpy_array_vectorization():
    """Verify array inputs evaluated across both segments simultaneously."""
    x_break = 2.0
    slope_2 = 3.0
    x_arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    # Expected:
    # x <= 2 -> [0, 1, 2]
    # x > 2  -> 2 + 3*(3-2) = 5, 2 + 3*(4-2) = 8
    expected = np.array([0.0, 1.0, 2.0, 5.0, 8.0])

    result = constrained_piecewise(x_arr, x_break=x_break, slope_2=slope_2)
    np.testing.assert_array_equal(result, expected)


def test_pandas_series_compatibility():
    """Verify compatibility with pandas Series objects."""
    x_series = pd.Series([1.0, 5.0, 10.0])
    result = constrained_piecewise(x_series, x_break=5.0, slope_2=0.5)

    # 1.0 <= 5 -> 1.0
    # 5.0 <= 5 -> 5.0
    # 10.0 > 5 -> 5 + 0.5*(10-5) = 7.5
    expected = np.array([1.0, 5.0, 7.5])
    np.testing.assert_array_equal(result, expected)



######################### Tests for calculate_aic #########################

def test_calculate_aic_standard_computation():
    """Verify standard AIC computation against explicit formula result."""
    n, ss_res, k = 100, 250.0, 3

    # Formula: n * ln(ss_res / n) + 2 * k
    # 100 * ln(2.5) + 2 * 3 = 100 * 0.9162907318741551 + 6 = 97.62907318741552
    expected = 100 * np.log(250.0 / 100) + 2 * 3
    result = calculate_aic(n, ss_res, k)

    assert np.isclose(result, expected)


@pytest.mark.parametrize("invalid_ss_res", [0.0, -1.0, -100.5])
def test_calculate_aic_invalid_ss_res_returns_nan(invalid_ss_res):
    """Verify that ss_res <= 0 returns np.nan (prevents log(0) / log of negative number)."""
    result = calculate_aic(n=50, ss_res=invalid_ss_res, k=2)
    assert np.isnan(result)


def test_calculate_aic_penalizes_higher_complexity():
    """Verify that adding parameters (higher k) increases AIC for equal fit."""
    n, ss_res = 100, 50.0

    aic_simple = calculate_aic(n, ss_res, k=2)
    aic_complex = calculate_aic(n, ss_res, k=5)

    assert aic_complex > aic_simple
    # Difference should be exactly 2 * (5 - 2) = 6
    assert np.isclose(aic_complex - aic_simple, 6.0)


def test_calculate_aic_rewards_lower_residual():
    """Verify that lower residual error (lower ss_res) produces a lower AIC."""
    n, k = 100, 3

    aic_better_fit = calculate_aic(n, ss_res=10.0, k=k)
    aic_worse_fit = calculate_aic(n, ss_res=50.0, k=k)

    assert aic_better_fit < aic_worse_fit


def test_calculate_aic_accepts_integer_and_float_types():
    """Verify function handles mixed numeric types cleanly."""
    result = calculate_aic(n=10, ss_res=20, k=1)
    assert isinstance(result, float)
    assert not np.isnan(result)




########################### Tests for get_breakpoint_from_df #########################

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def make_df():
    """Factory fixture to build synthetic experiment DataFrames."""

    def _builder(y_func, exp_id="exp_1", gen_name="gen_A", model_name="model_v1"):
        np.random.seed(42)
        x = np.linspace(1, 20, 40)

        df_traj = pd.DataFrame(
            {
                "experiment_id": exp_id,
                "generator_name": gen_name,
                "param_stat_of": "trajectory",
                "param_stat_value": x,
            }
        )

        df_model = pd.DataFrame(
            {
                "experiment_id": exp_id,
                "generator_name": gen_name,
                "param_stat_of": model_name,
                "param_stat_value": y_func(x),
            }
        )

        return pd.concat([df_traj, df_model], ignore_index=True)

    return _builder


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_missing_trajectory_returns_empty_summary():
    """Verify groups without ground truth trajectory are skipped."""
    df_no_traj = pd.DataFrame(
        {
            "experiment_id": ["exp_1"] * 2,
            "generator_name": ["gen_A"] * 2,
            "param_stat_of": ["model_1", "model_1"],
            "param_stat_value": [1.0, 2.0],
        }
    )

    res = get_breakpoint_from_df(df_no_traj)
    assert res.empty


def test_accepted_breakpoint(make_df):
    """Verify synthetic piecewise data triggers breakpoint acceptance."""
    # Break at x=10 with slope change from 1.0 to 0.2 + low noise
    np.random.seed(42)

    def pw_func(x):
        return constrained_piecewise(x, x_break=10.0, slope_2=0.2) + np.random.normal(
            0, 0.05, len(x)
        )

    df = make_df(pw_func)
    res = get_breakpoint_from_df(df)

    assert len(res) == 1
    assert res.iloc[0]["Break Point (Yes/No)"] == "Yes"
    assert res.iloc[0]["Piecewise Model Assumption"] == "Accepted"
    assert res.iloc[0]["Breakpoint (x*)"] != "N/A"
    assert float(res.iloc[0]["Breakpoint (x*)"]) == pytest.approx(10.0, abs=0.5)


def test_rejected_calibrated_when_slope_near_one(make_df):
    """Verify perfect y = x relationship triggers 'Rejected'."""

    def linear_func(x):
        return x  # Slope = 1.0, no true breakpoint

    df = make_df(linear_func)
    res = get_breakpoint_from_df(df)

    assert len(res) == 1
    assert res.iloc[0]["Break Point (Yes/No)"] == "No"
    assert res.iloc[0]["Piecewise Model Assumption"] == "Rejected"
    assert res.iloc[0]["Breakpoint (x*)"] == "N/A"


def test_rejected_high_delta_aic(make_df):
    """Verify simple linear offset model favors linear regression over piecewise."""
    # y = 2x + 5 (Piecewise model will fit poorly compared to standard linear)
    np.random.seed(42)

    def pure_linear(x):
        return 2.0 * x + 5.0 + np.random.normal(0, 0.01, len(x))

    df = make_df(pure_linear)
    res = get_breakpoint_from_df(df)

    assert len(res) == 1
    assert res.iloc[0]["Break Point (Yes/No)"] == "No"
    assert res.iloc[0]["Piecewise Model Assumption"] == "Rejected"


