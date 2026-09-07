"""
This file contains the functions to create tables for the bias and break point quantification from the aggregated dataset created by create_dataset.py.

"""


import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import f as f_dist
import statsmodels.api as sm
from pathlib import Path
import sys

# Calculate the project root (2 levels up from src/prediction/)
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)


INPUT_FILE = Path(project_root) / "artifacts" / "dataset" / "aggregated_dataset.csv"
OUTPUT_DIR = Path(project_root) / "artifacts" / "dataset"


#### Bias Analysis Functions ####

def get_bias_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs Mincer-Zarnowitz regression across experiments and returns bias diagnostics.
    """
    results = []

    for (exp_id, gen_name), group in df.groupby(['experiment_id', 'generator_name']):
        # Ground truth baseline
        x_raw = group[group['param_stat_of'] == 'trajectory']['param_stat_value'].values
        
        for model_name in df['param_stat_of'].unique().tolist():
            if model_name == 'trajectory':
                continue
            y = group[group['param_stat_of'] == model_name]['param_stat_value'].values
            
            if len(x_raw) == 0 or len(y) == 0 or len(x_raw) != len(y):
                continue
                
            # Fit OLS
            X = sm.add_constant(x_raw)
            res = sm.OLS(y, X).fit()
            
            b0 = res.params[0]
            b1 = res.params[1]
            r2 = res.rsquared
            ci = res.conf_int(alpha=0.05)
            
            b0_ci = f"[{ci[0][0]:.4f}, {ci[0][1]:.4f}]"
            b1_ci = f"[{ci[1][0]:.4f}, {ci[1][1]:.4f}]"
            
            # Wald Test (H0: const = 0, x1 = 1)
            wald = res.wald_test("const = 0, x1 = 1")
            p_val = float(wald.pvalue)


            # Set default decision and bias classification
            decision = "Accept (Unbiased)"
            bias_type = "N/A"

            # Numerical Tolerance Catch for float precision artifacts (e.g., Exp 3)
            if r2 >= 0.9999 and np.isclose(b0, 0, atol=1e-3) and np.isclose(b1, 1, atol=1e-3):
                p_val = 1.0
                
            elif p_val < 0.05:
                decision = "Reject (Biased)"
                
                # Classify specific operational bias
                b0_biased = not (ci[0][0] <= 0 <= ci[0][1])
                b1_biased = not (ci[1][0] <= 1 <= ci[1][1])
                

                if b0_biased:
                    if b0 < 0:
                        b0_bias = "Negative Shift"
                    else:
                        b0_bias = "Positive Shift"
                if b1_biased:
                    if b1 < 1:
                        b1_bias = "Attenuation"
                    else:
                        b1_bias = "Amplification"

                if b0_biased and b1_biased:
                    bias_type = f"{b0_bias}, {b1_bias}"
                if b0_biased and not b1_biased:
                    bias_type = f"{b0_bias}"
                if not b0_biased and b1_biased:
                    bias_type = f"{b1_bias}"

            
            results.append({
                'Experiment': exp_id,
                'Generator': gen_name,
                'Model': model_name,
                'Intercept (b0)': round(b0, 4),
                '95% CI (b0)': b0_ci,
                'Slope (b1)': round(b1, 4),
                '95% CI (b1)': b1_ci,
                'R2': round(r2, 3),
                'Wald Test p-value': f"{p_val:.2e}",
                'Decision': decision,
                'Bias Classification': bias_type
            })

    summary_df = pd.DataFrame(results)
    return summary_df


#### Break-Point Analysis Functions ####



def constrained_piecewise(x, x_break, slope_2):
    """
    Segment 1 (x <= x_break): y = x (Fixed slope=1, intercept=0)
    Segment 2 (x > x_break):  y = x_break + slope_2 * (x - x_break)
    """
    # Convert x to a numpy array
    x = np.asarray(x)

    return np.piecewise(
        x,
        [x <= x_break, x > x_break],
        [lambda x: x, lambda x: x_break + slope_2 * (x - x_break)]
    )


def calculate_aic(n: int, ss_res: float, k: int) -> float:
    """Calculates Akaike Information Criterion (AIC)."""
    if ss_res <= 0:
        return np.nan
    return n * np.log(ss_res / n) + 2 * k


def get_breakpoint_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight Constrained Piecewise & Model Comparison Analysis.
    Returns a consolidated Pandas DataFrame with linear R2 comparisons and renamed columns.
    Processes all experiments and returns the updated breakpoint summary DataFrame.
    """
    results = []

    grouped = df.groupby(['experiment_id', 'generator_name'])
    
    for (exp_id, gen_name), group in grouped:
        x_raw = group[group['param_stat_of'] == 'trajectory']['param_stat_value'].values
        if len(x_raw) == 0:
            continue

        for model_name in df['param_stat_of'].unique().tolist():
            if model_name == 'trajectory':
                continue
            y_raw = group[group['param_stat_of'] == model_name]['param_stat_value'].values
            if len(y_raw) == 0 or len(x_raw) != len(y_raw):
                continue

            n_obs = len(y_raw)
            p0 = [np.median(x_raw), 0.5]
            bounds = ([x_raw.min(), -np.inf], [x_raw.max(), np.inf])

            # Total sum of squares for R2
            ss_tot = np.sum((y_raw - np.mean(y_raw)) ** 2)

            # 1. Null Model (y = x)
            ss_res_ideal = np.sum((y_raw - x_raw) ** 2)

            # 2. Standard Linear Model (y = b0 + b1*x)
            poly_coeffs = np.polyfit(x_raw, y_raw, deg=1)
            y_pred_linear = np.polyval(poly_coeffs, x_raw)
            ss_res_linear = np.sum((y_raw - y_pred_linear) ** 2)
            aic_linear = calculate_aic(n_obs, ss_res_linear, k=2)
            r2_linear = 1 - (ss_res_linear / ss_tot) if ss_tot != 0 else np.nan

            try:
                # 3. Constrained Piecewise Model
                popt, pcov = curve_fit(constrained_piecewise, x_raw, y_raw, p0=p0, bounds=bounds)
                x_break_opt, slope_2_opt = popt
                
                y_pred_pw = constrained_piecewise(x_raw, x_break_opt, slope_2_opt)
                ss_res_pw = np.sum((y_raw - y_pred_pw) ** 2)
                aic_pw = calculate_aic(n_obs, ss_res_pw, k=2)
                r2_pw = 1 - (ss_res_pw / ss_tot) if ss_tot != 0 else np.nan

                # Hypothesis Tests & Metrics
                df_num, df_denom = 2, n_obs - 2
                f_stat = ((ss_res_ideal - ss_res_pw) / df_num) / (ss_res_pw / df_denom) if ss_res_pw < ss_res_ideal else 0
                davies_p = f_dist.sf(f_stat, df_num, df_denom) if f_stat > 0 else 1.0

                delta_aic = aic_pw - aic_linear

                # 95% Confidence Interval Calculation
                perr = np.sqrt(np.diag(pcov))
                bp_ci_low = x_break_opt - 1.96 * perr[0]
                bp_ci_high = x_break_opt + 1.96 * perr[0]

                # Model Selection & Decision Logic
                if r2_pw < 0 or delta_aic > 2:
                    pw_assumption = "Rejected"
                    breakpoint_yes_no = "No"
                    x_str, ci_str = "N/A", "N/A"
                elif davies_p >= 0.05 or np.isclose(slope_2_opt, 1.0, atol=1e-3):
                    pw_assumption = "Rejected"
                    breakpoint_yes_no = "No"
                    x_str, ci_str = "N/A", "N/A"
                else:
                    pw_assumption = "Accepted"
                    breakpoint_yes_no = "Yes"
                    x_str = f"{x_break_opt:.2f}"
                    ci_str = f"[{bp_ci_low:.2f}, {bp_ci_high:.2f}]"

                results.append({
                    'Experiment': exp_id,
                    'Generator': gen_name,
                    'Model': model_name,
                    'R2 PW': round(r2_pw, 3),
                    'Post-Slope (a2)': round(slope_2_opt, 4),
                    'R2 Linear': round(r2_linear, 3),
                    'Delta AIC': f"{delta_aic:+.2f}",
                    'Piecewise Model Assumption': pw_assumption,
                    'Davies Test p-value': f"{davies_p:.2e}",
                    'Break Point (Yes/No)': breakpoint_yes_no,
                    'Breakpoint (x*)': x_str,
                    '95% CI (x*)': ci_str
                })

            except Exception:
                results.append({
                    'Experiment': exp_id,
                    'Generator': gen_name,
                    'Model': model_name,
                    'R2 PW': "N/A",
                    'Post-Slope (a2)': "N/A",
                    'R2 Linear': round(r2_linear, 3),
                    'Delta AIC': "N/A",
                    'Piecewise Model Assumption': "Rejected",
                    'Davies Test p-value': "N/A",
                    'Break Point (Yes/No)': "No",
                    'Breakpoint (x*)': "N/A",
                    '95% CI (x*)': "N/A"
                })

    summary_df = pd.DataFrame(results)
    return summary_df

### main function to run the analysis and save results ###

def main():
    # Load the aggregated dataset
    print(f"Loading aggregated dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)

    # Run bias analysis
    print("Running bias quantification...")
    bias_df = get_bias_from_df(df)
    bias_output_path = OUTPUT_DIR / "bias_quantification.csv"
    bias_df.to_csv(bias_output_path, index=False)
    print(f"Bias quantification saved to {bias_output_path}")

    # Run breakpoint analysis
    print("Running breakpoint quantification...")
    breakpoint_df = get_breakpoint_from_df(df)
    breakpoint_output_path = OUTPUT_DIR / "breakpoint_quantification.csv"
    breakpoint_df.to_csv(breakpoint_output_path, index=False)
    print(f"Breakpoint quantification saved to {breakpoint_output_path}")

if __name__ == "__main__":
    main()
