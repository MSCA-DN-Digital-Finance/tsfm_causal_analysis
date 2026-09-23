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

from breakpoint_funcs import *

# Calculate the project root (2 levels up from src/prediction/)
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)



INPUT_FILE = Path(project_root) / "artifacts" / "dataset" / "aggregated_dataset.csv"
OUTPUT_DIR = Path(project_root) / "artifacts" / "dataset"
FILE_DIR = OUTPUT_DIR / "breakpoint_files"


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
            wald = res.wald_test("const = 0, x1 = 1", use_f=False, scalar=True)
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
    breakpoint_df = breakpoint_runner(df, str(FILE_DIR))
    breakpoint_output_path = OUTPUT_DIR / "breakpoint_quantification.csv"
    breakpoint_df.to_csv(breakpoint_output_path, index=False)
    print(f"Breakpoint quantification saved to {breakpoint_output_path}")

if __name__ == "__main__":
    main()
