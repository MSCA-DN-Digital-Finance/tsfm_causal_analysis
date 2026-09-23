# This file contains all the functions required for the breakpoint analysis.

import numpy as np
import piecewise_regression
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
import json
import pandas as pd
import os


def fit_piecewise(x:np.array , y: np.array, path: str, n_breakpoints=1, plot=True, verbose=True)-> tuple:
    """
    Fits an unconstrained piecewise linear regression using the 'piecewise-regression' library.

    Parameters:
        x_data (array-like): Independent variable data.
        y_data (array-like): Dependent variable data.
        n_breakpoints (int): Number of breakpoints to fit (default is 1).
        plot (bool): Whether to generate the scatter plot with regression lines.
        verbose (bool): Whether to print the summary statistics and extracted parameters.
        
    Returns:
        pw_fit.get_results() (dict): Extracted model parameters (alpha, beta0, beta1, beta2), fit object, and figure handle.
        pw_fit.summary() (dict): Text summary of model.
    """
    x_data = np.asarray(x, dtype=float)
    y_data = np.asarray(y, dtype=float)

    # 1. Initialize and fit the model using Muggeo's segmented method
    pw_fit = piecewise_regression.Fit(x_data, y_data, n_breakpoints=n_breakpoints)



    # 2. Plotting
    if plot:
        # Plot the data, fit, breakpoints and confidence intervals
        pw_fit.plot_data(color="grey", s=20)
        # Pass in standard matplotlib keywords to control any of the plots
        pw_fit.plot_fit(color="red", linewidth=4)
        pw_fit.plot_breakpoints()
        pw_fit.plot_breakpoint_confidence_intervals()
        plt.xlabel("Trajectory Parameter Statistic")
        plt.ylabel("Model Parameter Statistic")
        plt.savefig(path +  "/" + "pw_plot.png")
        plt.close()


    return pw_fit.get_results(), pw_fit.summary()


def first_seg_wald(x: np.array,y: np.array, breakpoint_lb: float)-> dict:
    """
    Wald test for hypothesis "Intercept = 0, Slope = 1". If rejected, the segment adheres to x=y.

    Parameters:
           x_data (array-like): Independent variable data.
           y_data (array-like): Dependent variable data.
           breakpoint_lb (float): Lower bound of breakpoint confidence interval.
  
    Returns:
        wald_res (dict): Extracted Wald test results.
    """


    x_first_seg = x[x <= breakpoint_lb]
    y_first_seg = y[x <= breakpoint_lb]

    # return None if first segment has less than 2 data points
    if len(x_first_seg) < 2:
        return None

    # Fit OLS model for first segment

    data = {"y": y_first_seg, "x": x_first_seg}

    # 2. Fit the OLS model
    model = smf.ols("y ~ x", data=data).fit()

    # 3. Wald test for Intercept = 0 and Slope (x) = 1
    hypothesis = "Intercept = 0, x = 1"
    wald_res = model.wald_test(hypothesis, scalar=True)
    wald_res_dict = {
            "statistic": float(np.squeeze(wald_res.statistic)),
            "pvalue": float(wald_res.pvalue),
            "df_denom": int(wald_res.df_denom) if wald_res.df_denom is not None else None,
            "df_num": int(wald_res.df_num) if wald_res.df_num is not None else None,
        }

    return wald_res_dict

def breakpoint_analysis(x_data: np.ndarray, y_data: np.ndarray, path: str, n_breakpoints=1, plot=True, verbose=True)-> dict:
    """
    Performs breakpoint analysis piecewise regression and Wald test on the first segment.

    Parameters:
        x_data (array-like): Independent variable data.
        y_data (array-like): Dependent variable data.
        n_breakpoints (int): Number of breakpoints to fit (default is 1).
        plot (bool): Whether to generate plots for each step.
        verbose (bool): Whether to print detailed outputs.

    Returns:
        dict: Outputs of  piecewise regression parameters and Wald test results for the first segment.
    """
   
    
    # Step 1: Piecewise Regression
    pw_results, pw_summary = fit_piecewise(x_data, y_data, path, n_breakpoints=n_breakpoints, plot=plot, verbose=verbose)

    # Check if Davies test confirms at least one breakpoint, if not skip Wald test
    if pw_results['davies'] < 0.05 and pw_results['converged'] == True:
    

        # Step 2: Get lower bound for breakpoint interval
        breakpoint_lb = pw_results["estimates"]["breakpoint1"]["confidence_interval"][0]


        # Step 3: Wald Test on First Segment
        wald_results = first_seg_wald(x_data, y_data, breakpoint_lb=breakpoint_lb)
        if wald_results == None:
            results = {
                                "piecewise_results": pw_results,
                                "wald_results": None,
                                "comment": 'Wald result is None due to exception in first_seg_wald()'
                            }

        else:
            results = {
                "piecewise_results": pw_results,
                "wald_results": wald_results,
                "comment": 'Hypothesis of at least one breakpoint not rejected and wald test for first segmented was computed.'
            }

        # Step 4: Save results dict as json file to path
        os.makedirs(path, exist_ok=True)
        file_path_results = path + "/" + "breakpoint_results.json"

        
        with open(file_path_results, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=4, ensure_ascii=False)

    else:
        results = {
                    "piecewise_results": pw_results,
                    "wald_results": None,
                    "comment": 'Wald result is None because piecewise linear regression rejected hypothesis of at least one breakpoint.'
                }

    # Step 5: Save summary to txt file to path

    file_path_summary = path + "/" + "pw_summary.txt"

    with open(file_path_summary, "w", encoding="utf-8") as file:
        file.write(pw_summary)

    return results


def breakpoint_runner(df: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Loads aggregated experimental data and applies breakpoint analysis for each experiment and model.
    
    Parameters:
        df (pd.DataFrame): 'aggregated_dataset.csv'.
        path (str): location to store breakpoint analysis files in.
    
    Returns:
        results_df (pd.DataFrame): 'breakpoint_quantification.csv' dataset containing breakpoint analysis results for all 
                                    experiments and models.

    """

    results = []


    # Step 1: Group df by experiment and generator
    grouped = df.groupby(['experiment_id', 'generator_name'])

    # Step 2: iterate over experiments
    for (exp_id, gen_name), group in grouped:
        x = group[group['param_stat_of'] == 'trajectory']['param_stat_value'].values

        # Step 2.1: Skip if x array is len(0)
        if len(x) == 0:
            continue

        # Step 2.2: Iterate over models for each experiment
        for model_name in df['param_stat_of'].unique().tolist():

            # Ignore trajectory as it is not a model
            if model_name == 'trajectory':
                continue

            y = group[group['param_stat_of'] == model_name]['param_stat_value'].values

            # Skip if x and y do not have same length. 
            if len(x) != len(y):
                continue

            # Step 2.3: Create path for breakpoint analysis
            analysis_path = os.path.join(path, str(exp_id), str(model_name))

            # Ensure directory exists before proceeding
            os.makedirs(analysis_path, exist_ok=True)

            # Step 2.4: Call breakpoint_analysis()
            breakpoint_results = breakpoint_analysis(x, y, analysis_path)

            # Step 2.5: Create row for dataframe
            davies = breakpoint_results["piecewise_results"]["davies"]
            converged = breakpoint_results["piecewise_results"]["converged"]

            if davies < 0.05 and converged == True and breakpoint_results["wald_results"] != None:

                breakpoint = breakpoint_results["piecewise_results"]["estimates"]["breakpoint1"]["estimate"]
                break_ci = breakpoint_results["piecewise_results"]["estimates"]["breakpoint1"]["confidence_interval"]
                wald_p = breakpoint_results["wald_results"]["pvalue"]
                comment = breakpoint_results["comment"]

            elif davies < 0.05 and converged == True and breakpoint_results["wald_results"] == None:

                breakpoint = breakpoint_results["piecewise_results"]["estimates"]["breakpoint1"]["estimate"]
                break_ci = breakpoint_results["piecewise_results"]["estimates"]["breakpoint1"]["confidence_interval"]
                wald_p = None
                comment = breakpoint_results["comment"]

            else:
                breakpoint = None
                break_ci = None
                wald_p = None
                comment = breakpoint_results["comment"]

            results.append({
                'Experiment': exp_id,
                'Generator': gen_name,
                'Model': model_name,
                'Davies p-value': davies,
                'Convergence': converged,
                'Breakpoint': breakpoint,
                'Breakpoint CI': break_ci,
                'Wald p-value': wald_p,
                'Comment': comment
                
            })


    summary_df = pd.DataFrame(results)
    return summary_df