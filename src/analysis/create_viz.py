#!/usr/bin/env python3
from pathlib import Path
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # Registers 'science' and 'grid' styles

# ==============================================================================
# GEOMETRY & STYLING CONFIGURATION
# ==============================================================================
TEXTWIDTH_IN = 7.007  # Standard 178mm double-column width
ASPECT_RATIO = 0.45
SCALE = 1.0

FIGWIDTH = TEXTWIDTH_IN * SCALE
FIGHEIGHT = FIGWIDTH * ASPECT_RATIO
LINE_WIDTH = 0.5

# Setup paths
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

CSV_DATASET = Path(project_root) / "artifacts" / "dataset" / "aggregated_dataset.csv"
PLOT_OUTPUT_DIR = Path(project_root) / "artifacts" / "plots"

# Metadata string mappings using raw LaTeX formatting
LABEL_MAPPINGS = {
    "chronos": "Chronos-2",
    "timesfm": "TimesFM-2.5",
    "trajectory": "Trajectory",
    "estimated_mean": r"$\hat{\mu}$",
    "estimated_beta": r"$\hat{\beta}$",
    "estimated_wavelength": r"$\hat{\lambda}$",
    "estimated_dwell_time": r"$\hat{\tau}$",
    "estimated_threshold": r"$\hat{\kappa}$",
    "estimated_hurst_exponent": r"$\hat{H}$",
    "mu": r"$\mu$",
    "beta": r"$\beta$",
    "wavelength": r"$\lambda$",
    "dwell_time": r"$\tau$",
    "threshold": r"$\kappa$",
    "hurst": r"$H$",
}

COLOR_PALETTE = {
    "Trajectory": "#7FC97F",
    "TimesFM-2.5": "#FDC086",
    "Chronos-2": "#beaed4",
}


def apply_pgf_style():
    """Applies global style settings for vector export without requiring external TeX binaries."""
    # 1. Apply science style first (this internally sets text.usetex = True)
    plt.style.use(["science", "grid"])

    # 2. Explicitly override parameters AFTER applying style to block external pdflatex calls
    plt.rcParams["text.usetex"] = False
    plt.rcParams["pgf.rcfonts"] = True
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["axes.formatter.use_mathtext"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 9.0
    plt.rcParams["axes.titlesize"] = 10.0
    plt.rcParams["axes.labelsize"] = 9.5
    plt.rcParams["xtick.labelsize"] = 8.5
    plt.rcParams["ytick.labelsize"] = 8.5
    plt.rcParams["legend.fontsize"] = 8.0


def style_spines_and_ticks(ax, width=LINE_WIDTH):
    """Sets explicit line widths for axis borders and ticks."""
    for spine in ax.spines.values():
        spine.set_linewidth(width)
    ax.tick_params(width=width, direction="in")


def apply_legend_style(ax, loc="upper left"):
    """Formats explicit legend frame line width."""
    legend = ax.legend(loc=loc, fancybox=False, edgecolor="black")
    if legend:
        legend.get_frame().set_linewidth(LINE_WIDTH)
    return legend


# ==============================================================================
# PLOT GENERATORS
# ==============================================================================


def make_experiment_boxplots(csv_path: str, output_dir: str):
    """Plot 1: Boxplots of parameter statistics grouped by intervention."""
    apply_pgf_style()
    df = pd.read_csv(csv_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    unique_exps = sorted(df["experiment_id"].dropna().unique())
    for exp_id in unique_exps:
        df_exp = df[df["experiment_id"] == exp_id].copy()
        if df_exp.empty:
            continue

        df_exp["param_stat_of"] = df_exp["param_stat_of"].map(
            lambda x: LABEL_MAPPINGS.get(x, x)
        )
        df_exp = df_exp.sort_values(by="parameter_value")

        intervention_param = df_exp["intervention_param"].iloc[0]
        param_stat_name = df_exp["param_stat_name"].iloc[0]

        x_symbol = LABEL_MAPPINGS.get(intervention_param, intervention_param)
        y_symbol = LABEL_MAPPINGS.get(param_stat_name, param_stat_name)

        fig = plt.figure(figsize=(FIGWIDTH, FIGHEIGHT))
        ax = plt.gca()

        param_vals = sorted(df_exp["parameter_value"].unique())
        models = df_exp["param_stat_of"].unique()

        n_models = len(models)
        box_width = 0.8 / n_models
        x_indices = np.arange(len(param_vals))

        for i, model in enumerate(models):
            model_df = df_exp[df_exp["param_stat_of"] == model]
            data_per_val = [
                model_df[model_df["parameter_value"] == v][
                    "param_stat_value"
                ].dropna().values
                for v in param_vals
            ]

            offsets = x_indices + (i - (n_models - 1) / 2) * box_width
            color = COLOR_PALETTE.get(model, "#333333")

            ax.boxplot(
                data_per_val,
                positions=offsets,
                widths=box_width * 0.85,
                patch_artist=True,
                manage_ticks=False,
                boxprops=dict(facecolor=color, alpha=0.85, linewidth=LINE_WIDTH),
                medianprops=dict(color="black", linewidth=LINE_WIDTH),
                whiskerprops=dict(linewidth=LINE_WIDTH),
                capprops=dict(linewidth=LINE_WIDTH),
                flierprops=dict(markersize=2, markeredgewidth=LINE_WIDTH),
            )
            ax.plot([], [], color=color, label=model, linewidth=2.0)

        ax.set_xticks(x_indices)
        ax.set_xticklabels(param_vals)
        ax.set_xlabel(f"Intervention Parameter ({x_symbol})")
        ax.set_ylabel(f"Parameter Statistic ({y_symbol})")

        style_spines_and_ticks(ax)
        apply_legend_style(ax, loc="upper left")

        plt.tight_layout()
        save_file = out_path / f"experiment_{exp_id}_boxplot.pdf"
        plt.savefig(save_file)
        plt.close(fig)
        print(f" -> Saved: {save_file}")


def make_individual_scatter_plots(csv_path: str, output_dir: str):
    """Plot 2: Scatter plot matching models against ground truth."""
    apply_pgf_style()
    df = pd.read_csv(csv_path)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    unique_exps = sorted(df["experiment_id"].dropna().unique())
    for exp_id in unique_exps:
        df_exp = df[df["experiment_id"] == exp_id].copy()
        if df_exp.empty:
            continue

        param_stat_name = df_exp["param_stat_name"].iloc[0]
        y_symbol = LABEL_MAPPINGS.get(param_stat_name, param_stat_name)

        pivot_df = df_exp.pivot(
            index=["run_id", "parameter_value"],
            columns="param_stat_of",
            values="param_stat_value",
        ).reset_index()

        if "trajectory" not in pivot_df.columns:
            continue

        model_cols = [
            col
            for col in pivot_df.columns
            if col not in ["run_id", "parameter_value", "trajectory"]
        ]
        if not model_cols:
            continue

        # 1. Match figsize aspect ratio to the 1:1 plot aspect ratio to avoid white space & scaling blowups
        fig = plt.figure(figsize=(FIGWIDTH, FIGWIDTH * 0.75))
        ax = plt.gca()

        markers = ["+", "x", "d", ".", "*"]
        all_vals = [pivot_df["trajectory"].dropna()]

        for idx, col in enumerate(model_cols):
            model_label = LABEL_MAPPINGS.get(col, col)
            color = COLOR_PALETTE.get(
                model_label, COLOR_PALETTE.get(col, "#333333")
            )
            marker = markers[idx % len(markers)]

            x_vals = pivot_df["trajectory"]
            y_vals = pivot_df[col]
            all_vals.append(y_vals.dropna())

            ax.scatter(
                x_vals,
                y_vals,
                label=model_label,
                color=color,
                marker=marker,
                s=20,
                linewidths=LINE_WIDTH,
            )

        combined_vals = pd.concat(all_vals).dropna()
        if not combined_vals.empty:
            min_val, max_val = combined_vals.min(), combined_vals.max()
            padding = (
                (max_val - min_val) * 0.08 if max_val != min_val else 0.1
            )
            lims = [min_val - padding, max_val + padding]

            ax.plot(
                lims,
                lims,
                color="black",
                lw=LINE_WIDTH,
                linestyle="--",
                label=r"Ideal ($y=x$)",
            )
            ax.set_xlim(lims)
            ax.set_ylim(lims)

        ax.set_aspect("equal", adjustable="box")

        # 2. Apply spine styling FIRST so it doesn't override label properties
        style_spines_and_ticks(ax)

        # 3. Explicitly set smaller label fonts and tick sizes after styling
        ax.set_xlabel(f"Trajectory ParameterStatistic ({y_symbol})", fontsize=7.5, labelpad=4)
        ax.set_ylabel(f"Model Parameter Statistic ({y_symbol})", fontsize=7.5, labelpad=4)
        ax.tick_params(axis="both", labelsize=6.5)

        apply_legend_style(ax, loc="upper left")

        # 4. Use bbox_inches='tight' on export to crop remaining outer whitespace
        plt.tight_layout()
        save_file = out_path / f"experiment_{exp_id}_scatter_consistency.pdf"
        plt.savefig(save_file, bbox_inches="tight")
        plt.close(fig)
        print(f" -> Saved: {save_file}")

def make_experiment_trajectories(csv_path: str, output_dir: str):
    """Plot 3: Subplot grid showing ground truth vs predicted time-series trajectories (vertically stacked)."""
    apply_pgf_style()
    df = pd.read_csv(csv_path)
    artifacts_dir = Path(csv_path).parent.parent
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    unique_exps = sorted(df["experiment_id"].dropna().unique())
    for exp_id in unique_exps:
        df_exp = df[df["experiment_id"] == exp_id].copy()

        intervention_param = (
            df_exp["intervention_param"].iloc[0]
            if len(df_exp) > 0
            else "Parameter"
        )
        param_symbol = LABEL_MAPPINGS.get(
            intervention_param, intervention_param
        )
        param_values = sorted(df_exp["parameter_value"].unique())

        model_paths = {}
        gt_paths = {}

        for val in param_values:
            df_val = df_exp[df_exp["parameter_value"] == val]
            run_id = df_val["run_id"].iloc[0] if len(df_val) > 0 else None
            if not run_id:
                continue

            gt_file = (
                artifacts_dir
                / "generation"
                / str(run_id)
                / "trajectory.npz"
            )
            if gt_file.exists():
                gt_paths[val] = gt_file

            pred_base_dir = artifacts_dir / "prediction" / str(run_id)
            if pred_base_dir.exists():
                for model_dir in pred_base_dir.iterdir():
                    if model_dir.is_dir():
                        pred_file = model_dir / "predictions.npz"
                        if pred_file.exists():
                            model_name = model_dir.name
                            if model_name not in model_paths:
                                model_paths[model_name] = {}
                            model_paths[model_name][val] = pred_file

        discovered_models = sorted(list(model_paths.keys()))
        num_subplots = 1 + len(discovered_models)

        if not gt_paths and not model_paths:
            continue

        # Adjust vertical height dynamically based on the number of subplots
        row_height = FIGHEIGHT * 0.8
        fig, axes = plt.subplots(
            num_subplots,  # Vertical stack: nrows = num_subplots
            1,             # Single column: ncols = 1
            figsize=(FIGWIDTH, row_height * num_subplots),
            sharex=True,
            sharey=True,
        )
        if num_subplots == 1:
            axes = [axes]

        def load_npz_sequence(path):
            with np.load(path) as data:
                key = "x" if "x" in data.files else data.files[0]
                arr = data[key]
                return arr.flatten() if arr.ndim > 1 else arr

        # Plot Ground Truth Trajectory
        ax_gt = axes[0]
        for val, path in gt_paths.items():
            try:
                y_vals = load_npz_sequence(path)
                ax_gt.plot(y_vals, label=rf"{param_symbol} = {val}", lw=1.0)
            except Exception:
                pass

        ax_gt.set_title(LABEL_MAPPINGS.get("trajectory", "Trajectory"))
        ax_gt.set_ylabel("Value")
        style_spines_and_ticks(ax_gt)
        apply_legend_style(ax_gt, loc="upper right")

        # Plot Model Predictions
        for idx, model_name in enumerate(discovered_models, start=1):
            ax_m = axes[idx]
            model_configs = model_paths[model_name]

            for val, path in model_configs.items():
                try:
                    y_vals = load_npz_sequence(path)
                    ax_m.plot(
                        y_vals,
                        label=rf"{param_symbol} = {val}",
                        lw=1.0,
                        linestyle="--",
                    )
                except Exception:
                    pass

            mapped_title = LABEL_MAPPINGS.get(model_name, model_name)
            ax_m.set_title(mapped_title)
            ax_m.set_ylabel("Value")
            style_spines_and_ticks(ax_m)
            apply_legend_style(ax_m, loc="upper right")

        # Apply x-axis label only to the bottom-most subplot
        axes[-1].set_xlabel(r"Timestep ($t$)")

        plt.tight_layout()
        save_file = out_path / f"experiment_{exp_id}_trajectory_cascade.pdf"
        plt.savefig(save_file)
        plt.close(fig)
        print(f" -> Saved: {save_file}")
# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    make_experiment_boxplots(CSV_DATASET, PLOT_OUTPUT_DIR)
    make_individual_scatter_plots(CSV_DATASET, PLOT_OUTPUT_DIR)
    make_experiment_trajectories(CSV_DATASET, PLOT_OUTPUT_DIR)