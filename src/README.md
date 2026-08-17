# Source Code

This directory and its subdirectories contain all the code required to execute the experimental pipeline. Each subdirectory contains a `README.md` explaining the module dependencies and data flows.

## Data Flow Across Steps

The plot below shows the data flow across the steps contained in the subdirectories. Note that `experiment_config.yaml` is located in the root directory.

```mermaid
graph TD
    exp_conf[experiment_config.yaml] --> create_traj[create_traj.py]
    create_traj --> out_gen[artifacts/trajectories/...]
    out_gen --> create_preds[create_preds_modelname.py]
    exp_conf --> create_preds[create_traj.py]
    create_preds --> out_pred[artifacts/prediction/...]
    out_pred --> create_metrics[create_metrics.py]
    create_metrics --> out_metric[artifacts/analysis/...]
    out_metric --> create_dataset[create_dataset.py]
    create_dataset --> out_dataset[artifacts/dataset/...]
    out_dataset --> create_viz[create_viz.py]
    create_viz --> out_viz[artifacts/plots/...]
```