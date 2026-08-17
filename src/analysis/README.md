# Analysis Step

This directory handles the data analysis step.

## Module Dependency

The plot below shows the dependencies between the modules in this directory.

```mermaid
graph LR
    param_stats[param_stats.py] --> metric_runner[metric_runner.py]
    utils[utils.py] --> metric_runner
    metric_runner --> create_metrics[create_metrics.py]
```

## Data Flow

The plot below shows the flow of data between files in this directory. Note that `experiment_config.yaml` is located in the root directory.

```mermaid
graph LR
    exp_conf[experiment_config.yaml] --> create_metrics[create_metrics.py]
    preds[artifacts/prediction/...] --> create_metrics
    create_metrics --> out_metric[artifacts/analysis/...]
    out_metric --> create_dataset[create_dataset.py]
    create_dataset --> out_dataset[artifacts/dataset/...]
    out_dataset --> create_viz[create_viz.py]
    create_viz --> out_viz[artifacts/plots/...]
```