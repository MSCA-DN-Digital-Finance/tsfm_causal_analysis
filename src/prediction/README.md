# Prediction Step

This directory handles the prediction step for each Time Series Foundation Model.

## Module Dependency

The plot below shows the dependencies between the modules in this directory.

```mermaid
graph LR
    sample_builder[sample_builder.py] --> pred_runner[pred_runner.py]
    inference_pipeline[inference_pipeline.py] --> pred_runner
    model_factory[model_factory.py] -->  pred_runner
    utils[utils.py] --> pred_runner
    pred_runner --> create_preds[create_preds_modelname.py]
```

## Data Flow

The plot below shows the flow of data between files in this directory. Note that `experiment_config.yaml` is located in the root directory.

```mermaid
graph LR
    exp_conf[experiment_config.yaml] --> create_preds[create_preds_modelname.py]
    out_gen[artifacts/trajectories/...] --> create_preds
    create_preds --> out_pred[artifacts/prediction/...]
```