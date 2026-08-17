# Generation Step

This directory handles the data generation step.

## Module Dependency

The plot below shows the dependencies between the modules in this directory.

```mermaid
graph LR
    utils[utils.py] --> build_sweep[sweep_builder.py]
    utils[utils.py] --> run_sweep[sweep_runner.py]
    generators[generators.py] --> create_traj[create_traj.py]
    build_sweep --> create_traj
    run_sweep --> create_traj
```

## Data Flow

The plot below shows the flow of data between files in this directory. Note that `experiment_config.yaml` is located in the root directory.

```mermaid
graph LR
    exp_conf[experiment_config.yaml] --> create_traj[create_traj.py]
    create_traj --> out_root[artifacts/trajectories/...]
```
