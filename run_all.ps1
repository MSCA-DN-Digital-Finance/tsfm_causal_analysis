param(
    [switch]$Setup
)

$ErrorActionPreference = "Stop"

function Ensure-Env {
    param(
        [string]$EnvName,
        [string]$YamlPath
    )

    $envExists = conda env list | Select-String "^\s*$EnvName\s"
    if (-not $envExists) {
        Write-Host "Creating conda env '$EnvName' from $YamlPath..."
        conda env create -n $EnvName -f $YamlPath
    } else {
        Write-Host "Conda env '$EnvName' already exists."
    }
}

if ($Setup) {
    Ensure-Env "ct3-core"  "envs/env_ct3-core.yml"
    Ensure-Env "chronos"   "envs/env_chronos.yml"
    Ensure-Env "timesfm"   "envs/env_timesfm.yml"
}

function Run-Step {
    param(
        [string]$Message,
        [string]$Env,
        [string]$Script
    )
    Write-Host $Message
    conda run -n $Env --no-capture-output python $Script
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Message (env=$Env, script=$Script)"
    }
}

# Change these lines to target where the files actually live:
Run-Step "Creating trajectories..."       "ct3-core"  "src/generation/create_trajs.py"
Run-Step "Running TimesFM predictions..." "timesfm"   "src/prediction/create_preds_timesfm.py"
Run-Step "Running Chronos predictions..." "chronos"   "src/prediction/create_preds_chronos.py"
Run-Step "Computing CT3 metrics..."       "ct3-core"  "src/analysis/create_metrics.py" 
Run-Step "Aggregating data..."            "ct3-core"  "src/analysis/create_dataset.py"
Run-Step "Creating plots..."              "ct3-core"   "src/analysis/create_viz.py"


Write-Host "Done."
