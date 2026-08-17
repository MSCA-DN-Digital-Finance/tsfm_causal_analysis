# Experimental Codebase – Workflow Overview

This repository implements causal experiments for time-series foundation models under controlled generator interventions.

The pipeline is **hash-addressed, restartable, and modular**, with a strict separation between:
- generation
- prediction
- analysis

Each stage can be re-run independently without recomputing previous stages.

## Repository Structure

The codebase maintains a strict one-to-one mapping between implementation and test coverage.For each component in `src/`, its corresponding test will be found at the exact same relative path in `tests/`.

```bash
├── src/                    # Source code for the CT3 pipeline
│   ├── generation/         # Counterfactual trajectory generation
│   ├── prediction/         # Chronos-2 and TimesFM inference logic
│   └── analysis/           # CT3 implementation and 
│
└── tests/                  # Unit tests
    ├── generation/         # Tests for generator functions
    ├── prediction/         # Tests for model inference pipeline
    └── analysis/           # Tests for analysis metrics & outputs
```

## Getting Started

### 1. Environment Setup
First, ensure you have the core environment installed, then open the workspace:
```bash
# Install the core environment (adjust command if you use an environment.yml)
conda activate ct3-core

# Open the project in VS Code
code .
```

### 2. Initialization & Execution

From your VS Code PowerShell terminal, run the automation script. This will handle remaining environment setups (for Chronos and TimesFM inference) and execute the pipeline:

```bash
#First-time setup and run:
./run_all.ps1 -Setup

#Subsequent runs:
./run_all.ps1
```


