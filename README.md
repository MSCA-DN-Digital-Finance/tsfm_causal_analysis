
[![arXiv](https://img.shields.io/badge/arXiv-2608.24303-b31b1b.svg)](https://arxiv.org/abs/2608.24303)



# Experimental Codebase – Workflow Overview

This repository implements experiments for causal analysis of time-series foundation models under controlled generator interventions.

The pipeline is **hash-addressed and modular**, with a strict separation between:
- generation
- prediction
- analysis

Each stage can be re-run independently without recomputing previous stages.

## Repository Structure

The codebase maintains a strict one-to-one mapping between implementation and test coverage.For each component in `src/`, its corresponding test will be found at the exact same relative path in `tests/`.

```bash
├── src/                    # Source code
│   ├── generation/         # Counterfactual trajectory generation
│   ├── prediction/         # TSFMs inference logic
│   └── analysis/           # Visual and quantitative analysis
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

From your VS Code PowerShell terminal, run the automation script. This will handle remaining environment setups (for TSFM inference) and execute the pipeline:

```bash
#First-time setup and run:
./run_all.ps1 -Setup

#Subsequent runs:
./run_all.ps1
```

## Citation

If you use this code or research in your work, please cite:

Jander, M., van Heeswijk, W., & Mes, M. (2026). *Causal Analysis for Time Series Foundation Models*. arXiv preprint [arXiv:2608.24303](https://arxiv.org/abs/2608.24303).

```bibtex
@misc{jander2026causalanalysistimeseries,
      title={Causal Analysis for Time Series Foundation Models}, 
      author={Mathis Jander and Wouter van Heeswijk and Martijn Mes},
      year={2026},
      eprint={2608.24303},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={[https://arxiv.org/abs/2608.24303](https://arxiv.org/abs/2608.24303)}, 
}
