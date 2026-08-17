from pathlib import Path
import json
import sys
from typing import Dict, Any
import numpy as np


root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from utils import stable_hash, ensure_dir


def save_run(run_dir: Path, traj: Dict[str, Any], config: Dict[str, Any]) -> None:
    ensure_dir(run_dir)

    # Save arrays in compressed npz
    arrays = {
        "noise": np.asarray(traj["noise"], dtype=float),
    }
    if "x" in traj:
        arrays["x"] = np.asarray(traj["x"], dtype=float)

    np.savez_compressed(run_dir / "trajectory.npz", **arrays)

    # Save metadata/config separately (JSON)
    meta = {
        "run_id": run_dir.name,
        "sweep_config": config,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def run_sweep(
        sweep_list: list,
        gen_registry: dict,
        out_root: Path
        ) -> None:
    
    """
    Function runs the sweeps defined by sweep list through generators and saves trajectories to disk.

    params:
    - sweep_list: list of sweeps to be run. 
    - gen_registry: dictionary mapping generator name to function
    
    return: nothing, saves files to disk.
    """

    print(f"Planned runs: {len(sweep_list)}")

    ensure_dir(out_root)


    for i, sweep in enumerate(sweep_list):

        gen_name = sweep["generator"]

        run_id = stable_hash(sweep)  # still hashes full config (incl. seed/task/experiment_id)
        run_dir = out_root / run_id

        if (run_dir / "trajectory.npz").exists() and (run_dir / "meta.json").exists():
            print("This sweep already exists...skipping...")
            continue

        # extract the generator
        
        gen_func = gen_registry[gen_name]

        params = sweep["params"]
        traj_dict = gen_func(**params)


        
        print(run_dir)
        save_run(run_dir=run_dir, traj=traj_dict, config=sweep)

        print(f"Sweep {i} out of {len(sweep_list)} completed.")

    print("Done.")




