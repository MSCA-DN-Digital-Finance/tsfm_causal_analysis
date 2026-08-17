import yaml
import numpy as np
import json
import hashlib
from typing import Dict
from pathlib import Path

def load_config(path="config.yaml"):
    """
    Reads a YAML file from disk and parses it into a Python dictionary.
    
    Args:
        path (str): The relative or absolute path to the .yaml file.
        
    Returns:
        dict: The configuration settings.
    """
    with open(path, "r") as f:
        # safe_load prevents the execution of arbitrary code in the YAML file
        return yaml.safe_load(f)
    


def stable_hash(obj: Dict) -> str:
    """
    Stable hash of JSON-serializable objects (dict/list/str/int/float/bool/None).
    """
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def get_exp_id_from_meta(run_dir: Path) -> str:
    """
    Extracts the experiment ID from a meta.json file located in the given run directory.

    Args:
        run_dir (Path): The directory containing the meta.json file.

    Returns:
        str: The experiment ID if found, otherwise None.
    """
    meta_json = run_dir / "meta.json"
    if meta_json.exists():
        try:
            meta = json.loads(meta_json.read_text())
            
            # 1. Try to get it from the nested 'config' block first
            # 2. Fall back to the top level (for backward compatibility/consistency)
            config_block = meta.get("sweep_config", {})
            exp_id = config_block.get("experiment_id") or meta.get("experiment_id")
            
            return exp_id
        except Exception as e:
            print(f"Error reading meta.json for {run_dir}: {e}")
            return None
    else:
        print(f"No meta.json found in {run_dir}")
        return None