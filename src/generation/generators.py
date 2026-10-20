# generators.py
# Minimal trajectory generators with *separated* noise trajectories.
# - Each generator returns a "signal" and a "noise" component (both length T),
#   plus "x" which is signal + noise (for convenience).
# - Noise is generated independently so you can later do counterfactual swaps
#   (e.g., same signal, different noise; same noise, different signal).

from __future__ import annotations

from typing import Any, Dict
import numpy as np
import pandas as pd
from fbm import FBM
from pathlib import Path


Array = np.ndarray


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def gaussian_noise(T: int, sigma: float, seed: int) -> Array:
    """
    Independent Gaussian noise trajectory eps_t ~ N(0, sigma^2).
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma < 0:
        raise ValueError("sigma must be >= 0.")
    r = _rng(seed)
    return r.normal(loc=0.0, scale=float(sigma), size=T)


def random_walk_with_drift(
    T: int,
    *,
    x0: float = 0.0,
    mu: float = 0.0,
    sigma: float = 1.0,
    seed_noise: int = 1,
    return_x: bool = True,
) -> Dict[str, Any]:
    """
    Random walk with drift (signal) + independent noise:
        signal: s_{t+1} = s_t + mu
        noise:  eps_t ~ N(0, sigma^2)
        x_t = s_t + eps_t

    Note: This is a drift-only signal; all randomness is in noise so you can
    counterfactually swap noise trajectories later.
    """
    if T <= 0:
        raise ValueError("T must be positive.")

    # deterministic signal given parameters
    t = np.arange(T, dtype=float)
    signal = float(x0) + float(mu) * t

    noise = gaussian_noise(T=T, sigma=sigma, seed=seed_noise)

    out = {
        "name": "rw_drift",
        "T": int(T),
        "params": {"x0": float(x0), "mu": float(mu), "sigma": float(sigma)},
        "seeds": { "noise": int(seed_noise)},
        "signal": signal.astype(float),
        "noise": noise.astype(float),
    }
    if return_x:
        out["x"] = (out["signal"] + out["noise"]).astype(float)
    return out


def ar1_with_noise(
    T: int,
    *,
    x0: float = 0.0,
    beta: float = 0.9,
    c: float = 0.0,
    sigma: float = 1.0,
    seed_noise: int = 1,
    return_x: bool = True,
) -> Dict[str, Any]:
    """
    AR(1) (deterministic recursion) + independent noise:
        noise:  eps_t ~ N(0, sigma^2)
        x_t = c + beta * x_t + eps_t

    Again: all randomness is in the separate noise trajectory.
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    if not (-1.0 <= beta <= 1.0):
        # you may allow outside [-1,1], but it's usually unstable
        raise ValueError("beta should be within [-1, 1] for a stable AR(1).")
    if sigma < 0:
        raise ValueError("sigma must be >= 0.")

    

    noise = gaussian_noise(T=T, sigma=sigma, seed=seed_noise)

    out = {
        "name": "ar1",
        "T": int(T),
        "params": {"x0": float(x0), "beta": float(beta), "c": float(c), "sigma": float(sigma)},
        "seeds": {"noise": int(seed_noise)},
        "noise": noise,
    }
    if return_x:
        x = np.empty(T, dtype=float)
        x[0] = float(x0)
        for t in range(T - 1):
            x[t + 1] = float(c) + float(beta) * x[t] + noise[t]
        out["x"] = x.astype(float)
    return out


def harmonic_oscillator_with_noise(
    T: int,
    *,
    A: float = 1.0,
    wavelength: float =  50.0,
    phase: float = 0.0,
    offset: float = 0.0,
    sigma: float = 0.1,
    seed_noise: int = 1,
    return_x: bool = True,
) -> Dict[str, Any]:
    """
    Harmonic oscillator (sinusoid) + independent noise:
        signal: s_t = offset + A * sin(omega * t + phase)
        noise:  eps_t ~ N(0, sigma^2)
        x_t = s_t + eps_t
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma < 0:
        raise ValueError("sigma must be >= 0.")
    if A < 0:
        raise ValueError("A should be >= 0 (use phase shift if you want sign flips).")

    t = np.arange(T, dtype=float)
    wavenumber = 2.0 * np.pi / wavelength
    signal = float(offset) + float(A) * np.sin(float(wavenumber) * t + float(phase))

    noise = gaussian_noise(T=T, sigma=sigma, seed=seed_noise)

    out = {
        "name": "harmonic",
        "T": int(T),
        "params": {
            "A": float(A),
            "wavelength": float(wavelength),
            "phase": float(phase),
            "offset": float(offset),
            "sigma": float(sigma),
        },
        "seeds": {"noise": int(seed_noise)},
        "signal": signal.astype(float),
        "noise": noise.astype(float),
    }
    if return_x:
        out["x"] = (out["signal"] + out["noise"]).astype(float)
    return out



def regime_switch_with_noise(
    T: int,
    *,
    x0: float = 0.0,
    dwell_time: int = 10,
    slopes: list[float] = [1.0, -1.0],
    sigma: float = 1.0, 
    seed_noise: int = 1,
    return_x: bool = True,
) -> Dict[str, Any]:
    """
    Regime-switching trend (signal) + independent noise:
        signal: s_t = s_{t-1} + slope_t  (cycling through slopes every dwell_time)
        noise:  eps_t ~ N(0, sigma^2)
        x_t = s_t + eps_t

    Note: All randomness is kept in the noise trajectory to allow counterfactual swaps.
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    if sigma < 0:
        raise ValueError("sigma must be >= 0.")
    if len(slopes) == 0:
        raise ValueError("slopes list cannot be empty.")

    # 1. Generate the deterministic signal
    indices = np.arange(T)
    num_regimes = len(slopes)
    regimes = (indices // dwell_time) % num_regimes
    
    slopes_arr = np.array(slopes, dtype=float)
    slope_array = slopes_arr[regimes]
    
    # Cumulative sum of slopes starting from x0
    # Note: we shift the cumsum so s_0 starts near x0 (or incorporates the first slope)
    signal = float(x0) + np.cumsum(slope_array)

    # 2. Generate the noise trajectory
    noise = gaussian_noise(T=T, sigma=sigma, seed=seed_noise)

    # 3. Build identical output dictionary structure
    out = {
        "name": "regime_switch",
        "T": int(T),
        "params": {
            "dwell_time": int(dwell_time),
            "slopes": [float(s) for s in slopes],
            "sigma": float(sigma),
        },
        "seeds": {"noise": int(seed_noise)},
        "signal": signal.astype(float),
        "noise": noise.astype(float),
    }

    if return_x:
        out["x"] = (out["signal"] + out["noise"]).astype(float)

    return out

def energy_release_with_noise(
    T: int,
    *,
    threshold: float = 10.0,
    mu: float = 0.2,
    sigma: float = 0.05,
    seed_noise: int = 1,
    return_x: bool = True,
) -> Dict[str, Any]:
    """
    Integrate-and-Fire / Energy-Release trajectory:
        signal: s_t = s_{t-1} + mu, resetting to 0.0 when s_t >= threshold
        noise:  eps_t ~ N(0, sigma^2)
        x_t:    x_t = x_{t-1} + mu + |eps_t|, resetting to 0.0 when x_t >= threshold

    Note: The absolute value of the noise is used to ensure strictly non-negative 
    tension increments, matching the build-up process.
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    if threshold <= 0:
        raise ValueError("threshold must be > 0.")
    if mu <= 0:
        raise ValueError("mu must be > 0 to guarantee accumulation and resets.")
    if sigma < 0:
        raise ValueError("sigma must be >= 0.")

    # 1. Generate the deterministic signal (pure build-up and reset)
    signal = np.zeros(T, dtype=float)
    current_signal_energy = 0.0
    for t in range(T):
        current_signal_energy += float(mu)
        if current_signal_energy >= float(threshold):
            signal[t] = current_signal_energy
            current_signal_energy = 0.0
        else:
            signal[t] = current_signal_energy

    # 2. Generate the independent noise trajectory
    noise = gaussian_noise(T=T, sigma=sigma, seed=seed_noise)

    # 3. Build identical output dictionary structure
    out = {
        "name": "energy_release",
        "T": int(T),
        "params": {
            "threshold": float(threshold),
            "mu": float(mu),
            "sigma": float(sigma),
        },
        "seeds": {"noise": int(seed_noise)},
        "signal": signal,
        "noise": noise,
    }

    # 4. Generate the combined noisy trajectory (x) and record event indices
    if return_x:
        x = np.zeros(T, dtype=float)
        events = []
        current_energy = 0.0
        
        for t in range(T):
            # Stochastic accumulation using absolute noise to guarantee non-negative increments
            increment = float(mu) + np.abs(noise[t])
            current_energy += increment
            
            if current_energy >= float(threshold):
                x[t] = current_energy
                events.append(t)
                current_energy = 0.0
            else:
                x[t] = current_energy
                
        out["x"] = x
        out["events"] = np.array(events, dtype=int)

    return out

def fractal_brownian_motion(
    T: int,
    *,
    hurst: float = 0.5,
    seed_noise: int = 1,
    return_x: bool = True,
) -> Dict[str, Any]:
    """
    Generates a pure Fractional Brownian Motion (fBm) trajectory:
        signal: s_t = 0.0 (No trend/drift)
        noise:  eps_t ~ fBm(Hurst=hurst, scale=sigma)
        x_t:    x_t = fBm_t (Identical to noise)
    """
    if T <= 0:
        raise ValueError("T must be positive.")
    if not (0.0 < hurst < 1.0):
        raise ValueError("hurst exponent must be in the open interval (0, 1).")


    # 1. No drift means signal is flat zero
    signal = np.zeros(T, dtype=float)

    # 2. Seed and generate the exact fBm path directly
    if seed_noise is not None:
        np.random.seed(seed_noise)

    # fbm(n=T-1) returns an array of length T (including the starting 0)
    noise = FBM(n=T-1, hurst=hurst, length=1, method='daviesharte').fbm()

    # 3. Build the standardized output dictionary
    out = {
        "name": "pure_fbm",
        "T": int(T),
        "params": {
            "hurst": float(hurst)
        },
        "seeds": {"noise": int(seed_noise) if seed_noise is not None else None},
        "signal": signal,
        "noise": noise,
    }

    # 4. Since there is no drift, x is just a copy of the direct fBm path
    if return_x:
        out["x"] = np.copy(noise)
        out["events"] = np.array([], dtype=int)

    return out


def unisolar_generator(
    T: int = 200, # dummy that is not used but required for compatibility with the generator registry
    slice_length: int = 200,
    *,
    seed_noise: int = 1, # dummy that is not used but required for compatibility with the generator registry
    return_x: bool = True 
) -> Dict[str, Any]:
    """
    Generates a trajecories from the Unisolar dataset by loading the dataset,
    Selecting 'SolarGeneration' column and returning 'slice_length' number of values from
    randomly selected starting point anchored by 'seed_noise'.
    """

    # Check for valid T and slice_length

    if T <= 0:
            raise ValueError("T must be positive.")
    if slice_length <= 0:
        raise ValueError("slice_length must be positive.")

    # Load the Unisolar dataset
    try:
        # Get the project root dynamically based on the current file's location
        ROOT_DIR = Path(__file__).resolve().parents[2]  # Adjust parent levels if needed
        DATA_PATH = ROOT_DIR / "case_studies" / "unisolar" / "Solar_Energy_Generation.csv"
        df = pd.read_csv(DATA_PATH)

    except FileNotFoundError:
        raise FileNotFoundError(f"Unisolar dataset not found at {DATA_PATH}. Please ensure the dataset is available.")

    # preprocess the data to select the 'SolarGeneration' column
    df_site = df[(df['CampusKey'] == 2) & (df['SiteKey'] == 1)]
    df_site['SolarGeneration'] = df_site['SolarGeneration'].fillna(0)
    data = df_site['SolarGeneration'].values

    # Use seed_noise to determine start_index independently of slice_length
    # We use a fixed maximum reference horizon (e.g., maximum expected slice or dataset bounds)
    # Here we seed the RNG with seed_noise directly:
    rng = np.random.default_rng(seed_noise)
    
    # Define a safe maximum possible start index based on a reference max slice (e.g., 1000 or len(data))
    # To allow maximum flexibility, we draw from the full valid index range of the dataset 
    # minus a small buffer, or clamp it against the requested slice_length afterwards.
    max_possible_start = len(data) - slice_length
    if max_possible_start <= 0:
        raise ValueError("slice_length is too large for the available data.")

    # To keep start_index invariant of slice_length, we can derive it from a fixed upper bound 
    # or use the generator's state consistently. Alternatively, drawing from a fixed dataset length:
    absolute_max_start = len(data) - 200  # assuming 200 is your base/minimum reference slice length
    if absolute_max_start <= 0:
        absolute_max_start = max_possible_start

    start_index = rng.integers(0, absolute_max_start)

    # Validate that the fixed start_index + current slice_length does not exceed data bounds
    if start_index + slice_length > len(data):
        raise ValueError(f"seed_noise {seed_noise} yields start_index {start_index}, which exceeds dataset bounds for slice_length {slice_length}.")

    slice_data = data[start_index:start_index + slice_length]

    # Build the output dictionary
    out = {
        "name": "unisolar",
        "T": int(slice_length), # we replace T with slice_length to allow for sweep configuration in experiment_config.yaml
        "params": {
            "slice_length": int(slice_length)
        },
        "seeds": {"start_index": int(start_index)},
        "signal": slice_data.astype(float),
        "noise": np.zeros(slice_length, dtype=float),  # No noise added
    }

    if return_x:
        out["x"] = slice_data.astype(float)  # x is just the signal in this case

    return out



def swat_generator(
    T: int = 200, # dummy that is not used but required for compatibility with the generator registry
    slice_length: int = 200,
    *,
    seed_noise: int = 1, # dummy that is not used but required for compatibility with the generator registry
    return_x: bool = True 
) -> Dict[str, Any]:
    """
    Generates a trajecories from the SWAT dataset by loading the dataset,
    Selecting 'P101' column and returning 'slice_length' number of values from
    randomly selected starting point.
    """

    # Check for valid T and slice_length

    if T <= 0:
            raise ValueError("T must be positive.")
    if slice_length <= 0:
        raise ValueError("slice_length must be positive.")

    # Load the SWAT dataset
    try:
        # Get the project root dynamically based on the current file's location
        ROOT_DIR = Path(__file__).resolve().parents[2]  # Adjust parent levels if needed
        DATA_PATH = ROOT_DIR / "case_studies" / "swat" / "normal.csv"
        df = pd.read_csv(DATA_PATH)

    except FileNotFoundError:
        raise FileNotFoundError(f"SWAT dataset not found at {DATA_PATH}. Please ensure the dataset is available.")

    # preprocess the data to select the 'P101' column
    data = df['P101']


    # downsample to every 10 value
    data = data[::10].reset_index(drop=True)

    # extrat the values as a numpy array
    data = data.values

    # Use seed_noise to determine start_index independently of slice_length
    # We use a fixed maximum reference horizon (e.g., maximum expected slice or dataset bounds)
    # Here we seed the RNG with seed_noise directly:
    rng = np.random.default_rng(seed_noise)
    
    # Define a safe maximum possible start index based on a reference max slice (e.g., 1000 or len(data))
    # To allow maximum flexibility, we draw from the full valid index range of the dataset 
    # minus a small buffer, or clamp it against the requested slice_length afterwards.
    max_possible_start = len(data) - slice_length
    if max_possible_start <= 0:
        raise ValueError("slice_length is too large for the available data.")

    # To keep start_index invariant of slice_length, we can derive it from a fixed upper bound 
    # or use the generator's state consistently. Alternatively, drawing from a fixed dataset length:
    absolute_max_start = len(data) - 200  # assuming 200 is your base/minimum reference slice length
    if absolute_max_start <= 0:
        absolute_max_start = max_possible_start

    start_index = rng.integers(0, absolute_max_start)

    # Validate that the fixed start_index + current slice_length does not exceed data bounds
    if start_index + slice_length > len(data):
        raise ValueError(f"seed_noise {seed_noise} yields start_index {start_index}, which exceeds dataset bounds for slice_length {slice_length}.")

    slice_data = data[start_index:start_index + slice_length]

    # Build the output dictionary
    out = {
        "name": "swat",
        "T": int(slice_length), # we replace T with slice_length to allow for sweep configuration in experiment_config.yaml
        "params": {
            "slice_length": int(slice_length)
        },
        "seeds": {"start_index": int(start_index)},
        "signal": slice_data.astype(float),
        "noise": np.zeros(slice_length, dtype=float),  # No noise added
    }

    if return_x:
        out["x"] = slice_data.astype(float)  # x is just the signal in this case

    return out


GENERATOR_REGISTRY = {
    "rw_drift": random_walk_with_drift,
    "ar1": ar1_with_noise,
    "harmonic": harmonic_oscillator_with_noise,
    "regime": regime_switch_with_noise,
    "energy": energy_release_with_noise,
    "fbm": fractal_brownian_motion,
    "unisolar": unisolar_generator,
    "swat": swat_generator,  # Assuming the same function is used for SWAT dataset
}





# Optional: counterfactual utility helpers
def swap_noise(traj: Dict[str, Any], new_noise: Array) -> Dict[str, Any]:
    """
    Return a shallow-copied trajectory dict where 'noise' is replaced and 'x' recomputed.
    """
    if "signal" not in traj:
        raise ValueError("traj must contain 'signal'.")
    if len(new_noise) != len(traj["signal"]):
        raise ValueError("new_noise must have same length as traj['signal'].")

    out = dict(traj)
    out["noise"] = np.asarray(new_noise, dtype=float)
    out["x"] = (np.asarray(out["signal"], dtype=float) + out["noise"]).astype(float)
    return out
