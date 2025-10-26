import os
import numpy as np
from scipy.io import loadmat

# ========================
# Internal helpers
# ========================

def _trim_segments_to_min_len(signal_list):
    """
    Takes a list of 1D signal arrays and trims them all to the shortest one.
    Returns a 2D array with shape (num_signals, min_length).
    """
    valid_signals = []

    for seg in signal_list:
        if seg is None:
            continue
        s = np.asarray(seg, dtype=float).ravel()
        if s.size > 0:
            valid_signals.append(s)

    if len(valid_signals) == 0:
        raise ValueError("Couldn't find any usable signals. All empty or None.")

    # Get the shortest segment length
    min_len = min(len(s) for s in valid_signals)

    # Stack into a 2D array (all signals cut to same length)
    return np.vstack([s[:min_len] for s in valid_signals])


def _normalize_shape(input_data):
    """
    Try to coerce various MATLAB structures into a uniform 2D shape:
    (n_segments, signal_length)

    Handles:
    - list of arrays
    - object arrays (MATLAB cell arrays)
    - single vectors
    - 2D arrays (including transposing if needed)
    """
    if isinstance(input_data, (list, tuple)):
        return _trim_segments_to_min_len(input_data)

    a = np.asarray(input_data)

    if a.dtype == object:
        # MATLAB cell arrays show up as object arrays — flatten each entry
        flattened = [np.asarray(x).ravel() for x in a.tolist()]
        return _trim_segments_to_min_len(flattened)

    if a.ndim == 1:
        # Just one signal? Wrap it into a 2D shape
        return a[None, :]

    if a.ndim == 2:
        # Transpose if needed (some data comes as shape (length, n))
        if a.shape[1] < a.shape[0]:
            return a.T

    return a  # already in good shape


# ========================
# Public API functions
# ========================

def load_pulsedb_mat(mat_path: str, signal_type: str = "PPG", limit: int = 1000):
    """
    Load a PulseDB subset saved as a .mat file.

    Arguments:
        mat_path : str
            Full path to .mat file from VitalDB (user must provide)
        signal_type : str
            One of "PPG", "ABP", "ECG", etc.
        limit : int
            Max number of signals to load

    Returns:
        ids : list of segment names like ["seg_0000", "seg_0001", ...]
        X   : 2D np.ndarray with shape (n_segments, segment_length)
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Hmm... couldn't find the file at: {mat_path}")

    mat_data = loadmat(mat_path, squeeze_me=True, simplify_cells=True)

    # Attempt to grab the signal array from various possible key formats
    key_variants = [signal_type, signal_type.upper(), signal_type.lower()]
    data_array = None

    for key in key_variants:
        if key in mat_data:
            data_array = mat_data[key]
            break

    # Still not found? Dive into nested dicts
    if data_array is None:
        for val in mat_data.values():
            if isinstance(val, dict):
                for key in key_variants:
                    if key in val:
                        data_array = val[key]
                        break
            if data_array is not None:
                break

    if data_array is None:
        top_keys = list(mat_data.keys())[:10]
        raise ValueError(
            f"Could not locate '{signal_type}' in the .mat file.\n"
            f"Top-level keys were: {top_keys}"
        )

    # Coerce to 2D and ensure numeric type
    X = _normalize_shape(data_array).astype(float)

    # Chop off extra entries if needed
    if X.shape[0] > limit:
        X = X[:limit]

    # Force all segments to the same length (again) just to be extra safe
    seg_len = min(len(row) for row in X)
    X = X[:, :seg_len]

    # Create segment names (a bit of flavor added)
    ids = [f"seg_{idx:04d}" for idx in range(X.shape[0])]
    return ids, X


def make_demo(n: int = 50, length: int = 500, seed: int = 42):
    """
    Generates fake data for testing or visualization purposes.

    Returns:
        (ids, X) where:
            ids is a list of string IDs
            X is a 2D np.ndarray (n, length)
    """
    rng = np.random.default_rng(seed)
    time_axis = np.linspace(0, 2 * np.pi, length)
    ids = []
    X = []

    for i in range(n):
        if i % 2 == 0:
            # Sine wave with slight jitter
            base_freq = 1.0 + 0.03 * rng.standard_normal()
            noise = 0.10 * rng.standard_normal(length)
            sig = np.sin(time_axis * base_freq) + noise
        else:
            # Random sparse spikes smoothed a bit
            impulses = (rng.random(length) > 0.97).astype(float)
            smooth = np.convolve(impulses, np.ones(5)/5.0, mode="same")
            noise = 0.05 * rng.standard_normal(length)
            sig = smooth + noise

        ids.append(f"demo_{i:04d}")
        X.append(sig.astype(float))

    return ids, np.vstack(X)
