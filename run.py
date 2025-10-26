# run.py
# --------------------------------------------------
# Quick launcher for CMPSC-463 clustering project
# --------------------------------------------------
# USAGE:
#   Demo mode (fake data, no .mat needed):
#       python run.py
#       python run.py --demo
#
#   With real dataset (.mat format):
#       python run.py --mat path/to/file.mat
#       # optionally: --signal ECG|PPG|ABP to override key detection
#
# OUTPUTS:
#   out/summary.csv
#   out/cluster_examples.png
#   out/tree.json
# --------------------------------------------------

import argparse
import os
import json
import sys

# -- quick and dirty import checker, no fancy installers here
def check_requirements(mod, pip_name=None):
    try:
        __import__(mod)
    except Exception as err:
        print(f"\n[ERROR] Missing required package: {pip_name or mod}")
        print(f"Install it manually with: pip install {pip_name or mod}")
        print(f"(Details: {err})\n")
        sys.exit(1)

check_requirements("numpy")
check_requirements("scipy")
check_requirements("matplotlib")

# -- imports after confirming deps
import numpy as np
from scipy.io import loadmat

# -- project-specific bits
from src.data_io import load_pulsedb_mat, make_demo
from src.clustering import DivideAndConquerClustering
from src.algorithms import closest_pair_bruteforce, kadane
from src.reporting import save_summary, plot_clusters_quick

# -----------------------------------
# Helper to auto-guess signal type
# -----------------------------------
def guess_signal_key(matfile_path: str) -> str | None:
    """
    Try to infer which signal to use (ECG, PPG, etc.) from the .mat file.
    This tries both top-level keys and single-level nested dicts.
    """
    print(f"[guess] Looking inside {matfile_path} for signal data…")
    mat_data = loadmat(matfile_path, squeeze_me=True, simplify_cells=True)

    def usable_keys(d):
        return [k for k in d if not k.startswith("__")]

    priority_signals = ["PPG", "ECG", "ABP"]
    alt_names = [s.lower() for s in priority_signals]

    # Try top-level keys first
    for key in usable_keys(mat_data):
        if key in priority_signals or key.lower() in alt_names:
            print(f"[guess] Found signal at top level: '{key}'")
            return key

    # Try one level deeper
    for key in usable_keys(mat_data):
        val = mat_data[key]
        if isinstance(val, dict):
            for subkey in val:
                if subkey in priority_signals or subkey.lower() in alt_names:
                    print(f"[guess] Found nested signal under '{key}': '{subkey}'")
                    return subkey

    # Last resort: find something array-ish
    for key in usable_keys(mat_data):
        val = mat_data[key]
        try:
            arr = np.asarray(val)
            if arr.size > 0 and arr.ndim in (1, 2):  # keeping it simple
                print(f"[guess] Using fallback key: '{key}' (looks like data)")
                return key
        except Exception:
            continue

    print("[guess] No valid signal found.")
    return None

# quick mkdir helper
def ensure_out_folder(path: str) -> str:
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# -------------------------------------------------
# Main function — this is where everything kicks off
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run clustering + Kadane’s on input signals.")
    parser.add_argument("--mat", type=str, default="", help="Path to .mat file. If not set, demo is used.")
    parser.add_argument("--signal", type=str, default="AUTO", help="Signal key to use (ECG, PPG, etc.)")
    parser.add_argument("--n", type=int, default=200, help="Max number of segments to process.")
    parser.add_argument("--out", type=str, default="out", help="Output directory.")
    parser.add_argument("--min_size", type=int, default=25, help="Minimum cluster size for leaf nodes.")
    parser.add_argument("--max_depth", type=int, default=None, help="Optional max tree depth.")
    parser.add_argument("--demo", action="store_true", help="Force demo mode regardless of --mat.")

    args = parser.parse_args()

    out_path = ensure_out_folder(args.out)

    # ---- Load Data ----
    if args.demo or not args.mat:
        print("[data] Demo mode ON — generating synthetic signals")
        ids, X = make_demo(n=min(max(args.n, 1), 10000), length=500)
    else:
        signal_key = args.signal
        if signal_key.upper() == "AUTO":
            signal_key = guess_signal_key(args.mat) or "PPG"  # fallback to PPG
            print(f"[data] Signal auto-selected: {signal_key}")

        print(f"[data] Loading data from {args.mat} (signal={signal_key}) …")
        try:
            ids, X = load_pulsedb_mat(args.mat, signal_type=signal_key, limit=args.n)
        except Exception as err:
            print(f"[warn] Problem reading '{signal_key}' from .mat: {err}")
            print("[warn] Reverting to demo mode instead.")
            ids, X = make_demo(n=min(max(args.n, 1), 10000), length=500)

    # Enforce string IDs
    ids = [str(i) for i in ids]
    X = np.asarray(X, dtype=float)
    num_samples, signal_len = X.shape
    print(f"[data] Loaded {num_samples} samples, each of length {signal_len}")

    # ---- Run Clustering ----
    print(f"[cluster] Starting D&C clustering (min_size={args.min_size}) …")
    dac = DivideAndConquerClustering(min_size=args.min_size, max_depth=args.max_depth)
    cluster_tree = dac.fit(X)

    # Save the tree
    tree_file = os.path.join(out_path, "tree.json")
    try:
        with open(tree_file, "w") as f:
            json.dump(cluster_tree, f)
        print(f"[save] Cluster tree saved to: {tree_file}")
    except Exception as err:
        print(f"[warn] Could not write cluster tree: {err}")

    # ---- Closest Pair in Each Leaf ----
    closest_results = []
    for cl_id, idx_list in cluster_tree["leaves"].items():
        members = list(idx_list)
        if len(members) < 2:
            closest_results.append({
                "cluster_id": cl_id,
                "size": len(members),
                "closest_pair_ids": "",
                "min_distance": float("inf"),
            })
            continue

        dist, (a, b) = closest_pair_bruteforce(X[members])
        closest_results.append({
            "cluster_id": cl_id,
            "size": len(members),
            "closest_pair_ids": f"{ids[members[a]]},{ids[members[b]]}",
            "min_distance": float(dist),
        })

    # ---- Kadane’s Algorithm Segment ----
    kad_results = []
    for i in range(num_samples):
        start, end, max_sum = kadane(X[i])
        kad_results.append({
            "id": ids[i],
            "kadane_start": start,
            "kadane_end": end,
            "kadane_sum": max_sum,
        })

    # ---- Write Summary CSV ----
    csv_output = os.path.join(out_path, "summary.csv")
    try:
        save_summary(closest_results, kad_results, csv_output)
        print(f"[save] Summary written to: {csv_output}")
    except Exception as err:
        print(f"[warn] Could not write summary CSV: {err}")

    # ---- Plot Cluster Snapshots ----
    img_output = os.path.join(out_path, "cluster_examples.png")
    try:
        plot_clusters_quick(X, cluster_tree, img_output)
        print(f"[save] Cluster plot saved as: {img_output}")
    except Exception as err:
        print(f"[warn] Plotting failed: {err}")
        # print("[debug] Might be matplotlib backend issue?")

    print(f"\n[done] All outputs saved in: {out_path}")


if __name__ == "__main__":
    main()
