# run.py — script to run clustering on segments pulled from a MATLAB HDF5 file
# Note: targeting .mat v7.3 format (uses HDF5 backend)

import argparse
import os
import sys
import json
import csv
import random
import hashlib

# --- Quick dependency check (not ideal, but it'll do for now) ---
def ensure_module(mod_name, pip_hint=None):
    try:
        __import__(mod_name)
    except Exception as e:
        pkg_name = pip_hint or mod_name
        print(f"[missing] '{pkg_name}' not installed. You can try: pip install {pkg_name}")
        print(f"[err] {e}")
        sys.exit(1)

ensure_module("numpy")
ensure_module("matplotlib")
ensure_module("h5py")

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")  # don't need interactive plots
import matplotlib.pyplot as plt

# default data folder — should contain .mat file(s)
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")

# --- util bits ---
def md5_hash(path, block_size=2**20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def find_first_mat(data_dir=DATA_FOLDER):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"No folder found: {data_dir}")
    mats = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir) if fn.lower().endswith(".mat")]
    if not mats:
        raise FileNotFoundError(f"No .mat files found in {data_dir}")
    return os.path.abspath(sorted(mats)[0])

def try_stack(arrays):
    valid = []
    for arr in arrays:
        if arr is None:
            continue
        flat = np.asarray(arr, dtype=float).ravel()
        if flat.size > 0 and np.isfinite(flat).any():
            valid.append(flat)
    if not valid:
        raise ValueError("Empty or invalid segment list")
    min_len = min(len(a) for a in valid)
    return np.vstack([a[:min_len] for a in valid])

def force_2d(x):
    if isinstance(x, (list, tuple)):
        return try_stack(x)
    a = np.asarray(x, dtype=float)
    if a.dtype == object:
        try:
            return try_stack([np.asarray(t, dtype=float).ravel() for t in a.ravel().tolist()])
        except:
            raise ValueError("Couldn't parse object array into usable data")
    if a.ndim == 1:
        return a[None, :]
    if a.ndim == 2:
        return a
    raise ValueError("Unsupported shape for 2D coercion")

# --- HDF5/MAT v7.3 loading helpers ---
def list_h5_datasets(h5f):
    out = []
    def _gather(name, obj):
        if isinstance(obj, h5py.Dataset):
            out.append((name, obj))
    h5f.visititems(_gather)
    return out

def reshape_numeric_2d(arr):
    a = np.asarray(arr)
    if a.size == 0:
        raise ValueError("Empty array.")
    a = np.squeeze(a)
    if a.ndim == 1:
        return a[None, :]
    if a.ndim == 2:
        return a
    total = a.size
    longest = max(d for d in a.shape if d)
    if longest <= 1:
        raise ValueError(f"Array shape not usable: {a.shape}")
    rows = total // longest
    return a.reshape(rows, longest)

def extract_from_object_refs(f, obj_array):
    # pulls numeric datasets out of object refs
    extracted = []
    arr = obj_array[()]
    flat = arr.ravel()
    for item in flat:
        if isinstance(item, h5py.Reference) and item:
            tgt = f[item]
            if isinstance(tgt, h5py.Dataset):
                val = np.array(tgt[()])
                if val.size > 0:
                    extracted.append(np.squeeze(val))
        else:
            try:
                val = np.array(item)
                if np.issubdtype(val.dtype, np.number) and val.size > 0:
                    extracted.append(np.squeeze(val))
            except:
                pass
    if not extracted:
        raise ValueError("Nothing numeric found in object refs.")
    return try_stack(extracted)

def load_matlab_v73(mat_path, tag=None):
    def prod(shape):
        p = 1
        for d in shape:
            p *= int(d)
        return p

    with h5py.File(mat_path, "r") as f:
        datasets = list_h5_datasets(f)
        if not datasets:
            raise ValueError("No datasets found in file")

        numeric, objects = [], []
        for name, ds in datasets:
            shp = tuple(int(d) for d in ds.shape)
            if np.issubdtype(ds.dtype, np.number):
                numeric.append((name, ds, shp))
            elif ds.dtype.kind == "O":
                objects.append((name, ds, shp))

        def match_names(pool, keywords):
            return [(n, d, s) for n, d, s in pool if any(k in n.lower() for k in keywords)]

        candidates = []
        if tag:
            candidates += match_names(numeric, [tag])
            candidates += match_names(objects, [tag])
        candidates += match_names(numeric, ["ppg", "ecg", "abp"])
        candidates += match_names(objects, ["ppg", "ecg", "abp"])

        seen = set(n for n, _, _ in candidates)
        candidates += [(n, d, s) for n, d, s in numeric if n not in seen]
        seen |= set(n for n, _, _ in numeric)
        candidates += [(n, d, s) for n, d, s in objects if n not in seen]

        for name, ds, shape in candidates:
            try:
                if shape == () or prod(shape) == 0:
                    continue
                if np.issubdtype(ds.dtype, np.number):
                    arr = ds[()]
                    if arr.size == 0:
                        continue
                    X = reshape_numeric_2d(arr)
                else:
                    X = extract_from_object_refs(f, ds)

                X = np.asarray(X, dtype=float)
                if X.ndim != 2 or X.size == 0:
                    continue

                n, L = X.shape
                if n < 2 and L >= 2:
                    X = X.T
                    n, L = X.shape
                if n >= 2 and L >= 8:
                    print(f"[mat] dataset: {name}")
                    print(f"[mat] shape: {X.shape}")
                    return X, name
            except Exception as err:
                continue

        raise ValueError("Couldn't find usable 2D data in file.")

# --- signal processing utilities ---
def znorm(x):
    x = np.asarray(x, dtype=float)
    mean, std = x.mean(), x.std()
    return (x - mean) / (std + 1e-8) if std and np.isfinite(std) else (x - mean)

def corr_dist(x1, x2):
    x1 = znorm(x1)
    x2 = znorm(x2)
    dot = float(np.dot(x1, x2))
    denom = float(np.linalg.norm(x1) * np.linalg.norm(x2)) + 1e-8
    return 1.0 - (dot / denom)

def kadane_segment(arr):
    max_sum, current = -float("inf"), 0.0
    left = right = tmp = 0
    for i, val in enumerate(arr):
        val = float(val)
        if current <= 0:
            current, tmp = val, i
        else:
            current += val
        if current > max_sum:
            max_sum, left, right = current, tmp, i+1
    return left, right, max_sum

def find_closest_pair(X, idxs):
    idxs = np.asarray(idxs, dtype=int)
    if len(idxs) < 2:
        return (int(idxs[0]), int(idxs[0])), float("inf")
    best_pair = (None, None)
    best_dist = float("inf")
    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            d = corr_dist(X[idxs[i]], X[idxs[j]])
            if d < best_dist:
                best_dist = d
                best_pair = (int(idxs[i]), int(idxs[j]))
    return best_pair, best_dist

# --- main clustering class (DAC) ---
class DAC:
    def __init__(self, min_size=25, max_depth=None, k=16, seed=0):
        self.min_size = int(min_size)
        self.max_depth = max_depth
        self.k = k
        self.rng = random.Random(seed)
        self.leaves = {}

    def fit(self, X):
        idxs = np.arange(X.shape[0])
        root = self._recursive_split(X, idxs, depth=0, nid=0)
        return {"root": root, "leaves": self.leaves}

    def _choose_two(self, X, idxs):
        s0 = self.rng.choice(list(idxs))
        sample = self.rng.sample(list(idxs), min(self.k, len(idxs)))
        s1 = max(((corr_dist(X[s0], X[j]), j) for j in sample))[1]
        return s0, s1

    def _partition(self, X, idxs):
        if len(idxs) < 2:
            return idxs, np.array([], dtype=int)
        a, b = self._choose_two(X, idxs)
        left, right = [], []
        for j in idxs:
            da = corr_dist(X[j], X[a])
            db = corr_dist(X[j], X[b])
            (left if da <= db else right).append(j)
        return np.asarray(left), np.asarray(right)

    def _recursive_split(self, X, idxs, depth, nid):
        node = {"id": nid, "size": len(idxs), "children": []}
        if len(idxs) <= self.min_size or (self.max_depth is not None and depth >= self.max_depth):
            node["leaf"] = True
            self.leaves[nid] = {"idxs": np.array(idxs)}
            return node
        L, R = self._partition(X, idxs)
        if len(L) == 0 or len(R) == 0:
            node["leaf"] = True
            self.leaves[nid] = {"idxs": np.array(idxs)}
            return node
        node["leaf"] = False
        node["children"].append(self._recursive_split(X, L, depth+1, nid*2+1))
        node["children"].append(self._recursive_split(X, R, depth+1, nid*2+2))
        return node

# --- reporting ---
def write_results(leaf_info, kad_info, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# Cluster summary"])
        writer.writerow(["cluster_id", "size", "closest_pair_ids", "min_distance"])
        for row in leaf_info:
            writer.writerow([row["cluster_id"], row["size"], row["closest_pair_ids"], f"{row['min_distance']:.6f}"])
        writer.writerow([])
        writer.writerow(["# Kadane per segment"])
        writer.writerow(["id", "kadane_start", "kadane_end", "kadane_sum"])
        for row in kad_info:
            writer.writerow([row["id"], row["kadane_start"], row["kadane_end"], f"{row['kadane_sum']:.6f}"])

def plot_cluster_means(X, leaves, out_file, max_plots=6):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    top_clusters = sorted(leaves, key=lambda k: len(leaves[k]["idxs"]), reverse=True)[:max_plots]
    plt.figure(figsize=(10, 6))
    for cid in top_clusters:
        idxs = leaves[cid]["idxs"]
        avg = X[idxs].mean(axis=0)
        plt.plot(avg, label=f"cluster {cid} (n={len(idxs)})")
    plt.title("Mean signals by cluster")
    plt.xlabel("time")
    plt.ylabel("z-norm amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()

def json_friendly_tree(tree):
    import copy
    t = copy.deepcopy(tree)
    def clean(n):
        out = {k: int(n[k]) if k in ("id", "size") else n[k] for k in ("id", "size", "leaf") if k in n}
        out["children"] = [clean(c) for c in n.get("children", [])]
        return out
    result = {"root": clean(t["root"]), "leaves": {}}
    for k, v in t["leaves"].items():
        result["leaves"][str(k)] = {"idxs": [int(i) for i in np.asarray(v["idxs"]).ravel().tolist()]}
    return result

# --- main runner ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=str, default="", help="path to .mat file (or autodetect)")
    parser.add_argument("--signal", type=str, default="", help="optional signal tag to filter: PPG|ECG|ABP")
    parser.add_argument("--n", type=int, default=1000, help="how many segments to use")
    parser.add_argument("--out", type=str, default="out/", help="where to save outputs")
    parser.add_argument("--min_size", type=int, default=25, help="min samples per cluster leaf")
    parser.add_argument("--max_depth", type=int, default=None, help="optional max tree depth")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load dataset
    mat_path = args.mat.strip() or find_first_mat()
    print(f"[dataset] {mat_path}")
    print(f"[md5] {md5_hash(mat_path)}")

    Xraw, label = load_matlab_v73(mat_path, tag=args.signal.strip() or None)
    if Xraw.ndim != 2:
        raise ValueError("Expected 2D array.")

    if args.n and Xraw.shape[0] > args.n:
        Xraw = Xraw[:args.n]

    X = np.vstack([znorm(row) for row in Xraw])
    print(f"[data] final shape: {X.shape}")

    # Cluster
    clusterer = DAC(min_size=args.min_size, max_depth=args.max_depth)
    tree = clusterer.fit(X)
    print(f"[cluster] leaf count: {len(tree['leaves'])}")

    # Summaries
    leaf_summary = []
    for cid, meta in tree["leaves"].items():
        pair, dist = find_closest_pair(X, meta["idxs"])
        leaf_summary.append({
            "cluster_id": int(cid),
            "size": len(meta["idxs"]),
            "closest_pair_ids": f"{pair[0]}-{pair[1]}",
            "min_distance": dist
        })

    kad_summaries = []
    for gid in range(X.shape[0]):
        l, r, s = kadane_segment(X[gid])
        kad_summaries.append({"id": gid, "kadane_start": l, "kadane_end": r, "kadane_sum": s})

    # Output
    out_csv = os.path.join(args.out, "summary.csv")
    out_png = os.path.join(args.out, "cluster_plot.png")
    out_json = os.path.join(args.out, "tree.json")

    write_results(leaf_summary, kad_summaries, out_csv)
    plot_cluster_means(X, tree["leaves"], out_png)
    with open(out_json, "w") as f:
        json.dump(json_friendly_tree(tree), f, indent=2)

    print("[done] CSV:", out_csv)
    print("[done] PNG:", out_png)
    print("[done] JSON:", out_json)

if __name__ == "__main__":
    main()
