# run.py - self-contained runner for clustering signal segments

import argparse
import os, sys, json, csv, random, hashlib

# quick-and-dirty import checker for required packages
def _need(mod, pip_name=None):
    try:
        __import__(mod)
    except Exception as e:
        pip_fallback = pip_name or mod
        print(f"[error] You’re missing '{pip_fallback}'. Try: pip install {pip_fallback}\n{e}")
        sys.exit(1)

# basic checks for expected deps
_need("numpy")
_need("scipy")
_need("matplotlib")

import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")  # non-GUI backend for saving plots
import matplotlib.pyplot as plt

# default path where .mat files live
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


# ---------- data helpers ----------

def _stack_equal(chunks):
    # Flattens and clips sequences to shortest length
    valid = []
    for c in chunks:
        if c is None: continue
        a = np.asarray(c, dtype=float).ravel()
        if a.size > 0 and np.isfinite(a).any():
            valid.append(a)
    if not valid:
        raise ValueError("All input segments are empty or invalid")
    L = min(len(a) for a in valid)
    return np.vstack([a[:L] for a in valid])


def _to2d(x):
    # Not the prettiest coercion but works in general
    if isinstance(x, (list, tuple)):
        return _stack_equal(x)
    a = np.asarray(x, dtype=float)
    if a.dtype == object:
        return _stack_equal([np.asarray(t, dtype=float).ravel() for t in a.ravel().tolist()])
    if a.ndim == 1:
        return a[np.newaxis, :]
    if a.ndim == 2:
        return a
    raise ValueError("Can't force input to 2D")


def _md5(path, chunk_size=1 << 20):
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _autofind_mat(data_dir=DEFAULT_DATA_DIR):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"No such directory: {data_dir}")
    mats = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".mat")]
    if not mats:
        raise FileNotFoundError(f"No .mat files found in {data_dir}")
    mats.sort()
    return os.path.abspath(mats[0])


def load_mat(mat_path, want=None):
    d = loadmat(mat_path)
    keys = [k for k in d if not k.startswith("__")]
    print("[mat] found keys:", ", ".join(keys))

    def fuzzy_find(candidates):
        lookup = {k.lower(): k for k in keys}
        for c in candidates:
            if c in lookup:
                return lookup[c]
        for k in keys:
            if any(c in k.lower() for c in candidates):
                return k
        return None

    target_key = fuzzy_find([want.lower()]) if want else fuzzy_find(["ppg", "ecg", "abp"])

    if not target_key:
        for k in keys:
            try:
                _ = _to2d(d[k])
                target_key = k
                break
            except: pass
        if not target_key:
            raise KeyError(f"Couldn't locate usable signal in: {keys}")

    sig_data = _to2d(d[target_key])
    print("[mat] using:", target_key)
    print("[mat] shape:", sig_data.shape)

    if not np.isfinite(sig_data).any():
        raise ValueError("All signal values are non-finite")

    sig_data = np.where(np.isfinite(sig_data), sig_data, 0.0)
    return sig_data, target_key


def make_demo(n=120, L=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 6*np.pi, L)
    xs = []
    for i in range(n):
        z = 0.2 * rng.standard_normal(L)
        if i % 3 == 0:
            s = np.sin(t) + z
        elif i % 3 == 1:
            s = np.sin(2*t + 0.5) + 0.5*np.cos(0.5*t) + z
        else:
            s = np.sign(np.sin(0.8*t)) + 0.3*np.sin(3*t) + z
        xs.append(s)
    return np.vstack(xs), "DEMO"


# ---------- basic signal ops ----------

def znorm(x):
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    std = x.std()
    return (x - mu) / (std + 1e-8) if std and np.isfinite(std) else (x - mu)


def corr_dist(a, b):
    a = znorm(a)
    b = znorm(b)
    dot = float(np.dot(a, b))
    norm_product = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return 1.0 - (dot / norm_product)


def kadane(x):
    max_sum = -float("inf")
    curr_sum = 0.0
    L = R = l = 0
    for r, val in enumerate(x):
        val = float(val)
        if curr_sum <= 0:
            curr_sum = val
            l = r
        else:
            curr_sum += val
        if curr_sum > max_sum:
            max_sum = curr_sum
            L, R = l, r+1
    return L, R, max_sum


def closest_pair_bruteforce(X, idxs):
    idxs = np.asarray(idxs, dtype=int)
    if len(idxs) < 2:
        return (int(idxs[0]), int(idxs[0])), float("inf")
    best_pair = (None, None)
    min_dist = float("inf")
    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            d = corr_dist(X[idxs[i]], X[idxs[j]])
            if d < min_dist:
                min_dist = d
                best_pair = (int(idxs[i]), int(idxs[j]))
    return best_pair, min_dist


# ---------- clustering logic ----------

class DAC:
    def __init__(self, min_size=25, max_depth=None, k=16, seed=0):
        self.min = int(min_size)
        self.maxd = max_depth
        self.k = int(k)
        self.rng = random.Random(seed)
        self.leaves = {}

    def fit(self, X):
        all_idxs = np.arange(X.shape[0])
        root = self._build(X, all_idxs, 0, 0)
        return {"root": root, "leaves": self.leaves}

    def _pick2(self, X, idxs):
        s0 = self.rng.choice(list(idxs))
        sample = self.rng.sample(list(idxs), min(self.k, len(idxs)))
        s1 = max(((corr_dist(X[s0], X[j]), j) for j in sample))[1]
        return s0, s1

    def _split(self, X, idxs):
        if len(idxs) < 2:
            return idxs, np.array([], dtype=int)
        a, b = self._pick2(X, idxs)
        L, R = [], []
        for j in idxs:
            da = corr_dist(X[j], X[a])
            db = corr_dist(X[j], X[b])
            (L if da <= db else R).append(j)
        return np.asarray(L, int), np.asarray(R, int)

    def _build(self, X, idxs, depth, nid):
        node = {"id": int(nid), "size": int(len(idxs)), "children": []}
        stop_here = len(idxs) <= self.min or (self.maxd is not None and depth >= self.maxd)
        if stop_here:
            self.leaves[nid] = {"idxs": np.array(idxs, int)}
            node["leaf"] = True
            return node
        L, R = self._split(X, idxs)
        if len(L) == 0 or len(R) == 0:
            self.leaves[nid] = {"idxs": np.array(idxs, int)}
            node["leaf"] = True
            return node
        node["leaf"] = False
        node["children"].append(self._build(X, L, depth + 1, nid * 2 + 1))
        node["children"].append(self._build(X, R, depth + 1, nid * 2 + 2))
        return node


# ---------- output + plots ----------

def write_summary(leaf_rows, kad_rows, path_csv):
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# Cluster summary"])
        w.writerow(["cluster_id", "size", "closest_pair_ids", "min_distance"])
        for r in leaf_rows:
            w.writerow([r["cluster_id"], r["size"], r["closest_pair_ids"], f"{r['min_distance']:.6f}"])
        w.writerow([])
        w.writerow(["# Kadane per segment"])
        w.writerow(["id", "kadane_start", "kadane_end", "kadane_sum"])
        for r in kad_rows:
            w.writerow([r["id"], r["kadane_start"], r["kadane_end"], f"{r['kadane_sum']:.6f}"])


def plot_means(X, leaves, path_png, top=6):
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    top_ids = sorted(leaves, key=lambda k: len(leaves[k]["idxs"]), reverse=True)[:top]
    plt.figure(figsize=(10, 6))
    for nid in top_ids:
        idxs = leaves[nid]["idxs"]
        mean_sig = X[idxs].mean(axis=0)
        plt.plot(mean_sig, label=f"leaf {nid} (n={len(idxs)})")
    plt.title("Cluster mean signals")
    plt.xlabel("time")
    plt.ylabel("amplitude (z-norm)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_png, dpi=150)
    plt.close()


# ---------- main program ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=str, default="", help="path to .mat (optional)")
    parser.add_argument("--signal", type=str, default="", help="Signal name: PPG|ECG|ABP")
    parser.add_argument("--n", type=int, default=1000, help="max segments to load")
    parser.add_argument("--out", type=str, default="out/", help="output folder")
    parser.add_argument("--min_size", type=int, default=25, help="min leaf size")
    parser.add_argument("--max_depth", type=int, default=None, help="optional depth limit")
    parser.add_argument("--demo", action="store_true", help="use toy data instead of .mat")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.demo:
        Xraw, sig = make_demo()
        dataset_path, dataset_md5 = "<DEMO>", "<NA>"
    else:
        dataset_path = args.mat.strip() or _autofind_mat()
        dataset_md5 = _md5(dataset_path)
        Xraw, sig = load_mat(dataset_path, args.signal.strip() or None)

    print("[dataset]", dataset_path)
    print("[md5]    ", dataset_md5)

    if Xraw.ndim != 2:
        raise ValueError(f"expected 2D array, got {Xraw.ndim}D")
    n, L = Xraw.shape
    if n < 2 or L < 16:
        raise ValueError(f"not enough data to continue: (n,L)=({n},{L})")

    print("[sample] segment preview:", np.round(Xraw[0][:5], 3).tolist())

    if args.n and n > args.n:
        Xraw = Xraw[:args.n]

    X = np.vstack([znorm(seg) for seg in Xraw])
    print("[data] using (n,L):", X.shape)

    dac = DAC(min_size=args.min_size, max_depth=args.max_depth, seed=0)
    tree = dac.fit(X)
    print("[cluster] total leaves:", len(tree["leaves"]))

    leaf_rows = []
    for nid, meta in tree["leaves"].items():
        idxs = meta["idxs"]
        pair, dist = closest_pair_bruteforce(X, idxs)
        leaf_rows.append({
            "cluster_id": int(nid),
            "size": len(idxs),
            "closest_pair_ids": f"{pair[0]}-{pair[1]}",
            "min_distance": float(dist)
        })

    kad_rows = []
    for gid in range(X.shape[0]):
        l, r, s = kadane(X[gid])
        kad_rows.append({
            "id": gid,
            "kadane_start": l,
            "kadane_end": r,
            "kadane_sum": s
        })

    out_csv = os.path.join(args.out, "summary.csv")
    out_png = os.path.join(args.out, "cluster_examples.png")
    out_json = os.path.join(args.out, "tree.json")

    write_summary(leaf_rows, kad_rows, out_csv)
    plot_means(X, tree["leaves"], out_png)

    with open(out_json, "w") as f:
        json.dump(tree, f, indent=2)

    print("[ok] summary written to:", out_csv)
    print("[ok] plot saved to:", out_png)
    print("[ok] tree saved to:", out_json)


if __name__ == "__main__":
    main()
