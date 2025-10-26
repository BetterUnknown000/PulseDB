# run.py
import argparse, os, sys, json, csv, math, random

def _need(mod, pipname=None):
    try:
        __import__(mod)
    except Exception as e:
        pkg = pipname or mod
        print(f"[error] missing '{pkg}'. install with: pip install {pkg}\n{e}")
        sys.exit(1)

_need("numpy"); _need("scipy"); _need("matplotlib")
import numpy as np
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- data io ----------------
def _stack_equal(seqs):
    xs = []
    for s in seqs:
        if s is None: 
            continue
        a = np.asarray(s, dtype=float).ravel()
        if a.size > 0 and np.isfinite(a).any():
            xs.append(a)
    if not xs:
        raise ValueError("no usable segments")
    L = min(len(a) for a in xs)
    return np.vstack([a[:L] for a in xs])

def _to2d(x):
    if isinstance(x, (list, tuple)):
        return _stack_equal(x)
    a = np.asarray(x, dtype=float)
    if a.dtype == object:
        return _stack_equal([np.asarray(t, dtype=float).ravel() for t in a.ravel().tolist()])
    if a.ndim == 1: 
        return a[None, :]
    if a.ndim == 2: 
        return a
    raise ValueError("cannot coerce to 2d")

def load_mat(mat_path, want=None):
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(mat_path)
    d = loadmat(mat_path)
    keys = [k for k in d.keys() if not k.startswith("__")]
    print("[mat] keys:", ", ".join(keys))

    def pick(cands):
        low = {k.lower(): k for k in keys}
        for c in cands:
            if c in low: 
                return low[c]
        for k in keys:
            if any(c in k.lower() for c in cands):
                return k
        return None

    if want:
        k = pick([want.lower()])
        if k is None:
            raise KeyError(f"signal '{want}' not found; available: {keys}")
    else:
        k = pick(["ppg","ecg","abp"])
        if k is None:
            # fallback: first coercible numeric thing
            for kk in keys:
                try:
                    _ = _to2d(d[kk])
                    k = kk; break
                except Exception:
                    pass
            if k is None:
                raise KeyError(f"no numeric signal key detected; available: {keys}")

    X = _to2d(d[k])
    print("[mat] selected:", k)
    print("[mat] raw shape:", X.shape)
    if not np.isfinite(X).any():
        raise ValueError("all values are non-finite")
    X = np.where(np.isfinite(X), X, 0.0)
    return X, k

def make_demo(n=120, L=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 6*np.pi, L)
    xs = []
    for i in range(n):
        z = 0.2 * rng.standard_normal(L)
        if i % 3 == 0: s = np.sin(t) + z
        elif i % 3 == 1: s = np.sin(2*t+0.5) + 0.5*np.cos(0.5*t) + z
        else: s = np.sign(np.sin(0.8*t)) + 0.3*np.sin(3*t) + z
        xs.append(s)
    return np.vstack(xs), "DEMO"

# ---------------- algorithms ----------------
def znorm(x):
    x = np.asarray(x, dtype=float)
    m, s = x.mean(), x.std()
    if s == 0 or not np.isfinite(s): 
        return x - m
    return (x - m) / (s + 1e-8)

def corr_dist(a, b):
    a = znorm(a); b = znorm(b)
    num = float(np.dot(a, b))
    den = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return 1.0 - (num / den)

def kadane(x):
    best, cur = -float("inf"), 0.0
    L = R = l = 0
    for r, v in enumerate(x):
        v = float(v)
        if cur <= 0:
            cur, l = v, r
        else:
            cur += v
        if cur > best:
            best, L, R = cur, l, r+1
    return L, R, best

def closest_pair_bruteforce(X, idxs):
    idxs = np.asarray(idxs, dtype=int)
    if len(idxs) < 2:
        return (int(idxs[0]), int(idxs[0])), float("inf")
    best = (None, None); bd = float("inf")
    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            d = corr_dist(X[idxs[i]], X[idxs[j]])
            if d < bd: bd, best = d, (int(idxs[i]), int(idxs[j]))
    return best, bd

# ---------------- clustering ----------------
class DAC:
    def __init__(self, min_size=25, max_depth=None, k=16, seed=0):
        self.min = int(min_size)
        self.maxd = max_depth
        self.k = int(k)
        self.rng = random.Random(seed)
        self.leaves = {}

    def fit(self, X):
        idxs = np.arange(X.shape[0])
        root = self._build(X, idxs, 0, 0)
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
        stop = len(idxs) <= self.min or (self.maxd is not None and depth >= self.maxd)
        if stop:
            self.leaves[nid] = {"idxs": np.array(idxs, int)}
            node["leaf"] = True
            return node
        L, R = self._split(X, idxs)
        if len(L) == 0 or len(R) == 0:
            self.leaves[nid] = {"idxs": np.array(idxs, int)}
            node["leaf"] = True
            return node
        node["leaf"] = False
        node["children"].append(self._build(X, L, depth+1, nid*2+1))
        node["children"].append(self._build(X, R, depth+1, nid*2+2))
        return node

# ---------------- reporting ----------------
def write_summary(leaf_rows, kad_rows, path_csv):
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    with open(path_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# Cluster summary"])
        w.writerow(["cluster_id","size","closest_pair_ids","min_distance"])
        for r in leaf_rows:
            w.writerow([r["cluster_id"], r["size"], r["closest_pair_ids"], f"{r['min_distance']:.6f}"])
        w.writerow([])
        w.writerow(["# Kadane per segment"])
        w.writerow(["id","kadane_start","kadane_end","kadane_sum"])
        for r in kad_rows:
            w.writerow([r["id"], r["kadane_start"], r["kadane_end"], f"{r['kadane_sum']:.6f}"])

def plot_means(X, leaves, path_png, top=6):
    os.makedirs(os.path.dirname(path_png), exist_ok=True)
    order = sorted(leaves.keys(), key=lambda nid: len(leaves[nid]["idxs"]), reverse=True)[:top]
    plt.figure(figsize=(10,6))
    for nid in order:
        idxs = leaves[nid]["idxs"]
        m = X[idxs].mean(axis=0)
        plt.plot(m, label=f"leaf {nid} (n={len(idxs)})")
    plt.title("Cluster mean signals")
    plt.xlabel("time"); plt.ylabel("amplitude (z-norm)")
    plt.legend(); plt.tight_layout(); plt.savefig(path_png, dpi=150); plt.close()

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", type=str, default="", help="path to .mat (Announcement #1)")
    ap.add_argument("--signal", type=str, default="", help="PPG|ECG|ABP (optional)")
    ap.add_argument("--n", type=int, default=1000, help="max segments")
    ap.add_argument("--out", type=str, default="out/", help="output folder")
    ap.add_argument("--min_size", type=int, default=25, help="leaf size")
    ap.add_argument("--max_depth", type=int, default=None, help="depth cap")
    ap.add_argument("--demo", action="store_true", help="toy data")
    args = ap.parse_args()

    if not args.demo and not args.mat:
        print("[usage] provide --mat <file> or use --demo")
        sys.exit(2)

    os.makedirs(args.out, exist_ok=True)

    # load
    if args.demo:
        Xraw, sig = make_demo()
    else:
        Xraw, sig = load_mat(args.mat, args.signal.strip() or None)

    # shape checks
    if Xraw.ndim != 2:
        raise ValueError(f"expected 2d, got {Xraw.ndim}d")
    n, L = Xraw.shape
    if n < 2 or L < 16:
        raise ValueError(f"not enough data: (n,L)=({n},{L})")
    print("[sample] first segment (first 5):", np.asarray(Xraw[0][:5]).round(3).tolist())

    # limit n, z-norm
    if args.n and n > args.n:
        Xraw = Xraw[:args.n]
        n = Xraw.shape[0]
    X = np.vstack([znorm(x) for x in Xraw])
    print("[data] final shape (n, L):", X.shape)

    # cluster
    dac = DAC(min_size=args.min_size, max_depth=args.max_depth, seed=0)
    tree = dac.fit(X)
    print("[cluster] leaves:", len(tree["leaves"]))

    # leaf info
    leaf_rows = []
    for nid, meta in tree["leaves"].items():
        idxs = meta["idxs"]
        pair, dist = closest_pair_bruteforce(X, idxs)
        leaf_rows.append({
            "cluster_id": int(nid),
            "size": int(len(idxs)),
            "closest_pair_ids": f"{pair[0]}-{pair[1]}",
            "min_distance": float(dist),
        })

    # kadane per segment
    kad_rows = []
    for gid in range(X.shape[0]):
        l, r, s = kadane(X[gid])
        kad_rows.append({"id": int(gid), "kadane_start": int(l), "kadane_end": int(r), "kadane_sum": float(s)})

    # outputs
    out_csv = os.path.join(args.out, "summary.csv")
    out_png = os.path.join(args.out, "cluster_examples.png")
    out_json = os.path.join(args.out, "tree.json")

    write_summary(leaf_rows, kad_rows, out_csv)
    plot_means(X, tree["leaves"], out_png)
    with open(out_json, "w") as f: json.dump(tree, f, indent=2)

    print("[ok] wrote", out_csv)
    print("[ok] wrote", out_png)
    print("[ok] wrote", out_json)

if __name__ == "__main__":
    main()
