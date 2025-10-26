import argparse
import os
import json

# Note: importing from our local modules - double-check these paths if things break
from src.data_io import load_pulsedb_mat, make_demo
from src.clustering import DivideAndConquerClustering
from src.algorithms import closest_pair_bruteforce, kadane
from src.reporting import save_summary, plot_clusters_quick


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=str, default="", help="path to .mat file (should be local)")
    parser.add_argument("--signal", type=str, default="PPG", help="PPG / ECG / ABP")
    parser.add_argument("--n", type=int, default=1000, help="maximum number of segments")
    parser.add_argument("--out", type=str, default="out/", help="folder to save outputs")
    parser.add_argument("--min_size", type=int, default=25, help="min size before clustering stops")
    parser.add_argument("--demo", action="store_true", help="if set, loads toy/demo data")
    args = parser.parse_args()

    # Make sure output directory is ready
    if not os.path.exists(args.out):
        os.makedirs(args.out)  # using if-check instead of exist_ok for more verbosity

    # Decide on data source
    if args.demo or args.mat == "":
        # Only load a small number for demo
        sample_ids, signals = make_demo(n=min(args.n, 50), length=500)
    else:
        # loading from the actual mat file
        sample_ids, signals = load_pulsedb_mat(
            mat_path=args.mat,
            signal_type=args.signal,
            limit=args.n
        )

    # Cluster using divide-and-conquer median-based approach
    clusterer = DivideAndConquerClustering(min_size=args.min_size)
    cluster_tree = clusterer.fit(signals)

    # Build per-leaf report (size + closest pair)
    leaf_report = []
    for clust_id, indices in cluster_tree["leaves"].items():
        indices = list(indices)
        if len(indices) < 2:
            leaf_report.append({
                "cluster_id": clust_id,
                "size": len(indices),
                "closest_pair_ids": "",          # nothing to compare
                "min_distance": float("inf"),
        })
        continue

        # compute closest pair **inside** the loop for this leaf
        dmin, (i1, i2) = closest_pair_bruteforce(signals[indices])
        leaf_report.append({
            "cluster_id": clust_id,
            "size": len(indices),
            "closest_pair_ids": f"{sample_ids[indices[i1]]},{sample_ids[indices[i2]]}",
            "min_distance": float(dmin),
        })


    # Kadane analysis across each signal
    kad_summary = []
    for idx in range(len(signals)):
        sig = signals[idx]
        start, end, max_sum = kadane(sig)
        kad_summary.append({
            "id": sample_ids[idx],
            "kadane_start": int(start),
            "kadane_end": int(end),
            "kadane_sum": float(max_sum)
        })

    # Save the outputs – CSV, PNG, and JSON
    summary_path = os.path.join(args.out, "summary.csv")
    tree_img_path = os.path.join(args.out, "cluster_examples.png")
    tree_json_path = os.path.join(args.out, "tree.json")

    save_summary(leaf_report, kad_summary, summary_path)
    plot_clusters_quick(signals, cluster_tree, tree_img_path)

    with open(tree_json_path, "w") as jf:
        json.dump(cluster_tree, jf)

    print("All done! Check outputs in:", args.out)


# just being safe with the standard main guard
if __name__ == "__main__":
    main()
