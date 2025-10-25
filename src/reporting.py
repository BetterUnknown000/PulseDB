import csv
import os
import matplotlib.pyplot as plt

# Writes out two sections to a CSV file:
# 1. Summary of each cluster, with (size, closest pair, min dist)
# 2. Kadane's result for each signal/segment

def save_summary(cluster_data, kadane_data, csv_path: str):
  # Check to see if directory exists before writing
  os.makedirs(os.path.dirname(csv_path), exist_ok=True)
  
  with open(csv_path, "w", newline="") as outfile:
    writer = csv.writer(outfile)
    
    # 1 - Cluster Summary Info to Print
    writer.writerow(["Cluster Summary"])
    writer.writerow(["cluster_id", "size", "closest_pair_ids", "min_distance"])
    for cluster in cluster_data:
      writer.writerow([
        cluster["cluster_id"],
        cluster["size"],
        cluster["closest_pair_ids"],
        f"{cluster['min_distance']:.6f}"
      ])
      
    writer.writerow([]) # Spacing

    writer.writerow(["Kadane per segment"])
    writer.writerow(["id", "kadane_start", "kadane_end", "kadane_sum"])
    for seg in kadane_data:
      writer.writerow([
        seg["id"],
        seg["kadane_start"],
        seg["kadane_end"],
        f"{seg['kadane_sum']:.6f}"
      ])

def plot_clusters_quick(X, tree_data, png_path: str, max_clusters=2, max_traces=5):
  # I there are no leaves, just return.
  if not tree_data.get("leaves"):
    return
 
  leaf_items = list(tree_data["leaves"].items())[:max_clusters]
  if not leaf_items:
    return

  num_panels = len(leaf_items)
  fig, axes = plt.subplots(1, num_panels, figsize=(6 * num_panels, 4))

  # In the case of only one plot, make sure axes are iterable
  if num_panels == 1:
    axes = [axes]

  for ax, (cluster_id, indices) in zip(axes, leaf_items):
    # Grab a few example series from the cluster
    sample_idxs = indices[:max_traces]
    for i in sample_idxs:
      ax.plot(X[i], linewidth=1)

    ax.set_title(f"Cluster {cluster_id} (n={len(indices)})")
    ax.set_xlabel("t")
    ax.set_ylabel("z-amplitude")

  fig.tight_layout()

  os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
  fig.savefig(png_path, dpi=150)
  plt.close(fig)

# References:
# 1. "Matplotlib .pyplot savefig() in Python" – GeeksforGeeks.
#    https://www.geeksforgeeks.org/matplotlib-pyplot-savefig-in-python/
# 2. "Writing CSV files in Python" – Programiz.
#    https://www.programiz.com/python-programming/writing-csv-files
# 3. "Use a loop to plot n charts Python" – Stack Overflow.
#    https://stackoverflow.com/questions/19189488/use-a-loop-to-plot-n-charts-python
# 4. "How to Plot Multiple Graphs in a For Loop with iPython/Jupyter Notebook and Pandas" – Saturn Cloud Blog.
#    https://saturncloud.io/blog/how-to-plot-multiple-graphs-in-a-for-loop-with-ipythonjupyter-notebook-and-pandas/
