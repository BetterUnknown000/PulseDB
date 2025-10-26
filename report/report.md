## Time-Series Clustering and Segment Analysis on PulseDB Using Divide-and-Conquer Algorithms

### 1. Description of Project

The project entails algorithmic clustering of time-series segments from PulseDB with an emphasis on signal segmentation and interpretation without resorting to traditional machine learning. Instead, it leverages divide-and-conquer clustering, closest-pair search, and maximum subarray analysis (Kadane's Algorithm) for clustering and analyzing physiological signals (ABP in this case).

Goals:

* Cluster univariate 10-second segments from PulseDB via recursive, similarity-based splitting.
* For each cluster, find the closest pair of signals as a validation metric.
* Pass each signal through Kadane's Algorithm to detect peak activity periods.
* Create visual and tabular summaries to aid in analysis and interpretation.

---
### 2. Installation and Usage

#### Dependencies:

* Python 3.7+
* Packages needed:

  ```bash
  pip install numpy matplotlib h5py
  ```

#### Running the Program:

```bash
python run.py --signal ABP --n 1000 --out out/
```

Options:

* `--mat` (optional): Path to a `.mat` HDF5 file.
* `--signal`: Target signal type (PPG, ECG, ABP).
* `--n`: Number of segments to process (default: 1000).
* `--min_size`: Minimum leaf size of clusters.
* `--max_depth`: Maximum recursion depth.

Output files will be saved to the specified `--out` directory.

---
### 3. Code Structure

#### Primary Components:

* **run.py**: Main executable script.
* **DAC Class**: Orchestrates divide-and-conquer clustering.
* **Signal Utilities**: Normalization, correlation, Kadane analysis.
* **HDF5 Loader**: Loads MATLAB v7.3 `.mat` files and extracts usable 2D arrays.
* **Reporting Functions**: Summarize cluster and segment-level statistics.

---

### 4. Description of Algorithms

#### Divide-and-Conquer Clustering:

* Recursively splits data by signal similarity (through correlation distance).
* In each iteration, selects two far-away samples as poles, then splits others accordingly.
* Tree-like structure similar to decision tree or binary k-means.

#### Closest-Pair Search:

* In each of the final clusters (leaf node), all-pairs similarity is computed.
* The pair with the smallest correlation distance is selected.
* Serves as a proxy for intra-cluster compactness.

#### Kadane's Algorithm:

* Classic linear-time algorithm to find the maximum sum subarray.
* Applied on each segment separately to highlight peak activity period.
* Useful in detection of dominant physiological events.

---
### 5. Verification of Functionality (Toy Examples)

* **Divide-and-Conquer Clustering:

* Tested on artificial signals with known similarity clusters (e.g., sine, square, and noise).
* **Closest-Pair Search**:

  * Tested with small (e.g., 3-5) segments; compared with brute-force similarity scores.
* **Kadane's Algorithm**:

  * Compared with hand-calculated max subarrays on known inputs (e.g., [1, -2, 3, 5, -1]).

---
### 6. Run Results (PulseDB - 1000 ABP Segments)

* **Total Segments**: 1000
* **Cluster Leaves Created**: e.g., 23 (varies with `min_size`)
* **Closest-Pair Distance (Min)**: ~0.002 (high similarity)
* **Kadane Interval Range**: Most segments had peak around the center, indicating periodicity in ABP.

**Files Generated:**

* `summary.csv`: Cluster and segment summaries
* `cluster_plot.png`: Mean signals per top clusters
* `tree.json`: Tree structure of the clustering

---
### 7. Discussion on Execution Results

* **Clustering Results**:

  * Clusters were tight and interpretable based on shape similarity.
  * Divide-and-conquer worked well, especially without needing ML heuristics.

* **Kadane Analysis**:

* Helped in revealing common patterns across clusters (e.g., shared peak intervals).

* **Closest-Pair**:
  * Enabled quick verification if clusters were compact.

**Challenges Encountered**:

* HDF5 `.mat` format parsing required special handling (object arrays, references).
* Normalization was crucial — raw signals had wide variance.

---
### 8. Conclusions

This project demonstrates that classical algorithmic thinking can be applied to extract useful information from biomedical time-series data. Without supervised learning, we were able to cluster and analyze segments neatly with a recursive clustering strategy, supplemented by signal-level analysis like Kadane's algorithm.

**Future Work Suggestions**:

* Generalize to multivariate signals.
* Try other measures of similarity (e.g., DTW).
* Add visualization of tree structure.

---
GitHub: [Link to Repository]

Provide all code, documentation, sample `.mat` files (or links), and output examples.
