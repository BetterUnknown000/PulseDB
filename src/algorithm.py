import numpy as np

# Normalize the input array using the Z-score
# Add a small epsilon to avoid any "divde by zero" errors"
def znorm(arr: np.ndarray) -> np.ndarray:
  arr = np.asarray(arr, dtype=float) # Float check
  mean = arr.mean()
  std = arr.std()
  if std == 0:
    return arr - mean
  return (arr -mean) / (std + 1e-8) # Adding epsilon just for safety

# Pearson correlation distance, based on z-normalized inputs:
def corr_distance(a: np.ndarray, b: np.ndarray) -> float:
  a = znorm(a)
  b = znorm(b)

  # If both are near the zero after z-norm, treat identically
  na = np.linalg.norm(a)
  nb = np.linalg.norm(b)
  if na < 1e-12 and nb < 1e-12:
    return 0.0

  sim = float(np.dot(a, b) / (na * nb + 1e-8))
  return 1.0 - sim

# Fast path when the whole matrix was z-normalized
# Avoids re-normalizing inside the loop
def corr_distance_znormed(a: np.ndarray, b: np.ndarray) -> float:
  na = np.linalg.norm(a)
  nb = np.linalg.norm(b)
  if na < 1e-12 and nb < 1e-12:
    return 0.0
  sim = float(np.dot(a, b) / (na * nb + 1e-8))
  return 1.0 - sim

# Kadane's Algorithm to find max subarray sum
# Returns (start_idx, end_idx, max_sum)
def kadane(arr: np.ndarray):
  best = -float("inf")
  curr = 0.0
  start = 0
  best_l = 0
  best_r = 0
  for r, val in enumerate(arr):
    if curr <= 0:
      curr = float(val)
      start = r
    else:
      curr += float(val)
    if curr > best:
      best = curr
      best_l = start
      best_r = r + 1
  return best_l, best_r, best

# Brute-force closest pair of rows by correlational distance.
# Returns (min_dist, (i, j)) with i < j. If fewer than 2 rows, return (inf, (None, None))
def closest_pair_bruteforce(data: np.ndarray, already_znormed: bool = False):
  m = data.shape[0]
  if m < 2:
    return float("inf"), (None, None)

  dmin, pair = float("inf"), (None, None)
  dist_fn = corr_distance_znormed if already_znormed else corr_distance

  for i in range(m):
    for j in range(i + 1, m):
      d = dist_fn(data[i], data[j])
      if d < dmin:
        dmin, pair = d, (i, j)
  return dmin, pair

# References:
# 1. "Kadane’s Algorithm – Maximum Subarray Sum", GeeksforGeeks.
#    https://www.geeksforgeeks.org/largest-sum-contiguous-subarray/
# 2. "Divide and Conquer Algorithm – Explained with Examples", Programiz.
#    https://www.programiz.com/dsa/divide-and-conquer
# 3. "How to Calculate a Z-Score in Python", Statology.
#    https://www.statology.org/z-score-python/
# 4. "How to Calculate Correlation in Python", DataCamp.
#    https://www.datacamp.com/tutorial/correlation-python
