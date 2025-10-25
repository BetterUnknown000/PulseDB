import numpy as np
from .algorithms import corr_distance

# This is a quick divide & conquer clustering approach
# This recursively splits data based on the correlation distance until it can't split
class DivideAndConquerClustering:
  def __init__(self, min_size=25, max_depth=None):
    self.min_size = min_size
    self.max_depth = max_depth
    self._leaf_clusters = {} # This will store the final clusters after the stopping condition

  def fit(self, X):
    # Starts the recursive clustering process
    total_indices = np.arange(X.shape[0])
    root_node = self._recurse(X, total_indices, depth=0, node_id=0)
    return {"root": root_node, "leaves": self._leaf_clusters}

  def _recurse(self, data_matrix, current_idxs, depth, node_id):
    # Each call builds a node
    this_node = {
      "id":  node_id,
      "size": int(len(current_idxs)), # Prevent floats
      "children": []
    }
    # Stopping condition 
    if len(current_idxs) <= self.min_size or (
      self.max_depth is not None and depth >= self.max_depth
    ):
      # Register this group as a terminal cluster
      self._leaf_clusters[node_id] = current_idxs.tolist()
      return this_node

    # Pick a pivot. Using the first one
    pivot_idx = current_idxs[0]

    # Computer the correlational distance from pivot to all other points
    distances = []
    for i in current_idxs:
      dist = corr_distance(data_matrix[pivot_idx], data_matrix[i])
      distances.append(dist)
    distances = np.array(distances)

    # Split using median distance
    median_dist = np.median(distances)
    is_left = distances <= median_dist
    is_right = ~is_left

    # If one side ends empty, stop splitting
    if np.sum(is_left) == 0 or np.sum(is_right) == 0:
      self._leaf_clusters[node_id] = current_idxs.tolist()
      return this_node

    # Split the indices now
    left = current_idxs[is_left]
    right = current_idxs[is_right]

    # Recursively build the child nodes
    # Building them BST style. Left child = 2i + 1. Right child = 2i + 2.
    left_child = self._recurse(data_matrix, left, depth + 1, node_id * 2 + 1)
    right_child = self._recurse(data_matrix, right, depth + 1, node_id * 2 + 2)
    
    this_node["children"].append(left_child)
    this_node["children"].append(right_child)
    
    return this_node


# References:
# 1. "NumPy Quickstart Tutorial" – NumPy.org
#    https://numpy.org/doc/stable/user/quickstart.html
# 2. "Python NumPy Tutorial" – W3Schools
#    https://www.w3schools.com/python/numpy_intro.asp
# 3. "NumPy Tutorial: Introduction to Numerical Python" – Programiz
#    https://www.programiz.com/python-programming/numpy


    
