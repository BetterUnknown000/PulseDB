# PulseDB

**Project for CMPSC 463**

This project is required for my CMPSC 463 class.  
It implements algorithmic clustering of physiological signal segments using divide-and-conquer techniques, correlation-based similarity, and Kadane’s algorithm.

---

## Overview

This project focuses on three main algorithmic ideas:

1. **Divide and Conquer Clustering** – recursively splits signal segments based on correlation distance.  
2. **Correlation-Based Similarity** – compares time-series segments using Pearson correlation.  
3. **Kadane’s Algorithm** – finds the most active (maximum-sum) interval within each signal segment.

---

## Installation

Before running the project, install the required libraries:

```bash
pip install -r requirements.txt
