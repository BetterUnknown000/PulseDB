# PulseDB
Project for CMPSC 463 

This project is required for my CMPSC 463 class. I am planning on implementing an algorithmic clustering of physiological signal segments.

## Overview
I will focus on three main ideas:

1. Divide and Conquer Clustering - recursive splitting of segments
2. Correlation-Based Similarity - comparing signals using Pearson correlation
3. Kadane's Algorithm - Used on each signal (hopefully) to detect active intervals.

## Run
'''bash
# Install requirements that I will implement
pip install -r requirements.txt

# Demo run
# Hopefully?
python run.py --demo --n 50 --out out/
