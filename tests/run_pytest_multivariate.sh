#!/bin/bash

# Activate conda
conda activate paranet

# Runs pytest for the univariate scripts
python3 -m pytest tests/test_multivariate_dists.py 
python3 -m pytest tests/test_multivariate_parametric.py

# Note! The censoring script takes a while to run!
python3 -m pytest tests/test_multivariate_censoring.py

