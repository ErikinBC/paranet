#!/bin/bash

# Activate conda
conda activate paranet

# Runs pytest for the univariate scripts
python3 -m pytest tests/test_multivariate_dists.py tests/test_multivariate_parametric.py

