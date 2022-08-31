#!/bin/bash

# Activate conda
conda activate paranet

# Runs pytest for the univariate scripts
python3 -m pytest tests/test_univariate_censoring.py tests/test_univariate_gradients.py tests/test_univariate_integration tests/test_univariate_vecdist
