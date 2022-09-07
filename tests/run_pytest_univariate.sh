#!/bin/bash

# Activate conda
conda activate paranet

# Runs pytest for the univariate scripts
python3 -m pytest tests/test_univariate_vecdist.py
python3 -m pytest tests/test_univariate_gradients.py
python3 -m pytest tests/test_univariate_censoring.py 
python3 -m pytest tests/test_univariate_integration.py

echo "~~~ End of run_pytest_univariate.sh ~~~"