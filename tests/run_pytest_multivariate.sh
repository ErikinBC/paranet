#!/bin/bash

# ---------------------------------------- #
# Runs pytest for the multivariate scripts #
# ---------------------------------------- #

# Activate conda
conda activate paranet

# Default is to run pytest unless argument is supplied
n_char1=${#1}

# Note! The last two script can take a while to run

if [[ $n_char1 -gt 0 ]]; then
    echo "Argument supplied, running python"
    python3 -m tests.test_multivariate_dists
    python3 -m tests.test_multivariate_parametric
    python3 -m tests.test_multivariate_scaling
    python3 -m tests.test_multivariate_censoring
    python3 -m tests.test_multivariate_solver
else
    echo "No argument supplied, running pytest"
    python3 -m pytest tests/test_multivariate_dists.py 
    python3 -m pytest tests/test_multivariate_parametric.py
    python3 -m pytest tests/test_multivariate_scaling.py 
    python3 -m pytest tests/test_multivariate_censoring.py
    python3 -m pytest tests/test_multivariate_solver.py
fi


echo "~~~ End of run_pytest_multivariate.sh ~~~"