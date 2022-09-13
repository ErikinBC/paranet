#!/bin/bash

# -------------------------------------- #
# Runs pytest for the univariate scripts #
# -------------------------------------- #

# Activate conda
conda activate paranet

# Default is to run pytest unless argument is supplied
n_char1=${#1}

if [[ $n_char1 -gt 0 ]]; then
    echo "Argument supplied, running python"
    python3 -m tests.test_univariate_vecdist
    python3 -m tests.test_univariate_gradients
    python3 -m tests.test_univariate_censoring
    python3 -m tests.test_univariate_integration
else
    echo "No argument supplied, running pytest"
    python3 -m pytest tests/test_univariate_vecdist.py
    python3 -m pytest tests/test_univariate_gradients.py
    python3 -m pytest tests/test_univariate_censoring.py 
    python3 -m pytest tests/test_univariate_integration.py
fi


echo "~~~ End of run_pytest_univariate.sh ~~~"