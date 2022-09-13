#!/bin/bash

# --------------------------------------- #
# Runs pytest for the elastic net scripts #
# --------------------------------------- #

# Activate conda
conda activate paranet

# Default is to run pytest unless argument is supplied
n_char1=${#1}

if [[ $n_char1 -gt 0 ]]; then
    echo "Argument supplied, running python"
    python3 -m tests.test_elnet_logistic
    python3 -m tests.test_elnet_solver
    python3 -m tests.test_elnet_survset
else
    echo "No argument supplied, running pytest"
    python3 -m pytest tests/test_elnet_logistic.py 
    python3 -m pytest tests/test_elnet_solver.py 
    python3 -m pytest tests/test_elnet_survset.py 
fi


echo "~~~ End of run_pytest_elnet.sh ~~~"