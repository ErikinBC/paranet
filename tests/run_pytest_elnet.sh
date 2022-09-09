#!/bin/bash

# Activate conda
conda activate paranet

python3 -m pytest tests/test_elnet_logistic.py 
python3 -m pytest tests/test_elnet_solver.py 
# python3 -m pytest tests/test_elnet_survset.py 