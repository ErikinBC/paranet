#!/bin/bash

# Run all the test scripts

echo "--- (i) test_censoring ---"
python3 -m tests.test_censoring

echo "--- (ii) test_vec_dist ---"
python3 -m tests.test_vec_dist

echo "--- (iii) test_gradients ---"
python3 -m tests.test_gradients

echo "--- (iv) test_integration ---"
python3 -m tests.test_integration

echo "--- End of run_all.sh ---"
