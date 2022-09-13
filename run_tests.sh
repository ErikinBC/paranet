#!/bin/bash

# - Run all python unit testing scripts - #

source tests/run_pytest_univariate.sh py
source tests/run_pytest_multivariate.sh py
source tests/run_pytest_elnet.sh py
source examples/run_examples.sh

echo "~~~ End of run_tests.sh ~~~"
