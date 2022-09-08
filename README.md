# Parametric survival models

This repo provides `sklearn`-like classes to carry out (regularized) linear parametric survival regression. 

The five distributions are currently supported:

1. Exponential
2. Weibull
3. Gompertz
4. Lognormal (UNDER-DEVELOPMENT)
5. Log-logistic (UNDER-DEVELOPMENT)

This package was built with a specific conda environment and developers can use `conda env create -f paranet.yml` and then `source paranet`.

See the parameterization section to understand how 

## (1) Probability distribution parameterization

We assume that one parameter from the probability distribution is a function of a linear combination of covariates.

$$
\begin{align}
\eta_i &= x_i^T \beta
\end{align}
$$

## (x) Unittests

The execute all unit tests run `python3 -m pytest` from the main folder. Alternatively, groups of tests can be run using the `run_pytest_{group}.sh`.