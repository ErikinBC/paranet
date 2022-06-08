# Parametric survival models

This repo provides `sklearn`-like classes to carry out (regularized) linear parametric survival regression.

The five distributions are currently supported:

1. Exponential
2. Weibull
3. Gompertz
4. Lognormal
5. Log-logistic

See the parameterization section to understand how 

## (1) Probability distribution parameterization

We assume that one parameter from the probability distribution is a function of a linear combination of covariates.

$$
\begin{align}
\eta_i &= x_i^T \beta
\end{align}
$$
