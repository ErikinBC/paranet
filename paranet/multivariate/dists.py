"""
Creates distribution/gradient specific functions for the multivariate parametric likelihoods
"""

# External modules
import numpy as np

# Internal modules
from paranet.utils import broadcast_dist, broadcast_long, dist2idx, check_interval, t_long


def check_multi_input(alpha_beta:np.ndarray, x:np.ndarray, t:np.ndarray, dist:list) -> None:
    """Check that inputs align to dimensional expectations"""
    p_ab, k_ab = alpha_beta.shape
    n, p = x.shape
    dist = broadcast_dist(dist, k_ab)
    n_t = len(t)
    assert k_ab == len(dist), 'Number of columns of alpha_beta needs to align with the length of dist'
    assert p_ab - 1 == p, 'Number of columns of x should be one less that the number of rows of alpha_beta (as the first row is the alpha/shape parameter)'
    assert n_t == n, 'The number of rows of x need to align with t'
    if len(t.shape) == 2:
        k_t = t.shape[1]
        if k_t != k_ab:
            assert k_t == 1, 'If length of dist is not equal to number of columns of t, then t must have a single column so it can be broadcast'


def hazard_multi(alpha_beta:np.ndarray, x:np.ndarray, t:np.ndarray, dist:list or str) -> np.ndarray:
    """
    Calculates the hazard function for the relevant classes with a matrix of covariates

    Inputs
    ------
    alpha_beta:         Matrix of coefficients ~ (p+1,k), where first row is the shape/alpha
    x:                  Matrix of covariates x~(n,p), where the first column is usually a one to indicate an intercept
    t:                  Time vector t~(n,1) or (n,k)
    dist:               A list of distributions to fit covariates for (length 1 or k)

    Returns
    -------
    An (n,k) matrix of hazards, where each column corresponds to a different distribution
    """
    # Input checks and transforms
    check_multi_input(alpha_beta, x, t, dist)
    k = alpha_beta.shape[1]
    n = x.shape[0]
    t = broadcast_long(t, k)
    dist = broadcast_dist(dist, k)

    # Calculate risks
    alpha = alpha_beta[[0]]
    beta = alpha_beta[1:]
    risk = np.exp(x.dot(beta))

    # Calculate hazard
    didx = dist2idx(dist)
    h_mat = np.zeros([n, k])
    
    for d, i in didx.items():
        if d == 'exponential':
            h_mat[:,i] = risk[:,i]
        if d == 'weibull':
            h_mat[:,i] = risk[:,i] * alpha[:,i] * t[:,i]**(alpha[:,i]-1)
        if d == 'gompertz':
            h_mat[:,i] = risk[:,i] * np.exp(alpha[:,i]*t[:,i])
    return h_mat



def survival_multi(alpha_beta:np.ndarray, x:np.ndarray, t:np.ndarray, dist:list or str) -> np.ndarray:
    """
    Calculates the survival function for the relevant classes with a matrix of covariates; see hazard_multi() for more details
    """
    # Input checks and transforms
    check_multi_input(alpha_beta, x, t, dist)
    k = alpha_beta.shape[1]
    n = x.shape[0]
    t = broadcast_long(t, k)
    dist = broadcast_dist(dist, k)

    # Calculate risks
    alpha = alpha_beta[[0]]
    beta = alpha_beta[1:]
    risk = np.exp(x.dot(beta))

    # Calculate survival
    didx = dist2idx(dist)
    s_mat = np.zeros([n, k])
    for d, i in didx.items():
        if d == 'exponential':
            s_mat[:,i] = np.exp(-risk[:,i] * t[:,i])
        if d == 'weibull':
            s_mat[:,i] = np.exp(-risk[:,i] * t[:,i]**alpha[:,i])
        if d == 'gompertz':
            s_mat[:,i] = np.exp(-risk[:,i]/alpha[:,i] * (np.exp(alpha[:,i]*t[:,i])-1))
    return s_mat


def pdf_multi(alpha_beta:np.ndarray, x:np.ndarray, t:np.ndarray, dist:list or str) -> np.ndarray:
    """
    Calculates the density function for the relevant classes with a matrix of covariates; see hazard_multi() for more details
    """
    h = hazard_multi(alpha_beta, x, t, dist)
    s = survival_multi(alpha_beta, x, t, dist)
    f = h * s
    return f


def broadcast_percentile(percentile:np.ndarray or float, n:int, k:int):
    """
    Broadcast a float or array to an n

    Inputs
    ------
    p:              Percentile input argument
    n:              Length
    k:              Number of columns
    """
    check_interval(percentile, 0, 1, equals=False)
    if isinstance(percentile, float):
        return np.zeros([n,k]) + percentile
    else:
        if not isinstance(percentile, np.ndarray):
            percentile = np.array(percentile)
        if len(percentile.shape) == 1:
            assert len(percentile) == n, 'If percentile is flat, it should be of length n'
            return np.tile(t_long(percentile), [1,k])
        else:
            assert percentile.shape == (n, k), 'If percentile has two dimensions, it should match (n,k)'
            return percentile


def quantile_multi(percentile:np.ndarray, alpha_beta:np.ndarray, x:np.ndarray, dist:list or str) -> np.ndarray:
    """
    Calculates the quantiles for the relevant classes with a matrix of covariates; see hazard_multi() for more details
    """
    # Input checks and transforms
    k, n = alpha_beta.shape[1], x.shape[0]
    percentile = broadcast_percentile(percentile, n, k)
    check_multi_input(alpha_beta, x, percentile, dist)
    dist = broadcast_dist(dist, k)

    # Calculate risks
    alpha = alpha_beta[[0]]
    beta = alpha_beta[1:]
    risk = np.exp(x.dot(beta))

    # Calculate quantile
    nlp = -np.log(1 - percentile)
    q_mat = np.zeros([n, k])
    didx = dist2idx(dist)
    for d, i in didx.items():
        if d == 'exponential':
            q_mat[:,i] = nlp[:,i] / risk[:,i]
        if d == 'weibull':
            q_mat[:,i] = (nlp[:,i] / risk[:,i]) ** (1/alpha[:,i])
        if d == 'gompertz':
            q_mat[:,i] = 1/alpha[:,i] * np.log(1 + alpha[:,i]/risk[:,i]*nlp[:,i])
    return q_mat

