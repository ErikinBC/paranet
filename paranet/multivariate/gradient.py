"""
Gradients for multivariate models
"""

# External modules
import numpy as np

# Internal modules
from paranet.utils import dist2idx, broadcast_long, broadcast_dist
from paranet.multivariate.dists import check_multi_input

def process_alphabeta_x_t_dist(alpha_beta:np.ndarray, x:np.ndarray, t:np.ndarray, d:np.ndarray, dist:list or str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Convenience wrapper for putting arguments used by all gradient functions"""
    check_multi_input(alpha_beta, x, t, dist)
    k = alpha_beta.shape[1]
    t = broadcast_long(t, k)
    d = broadcast_long(d, k)
    dist = broadcast_dist(dist, k)
    # Calculate risks
    alpha = alpha_beta[[0]]
    beta = alpha_beta[1:]
    risk = np.exp(x.dot(beta))
    assert risk.shape[1] == k, 'Risk should have column length of k'
    # Return shape, scale, time, and censoring
    return alpha, risk, t, d, k


def log_lik(alpha_beta:np.ndarray, x:np.ndarray, t:np.ndarray, d:np.ndarray, dist:list or str) -> np.ndarray:
    """
    Calculate the (negative) log-likelihood for a given shape/scale, covariate, time, and censoring value

    Inputs
    ------
    alpha_beta:             First row is shape parameters (alpha), second row onwards in scale parameters (should be (p+1,k))
    x:                      The (n,p) design matrix
    t:                      An (n,k) array of time values
    d:                      An (n,k) array of censoring values
    dist:                   The string or list of distributions

    Returns
    -------
    Returns an (k,) array of negative log-likelihoods, where k is the number of distributions
    """
    # Input checks and transforms
    alpha, risk, t, d, k = process_alphabeta_x_t_dist(alpha_beta, x, t, d, dist)

    # negative log-likelihood
    didx = dist2idx(dist)
    ll_vec = np.zeros(k)
    for s, i in didx.items():
        if s == 'exponential':
            ll_vec[i] = -np.mean(d[:,i] * np.log(risk[:,i]) - risk[:,i]*t[:,i], axis=0)
        if s == 'weibull':
            ll_vec[i] = -np.mean(d[:,i]*(np.log(alpha[:,i]*risk[:,i]) + (alpha[:,i]-1)*np.log(t[:,i])) - risk[:,i]*t[:,i]**alpha[:,i], axis=0)
        if s == 'gompertz':
            ll_vec[i] = -np.mean(d[:,i]*(np.log(risk[:,i])+alpha[:,i]*t[:,i]) - risk[:,i]/alpha[:,i]*(np.exp(alpha[:,i]*t[:,i])-1), axis=0)
    return ll_vec


def grad_ll_X(alpha_beta:np.ndarray, x:np.ndarray, t:np.ndarray, d:np.ndarray, dist:list or str) -> np.ndarray:
    """
    Calculate the gradient for the covariates.

    Inputs
    ------
    See log_lik()

    Returns
    -------
    A (p,k) array of gradients for the different scale parameters. Assumes that if there is an intercept it is assigned a column in x
    """
    # Input checks and transforms
    alpha, risk, t, d, k = process_alphabeta_x_t_dist(alpha_beta, x, t, d, dist)
    p = x.shape[1]

    # gradient of the negative log-likelihood
    didx = dist2idx(dist)
    grad_mat = np.zeros([p,k])
    for s, i in didx.items():
        if s == 'exponential':
            grad_i = -np.mean(x*(d[:,i] - t[:,i]*risk[:,i]),0)
        if s == 'weibull':
            grad_i = -np.mean(x*(d[:,i] - t[:,i]**alpha[:,i]*risk[:,i]),0)
        if s == 'gompertz':
            grad_i = -np.mean(x*(d[:,i] - risk[:,i]*(np.exp(alpha[:,i]*t[:,i]) - 1)/alpha[:,i]),0)
        grad_mat[:,i] = grad_i.reshape(grad_mat[:,i].shape)
    return grad_mat


def grad_ll_shape(alpha_beta:np.ndarray, x:np.ndarray, t:np.ndarray, d:np.ndarray, dist:list or str) -> np.ndarray:
    """
    Calculate the gradient for the shape parameters.

    Inputs
    ------
    See log_lik()

    Returns
    -------
    A (1,p) array of gradients for the different shape parameters
    """
    # Input checks and transforms
    alpha, risk, t, d, k = process_alphabeta_x_t_dist(alpha_beta, x, t, d, dist)

    # gradient of the negative log-likelihood
    didx = dist2idx(dist)
    grad_shape = np.zeros([1,k])
    for s, i in didx.items():
        if s == 'exponential':
            grad_i = -np.repeat(0, len(i))
        if s == 'weibull':
            grad_i = -np.mean( d[:,i]*(1/alpha[:,i] + np.log(t[:,i])) - risk[:,i]*t[:,i]**alpha[:,i]*np.log(t[:,i]), axis=0)
        if s == 'gompertz':
            grad_i = -np.mean( d[:,i]*t[:,i] - (risk[:,i]/alpha[:,i]**2)*(np.exp(alpha[:,i]*t[:,i])*(alpha[:,i]*t[:,i]-1) +1), axis=0)
        grad_shape[:,i] = grad_i.reshape(grad_shape[:,i].shape)
    return grad_shape


def grad_ll(alpha_beta:np.ndarray, x:np.ndarray, t:np.ndarray, d:np.ndarray, dist:list or str) -> np.ndarray:
    """
    Calculate the gradient for the shape and covariates parameters.

    Inputs
    ------
    See log_lik()

    Returns
    -------
    A (p+1,k) array of gradients, where the first row is the shape parameters, and the 1:p+1 rows are for the different scale parameters. Assumes that if there is an intercept it is assigned a column in x.
    """
    g_shape = grad_ll_shape(alpha_beta, x, t, d, dist)
    g_scale = grad_ll_X(alpha_beta, x, t, d, dist)
    g = np.vstack([g_shape, g_scale])
    return g
