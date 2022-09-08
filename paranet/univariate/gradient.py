"""
GRADIENT METHODS FOR MLL FITTING
"""

# External modules
import warnings
import numpy as np

# Internal modules
from paranet.utils import broadcast_dist, format_t_d_scale_shape, dist2idx

# Supress overflow warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def log_lik(t:np.ndarray, d:np.ndarray, scale:np.ndarray, shape:np.ndarray or None, dist:str) -> np.ndarray:
    """
    CALCULATES THE LOG-LIKELIHOOD

    Inputs
    ------
    t:                  A [n,k] or (n,) matrix/array of time-to-event values
    d:                  A [n,k] or (n,) matrix/array of censoring values (1=event, 0=right-censored)
    scale:              See (SurvDists): equivilent to lambda
    shape:              See (SurvDists): equivilent to alpha
    dist:               A valid distribution (currently: exponential, weibull, or gompertz)

    Returns
    ------
    ll:                 A (k,) array of log-likelihoods
    """
    # Note that this will broadcast t wide if k > 1
    t_vec, d_vec, scale, shape = format_t_d_scale_shape(t, d, scale, shape)
    k = scale.shape[1]
    ll_vec = np.zeros(k)
    # Calculate negative mean of log-likelihood
    dist = broadcast_dist(dist, k)
    didx = dist2idx(dist)
    for d, i in didx.items():
        if d == 'exponential':
            ll_vec[i] = -np.mean(d_vec[:,i] * np.log(scale[:,i]) - scale[:,i]*t_vec[:,i], axis=0)
        if d == 'weibull':
            ll_vec[i] = -np.mean(d_vec[:,i]*(np.log(shape[:,i]*scale[:,i]) + (shape[:,i]-1)*np.log(t_vec[:,i])) - scale[:,i]*t_vec[:,i]**shape[:,i], axis=0)
        if d == 'gompertz':
            ll_vec = -np.mean(d_vec[:,i]*(np.log(scale[:,i])+shape[:,i]*t_vec[:,i]) - scale[:,i]/shape[:,i]*(np.exp(shape[:,i]*t_vec[:,i])-1), axis=0)
    return ll_vec


def grad_ll(t:np.ndarray, d:np.ndarray, scale:np.ndarray, shape:np.ndarray or None, dist:str) -> np.ndarray:
    """
    CALCULATE GRADIENT FOR FOR SHAPE AND SCALE PARAMETER

    Inputs
    ------
    See log_like

    Returns
    -------
    grad:               An [p,k] matrix, where the first row corresponds to the shape parameter 
    """
    grad_alph = grad_ll_shape(t, d, scale, shape, dist)
    grad_lam = grad_ll_scale(t, d, scale, shape, dist)
    # Place shape/alpha in position zero
    grad = np.vstack([grad_alph, grad_lam])
    return grad


def grad_ll_scale(t:np.ndarray, d:np.ndarray, scale:np.ndarray, shape:np.ndarray or None, dist:str) -> np.ndarray:
    """
    CALCULATES GRADIENT FOR FOR SCALE PARAMETER

    Inputs
    ------
    See log_like

    Returns
    -------
    dll:               An (k,) array of gradients
    """
    t_vec, d_vec, scale, shape = format_t_d_scale_shape(t, d, scale, shape)
    k = scale.shape[1]
    dll_vec = np.zeros(k)
    # Calculate negative gradient of log-likelihood
    dist = broadcast_dist(dist, k)
    didx = dist2idx(dist)
    for d, i in didx.items():
        if d == 'exponential':
            dll_vec[i] = -np.mean(d_vec[:,i]/scale[:,i] - t_vec[:,i], axis=0)
        if d == 'weibull':
            dll_vec[i] = -np.mean(d_vec[:,i]/scale[:,i] - t_vec[:,i]**shape[:,i], axis=0)
        if d == 'gompertz':
            dll_vec[i] = -np.mean(d_vec[:,i]/scale[:,i] - (np.exp(shape[:,i]*t_vec[:,i]) - 1)/shape[:,i], axis=0)
    return dll_vec


def grad_ll_shape(t:np.ndarray, d:np.ndarray, scale:np.ndarray, shape:np.ndarray or None, dist:str) -> np.ndarray:
    """
    CALCULATES GRADIENT FOR FOR SHAPE PARAMETER

    Inputs
    ------
    See log_like

    Returns
    -------
    dll:               An (k,) array of gradients
    """
    t_vec, d_vec, scale, shape = format_t_d_scale_shape(t, d, scale, shape)
    k = scale.shape[1]
    dll_vec = np.zeros(k)
    # Calculate negative gradient of log-likelihood
    dist = broadcast_dist(dist, k)
    didx = dist2idx(dist)
    for d, i in didx.items():
        if d == 'exponential':
            dll_vec[i] = -np.repeat(0, len(i))
        if d == 'weibull':
            dll_vec[i] = -np.mean( d_vec[:,i]*(1/shape[:,i] + np.log(t_vec[:,i])) - scale[:,i]*t_vec[:,i]**shape[:,i]*np.log(t_vec[:,i]), axis=0)
        if d == 'gompertz':
            dll_vec[i] = -np.mean( d_vec[:,i]*t_vec[:,i] - (scale[:,i]/shape[:,i]**2)*(np.exp(shape[:,i]*t_vec[:,i])*(shape[:,i]*t_vec[:,i]-1) +1), axis=0)
    return dll_vec









