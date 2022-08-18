"""
GRADIENT METHODS FOR MLL FITTING
"""

# External modules
import numpy as np

# Internal modules
from paranet.utils import format_t_d_scale_shape


def log_lik(t:np.ndarray, d:np.ndarray, scale:np.ndarray, shape:np.ndarray or None, dist:str) -> np.ndarray:
    """
    CALCULATES THE LOG-LIKELIHOOD

    Inputs
    ------
    t:                  A [n,k] or (n,) matrix/array of time-to-event values
    d:                  A [n,k] or (n,) matrix/array of censoring values (1=event, 0=right-censored)
    scale:              See (SurvDists): equivilent to \lambda
    shape:              See (SurvDists): equivilent to \alpha
    dist:               A valid distribution (currently: exponential, weibull, or gompertz)

    Returns
    ------
    ll:                 A (k,) array of log-likelihoods
    """
    t_vec, d_vec, scale, shape = format_t_d_scale_shape(t, d, scale, shape, dist)
    if dist == 'exponential':
        ll = -np.mean(d_vec * np.log(scale) - scale*t_vec, axis=0)
    if dist == 'weibull':
        ll = -np.mean(d_vec*(np.log(shape*scale) + (shape-1)*np.log(t_vec)) - scale*t_vec**shape, axis=0)
    if dist == 'gompertz':
        ll = -np.mean(d_vec*(np.log(scale)+shape*t_vec) - scale/shape*(np.exp(shape*t_vec)-1), axis=0)
    return ll


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
    t_vec, d_vec, scale, shape = format_t_d_scale_shape(t, d, scale, shape, dist)
    if dist == 'exponential':
        dll = -np.mean(d_vec/scale - t_vec, axis=0)
    if dist == 'weibull':
        dll = -np.mean(d_vec/scale - t_vec**shape, axis=0)
    if dist == 'gompertz':
        dll = -np.mean(d_vec/scale - (np.exp(shape*t_vec) - 1)/shape, axis=0)
    return dll


def grad_ll_shape(t:np.ndarray, d:np.ndarray, scale:np.ndarray, shape:np.ndarray or None, dist:str) -> np.ndarray:
    """
    CALCULATES GRADIENT FOR FOR SHAPE PARAMETER

    Inputs
    ------
    See log_like

    Returns
    -------
    dll:               An (k,) or [p-1,k] array/matrix of gradients
    """
    t_vec, d_vec, scale, shape = format_t_d_scale_shape(t, d, scale, shape, dist)
    if dist == 'exponential':
        dll = -np.repeat(0, scale.shape[1]).astype(float)
    if dist == 'weibull':
        dll = -np.mean( d_vec*(1/shape + np.log(t_vec)) - scale*t_vec**shape*np.log(t_vec), axis=0)
    if dist == 'gompertz':
        dll = -np.mean( d_vec*t_vec - (scale/shape**2)*(np.exp(shape*t_vec)*(shape*t_vec-1) +1), axis=0)
    return dll









