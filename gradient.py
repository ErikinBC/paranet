"""
GRADIENT METHODS FOR MLL FITTING
"""

# External modules
import numpy as np

# Internal modules
from utils import format_t_d_scale_shape


def log_lik(t:np.ndarray, d:np.ndarray, scale:np.ndarray, shape:np.ndarray or None, dist:str):
    """
    CALCULATES THE LOG-LIKELIHOOD

    Inputs
    ------
    t:                  time
    d:                  censoring indicator (1=event, 0=right-censored)
    scale:              lambda
    shape:              alpha
    dist:               One of dist_valid
    """
    t_vec, d_vec, scale, shape = format_t_d_scale_shape(t, d, scale, shape, dist)
    if dist == 'exponential':
        ll = np.mean(d_vec * np.log(scale) - scale*t_vec, axis=0)
    if dist == 'weibull':
        ll = np.mean(d_vec*(np.log(shape*scale) + (shape-1)*np.log(t_vec)) - scale*t_vec**shape, axis=0)
    if dist == 'gompertz':
        ll = np.mean(d_vec*np.log(scale) - scale/shape*(np.exp(shape*t_vec)-1), axis=0)
    return ll


def grad_ll(t:np.ndarray, d:np.ndarray, scale:np.ndarray, shape:np.ndarray or None, dist:str):
    """
    WRAPPER TO CALCULATE GRADIENT FOR FOR SHAPE AND SCALE PARAMETER
    """
    1

def grad_ll_scale(t:np.ndarray, d:np.ndarray, scale:np.ndarray, shape:np.ndarray or None, dist:str):
    """
    CALCULATES GRADIENT FOR FOR SCALE PARAMETER
    """
    t_vec, d_vec, scale, shape = format_t_d_scale_shape(t, d, scale, shape, dist)
    if dist == 'exponential':
        dll = np.mean(d_vec/scale - t_vec, axis=0)
    if dist == 'weibull':
        dll = np.mean(d_vec/scale - t_vec**shape, axis=0)
    if dist == 'gompertz':
        dll = np.mean(d_vec/scale - np.exp(shape*t_vec)/shape + 1/shape, axis=0)
    return dll

def grad_ll_shape(t:np.ndarray, d:np.ndarray, scale:np.ndarray, shape:np.ndarray or None, dist:str):
    """
    CALCULATES GRADIENT FOR FOR SHAPE PARAMETER
    """
    t_vec, d_vec, scale, shape = format_t_d_scale_shape(t, d, scale, shape, dist)
    if dist == 'exponential':
        dll = 0
    if dist == 'weibull':
        dll = np.mean( d_vec*(1/shape + np.log(t_vec)) - scale*t_vec**shape*np.log(t_vec), axis=0)
    if dist == 'gompertz':
        dll = np.mean(scale/shape*(np.exp(shape*t_vec)*(1/shape-t_vec) - 1/shape), axis=0)
    return dll









