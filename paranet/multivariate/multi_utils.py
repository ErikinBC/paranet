"""
Utility functions to support multivariate functions and classes
"""

# External modules
import numpy as np
# Internal modules
from paranet.utils import not_none, t_wide

def has_args_init(arg1, init1, arg2, init2) -> bool:
    """Checks that is at least one value for arg{i} or init{i}"""
    # If argument exists, it willl be used
    has_arg1, has_arg2 = not_none(arg1), not_none(arg2)
    assert has_arg1 or not_none(init1), 'if argument1 is not supplied, init1 must exist'
    assert has_arg2 or not_none(init2), 'if argument2 is not supplied, init2 must exist'
    return has_arg1, has_arg2


def args_alpha_beta(k:int, p:int, alpha_args:np.ndarray or None=None, beta_args:np.ndarray or None=None, alpha_init:np.ndarray or None=None, beta_init:np.ndarray or None=None) -> np.ndarray:
    """
    Process the shape/scale parameters that provided to one of the parametric methods (e.g. hazard)

    Inputs
    ------
    k:                  Length/# of cols of alpha/beta
    p:                  Number of columns of design matrix x
    alpha_args:         Shape parameters to be evaluated by function call
    beta_args:          Scale parameters to be evaluated by function call
    alpha_init:         Shape parameters specified at class initialization
    beta_init:          Scale parameters specified at class initialization

    Returns
    -------
    A (p+1,k) array of weights
    """
    has_args = has_args_init(alpha_args, alpha_init, beta_args, beta_init)
    if has_args:
        alpha_beta = np.vstack((t_wide(alpha_args), t_wide(beta_args)))
    else:  # Use initialized values
        alpha_beta = np.vstack((t_wide(alpha_init), t_wide(beta_init)))
    assert alpha_beta.shape[1] == k, f'alpha and beta need to have {k} columns'
    assert alpha_beta.shape[0] == p + 1, f'alpha needs to a (row) vector and beta needs to be of length {p}'
    return alpha_beta