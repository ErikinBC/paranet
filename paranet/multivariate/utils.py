"""
Utility functions to support multivariate functions and classes
"""

# External modules
import numpy as np
# Internal modules
from paranet.utils import not_none, t_wide

def has_args_init(arg1, arg2, init1, init2) -> bool:
    """Checks that either arg1/arg2 are specified or init1/init2 are"""
    has_args = not_none(arg1) & not_none(arg2)
    has_init = not_none(init1) & not_none(init2)
    assert has_args or has_init, 'If arguments are not specified during class initialization, they must be specified as arguments for this function'
    return has_args


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
    has_args = has_args_init(alpha_args, beta_args, alpha_init, beta_init)
    if has_args:
        alpha_beta = np.vstack(t_wide(alpha_args), t_wide(beta_args))
    else:  # Use initialized values
        alpha_beta = np.vstack(t_wide(alpha_init), t_wide(beta_init))
    assert alpha_beta.shape[1] == k, f'alpha and beta need to have {k} columns'
    assert alpha_beta.shape[0] == p + 1, f'alpha needs to a (row) vector and beta needs to be of length {p}'
    return alpha_beta