"""
UTILITY FUNCTIONS
"""

# External modules
import numpy as np
import pandas as pd


def t_wide(x:np.ndarray or float or pd.Series or None) -> np.ndarray:
    """CONVERT 1D ARRAY OR FLOAT TO A 1xK VECTOR"""
    if x is None:
        return x
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if len(x.shape) > 1:
        return x
    return x.reshape([1, max(x.shape)])


def t_long(x:np.ndarray or float or pd.Series or None) -> np.ndarray:
    """CONVERT 1D ARRAY OR FLOAT TO A Kx1 VECTOR"""
    return t_wide(x).T


def len_of_none(x:np.ndarray or None) -> int:
    """Calculate length of array, or return 0 for nonetype"""
    l = 0
    if x is not None:
        l = len(x)
    return l


def param2array(x:float or np.ndarray or pd.Series) -> np.ndarray:
    """
    Checks that the input parameters to a distribution are either a float or a coercible np.ndarray
    """
    check = True
    # Check for conceivable floats
    lst_float = [float, np.float32, np.float64, int, np.int32, np.int64]
    if type(x) in lst_float:
        x = np.array([x])
    elif isinstance(x, list):
        x = np.array(x, dtype=float)
    elif isinstance(x, pd.Series):
        x = np.array(x, dtype=float)
    elif isinstance(x, np.ndarray):
        if len(x.shape) == 2:
            x = x.flatten()
    elif x is None:
        x = None
    else:
        check = False
    assert check, 'Input is not a float or coerible'
    return x
