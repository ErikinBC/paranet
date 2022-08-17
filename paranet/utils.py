"""
UTILITY FUNCTIONS
"""

# External modules
import numpy as np
import pandas as pd

# List of currently supported distributions
dist_valid = ['exponential', 'weibull', 'gompertz']


def check_dist_str(dist:str) -> None:
    """CHECK THAT STRING BELONGS TO VALID DISTRIBUTION"""
    assert isinstance(dist, str)
    assert dist in dist_valid, f'dist must be one of: {", ".join(dist_valid)}'


def is_vector(x:np.ndarray) -> None:
    """CHECKS THAT ARRAY HAS AT MOST POSSIBLE DIMENSION"""
    n_shape = len(x.shape)
    if n_shape <= 1:  # Scale or vector
        check = True
    elif n_shape == 2:
        if x.shape[1] == 1:
            check = False  # Is (k,1)
        else:
            check = False  # Is (p,k), k>1
    else:  # Must have 3 or more dimensions
        check = False
    assert check, 'Dimensionality not as expected'


def get_p_k(t:np.ndarray) -> tuple[int, int]:
    """
    RETURN THE DIMENSIONALITY OF THE DATA INPUT ARRAY

    *NOTE, WHEN WE MOVE TO COVARIATES, INPUT WILL NEED TO CHANGE TO X
    """
    n_shape = len(t.shape)
    assert n_shape <= 2, 'Time-to-event can have at most 2-dimensions'
    if n_shape <= 1:
        k, p = 1, 2
    else:
        k, p = t.shape[1], 2
    return p, k


def shape_scale_2vec(shape_scale:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    SPLIT THE [p,k] matrix into a [1,k] and [p-1,k] row vector/matrix
    """
    shape, scale = t_wide(shape_scale[0]), t_wide(shape_scale[1:])
    return shape, scale


def format_t_d(t:np.ndarray, d:np.ndarray, dist:str) -> tuple[np.ndarray, np.ndarray]:
    """
    ENSURES THAT TIME/CENSORING ARE IN LONG FORM, AND SCALE/SHAPE ARE IN WIDE FORM
    """
    check_dist_str(dist)
    t_vec, d_vec = t_long(t), t_long(d)
    assert t_vec.shape == d_vec.shape, 'time and censoring matrix should be teh same size'
    return t_vec, d_vec


def format_t_d_scale_shape(t:np.ndarray, d:np.ndarray, scale:np.ndarray, shape:np.ndarray or None, dist:str)  -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    ENSURES THAT TIME/CENSORING ARE IN LONG FORM, AND SCALE/SHAPE ARE IN WIDE FORM
    """
    t_vec, d_vec = format_t_d(t, d, dist)
    scale, shape = t_wide(scale), t_wide(shape)
    return t_vec, d_vec, scale, shape


def t_wide(x:np.ndarray or float or pd.Series or None) -> np.ndarray:
    """CONVERT 1D ARRAY OR FLOAT TO A 1xK VECTOR"""
    if x is None:
        return x
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    n_shape = len(x.shape)
    if n_shape == 0:
        x = x.reshape([1, 1])
    if n_shape == 1:
        x = x.reshape([1, max(x.shape)])
    return x


def t_long(x:np.ndarray or float or pd.Series or None) -> np.ndarray:
    """CONVERT 1D ARRAY OR FLOAT TO A Kx1 VECTOR"""
    if x is None:
        return x
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    n_shape = len(x.shape)
    if n_shape == 0:
        x = x.reshape([1, 1])
    if n_shape == 1:
        x = x.reshape([max(x.shape), 1])
    return x


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
